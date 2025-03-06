import torch
import math
import torch.nn.functional as F
from torch import nn

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
    """
    Dropout add function

    Args:
        x (`torch.tensor`):
            input tensor
        residual (`torch.tensor`):
            residual tensor
        prob (`float`):
            dropout probability
        training (`bool`):
            training mode
    """
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out

class SelfAttentionWrapper(nn.Module):
    def __init__(self, original_self_attn, layer_idx, device):
        super().__init__()
        self.original_self_attn = original_self_attn
        self.layer_idx = self.original_self_attn.layer_idx
        self.device = device
        self.q_proj = self.original_self_attn.q_proj
        self.k_proj = self.original_self_attn.k_proj
        self.v_proj = self.original_self_attn.v_proj
        self.o_proj = self.original_self_attn.o_proj

        self.num_heads = self.original_self_attn.num_heads
        self.head_dim = self.original_self_attn.head_dim

        self.rotary_emb = self.original_self_attn.rotary_emb

        self.attention_dropout = self.original_self_attn.attention_dropout
        self.training = self.original_self_attn.training


        self.num_key_value_heads = self.original_self_attn.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.hidden_size = self.original_self_attn.hidden_size


    def forward(self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_value = None,
        cache_position = None, **kwargs):
       
        #breakpoint()
        batch_size, seq_len, _ = hidden_states.size()
        q_projection = self.q_proj(hidden_states)
        k_projection = self.k_proj(hidden_states)
        v_projection = self.v_proj(hidden_states)

        q_projection = q_projection.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_projection = k_projection.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v_projection = v_projection.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self.original_self_attn, "past_key_value", past_key_value)
        # positions_ids = kwargs['positions_ids']
        # cos, sin = self.rotary_emb(v_projection, position_ids)
        cos, sin = position_embeddings
        q_projection, k_projection = apply_rotary_pos_emb(q_projection, k_projection, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; position_ids needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k_, v_projection = past_key_value.update(k_projection, v_projection, self.layer_idx, cache_kwargs)

        k_projection = repeat_kv(k_projection, self.num_key_value_groups)
        v_projection = repeat_kv(v_projection, self.num_key_value_groups)
        

        attn_weights = torch.matmul(q_projection, k_projection.transpose(2, 3))/ math.sqrt(self.head_dim) # batch, n_heads, seq_len, seq_len
        #breakpoint()
        delta_attn_weights = torch.matmul(q_projection.transpose(2, 3).unsqueeze(-1), k_projection.transpose(2, 3).unsqueeze(-1).transpose(-2, -1)) # batch, n_heads, head_dim, seq_len, seq_len
        #breakpoint()

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask
            if cache_position is not None:
                causal_mask = attention_mask[:, :, cache_position, : k_projection.shape[-2]]
            attn_weights = attn_weights + causal_mask
        
        attn_diff_weights = (attn_weights.unsqueeze(2).expand(-1, -1, q_projection.shape[-1], -1, -1)-delta_attn_weights)/math.sqrt(self.head_dim)
        #breakpoint()

        attn_diff_weights = F.softmax(attn_diff_weights, dim=-1, dtype=torch.float32).to(q_projection.dtype)
        #breakpoint()

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_projection.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, v_projection)
        #breakpoint()

        qk_importance = attn_diff_weights-attn_weights.unsqueeze(2).expand(-1,-1,q_projection.shape[-1],-1,-1) # batch, n_heads, head_dim, seq_len, seq_len
        qk_importance = qk_importance**2
        qk_importance = qk_importance.sum(dim=(-2,-1)).sum(dim=0).view(-1) # n_head*head_dim
        #breakpoint()

        self.query_importance = qk_importance.detach().cpu().float().unsqueeze(0)
        self.key_importance = qk_importance.detach().cpu().float().unsqueeze(0)
        #breakpoint()

        attn_output = attn_output.transpose(1, 2).contiguous()
        # attn_output_temp = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        #breakpoint()

        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)

        v_importance = torch.abs(attn_output).sum(dim=1).sum(dim=0)
        #breakpoint()


        self.value_importance = v_importance.detach().cpu().float().unsqueeze(0)
        #breakpoint()

        
        attn_output_o = attn_output
        attn_output = self.o_proj(attn_output)


        o_importance = torch.sum(torch.abs(attn_output_o), dim=1).sum(dim=0)
        
        self.output_importance = o_importance.detach().cpu().float().unsqueeze(0)
        #breakpoint()

        torch.cuda.empty_cache()
        return attn_output, attn_weights, past_key_value


class BloomSelfAttentionWrapper(nn.Module):
    def __init__(self, original_self_attn, layer_idx, device):
        super().__init__()
        self.original_self_attn = original_self_attn
        self.layer_idx = self.original_self_attn.layer_idx
        self.device = device

        self.query_key_value = self.original_self_attn.query_key_value

        self.num_heads = self.original_self_attn.num_heads
        self.head_dim = self.original_self_attn.head_dim


        self.attention_dropout = self.original_self_attn.attention_dropout
        self.training = self.original_self_attn.training


        self.hidden_size = self.original_self_attn.hidden_size

        self.beta = self.original_self_attn.beta
        self.inv_norm_factor = self.original_self_attn.inv_norm_factor
        self.dense = self.original_self_attn.dense
        self.hidden_dropout = self.original_self_attn.hidden_dropout

    def _reshape(self, fused_qkv):
        """
        Split the last dimension into (num_heads, head_dim) and reshapes to (bs, heads, len, dim) shape
        without making any copies, results share same memory storage as `fused_qkv`

        Args:
            fused_qkv (`torch.tensor`): [batch_size, seq_length, num_heads * 3 * head_dim]

        Returns:
            query: [batch_size, num_heads, seq_length, head_dim]
            key: [batch_size, num_heads, seq_length, head_dim]
            value: [batch_size, num_heads, seq_length, head_dim]
        """
        batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
        fused_qkv = fused_qkv.view(batch_size, seq_length, self.num_heads, 3, self.head_dim)
        query_layer = fused_qkv[..., 0, :].transpose(1, 2)
        key_layer = fused_qkv[..., 1, :].transpose(1, 2)
        value_layer = fused_qkv[..., 2, :].transpose(1, 2)
        return query_layer, key_layer, value_layer
    
    def _merge_heads(self, x):
        """
        Merge heads together over the last dimension

        Args:
            x (`torch.tensor`): [batch_size * num_heads, seq_length, head_dim]

        Returns:
            torch.tensor: [batch_size, seq_length, num_heads * head_dim]
        """
        # What we want to achieve is:
        # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.num_heads

        # First view to decompose the batch size
        # batch_size * num_heads, seq_length, head_dim -> batch_size, num_heads, seq_length, head_dim
        x = x.view(batch_size, self.num_heads, seq_length, self.head_dim)

        # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
        x = x.permute(0, 2, 1, 3)

        # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
        return x.reshape(batch_size, seq_length, self.num_heads * self.head_dim)

    def forward(self,
        hidden_states,
        residual,
        alibi,
        attention_mask,
        layer_past = None,
        head_mask = None,
        use_cache = False,
        output_attentions = False,
        cache_position = None):
       

        batch_size, q_length, _ = hidden_states.shape
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
        # 3 x [batch_size, num_heads, seq_length, head_dim]
        query_layer, key_layer, value_layer = self._reshape(fused_qkv)

        if layer_past is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_layer, value_layer = layer_past.update(key_layer, value_layer, self.layer_idx, cache_kwargs)
        # breakpoint()
        # reshape qkv for further computations
        query_layer = query_layer.reshape(batch_size * self.num_heads, -1, self.head_dim)
        key_layer = key_layer.reshape(batch_size * self.num_heads, -1, self.head_dim)
        value_layer = value_layer.reshape(batch_size * self.num_heads, -1, self.head_dim)

        # [batch_size * num_heads, q_length, kv_length]
        attention_scores = alibi.baddbmm(
            batch1=query_layer,
            batch2=key_layer.transpose(-2, -1),
            beta=self.beta,
            alpha=self.inv_norm_factor,
        )

        # breakpoint()
        # batch_size*num_heads, 1, seq_length
        delta_alibi = alibi.unsqueeze(1).expand(-1, query_layer.shape[-1], -1, -1).reshape(alibi.shape[0]*query_layer.shape[-1], alibi.shape[-2], alibi.shape[-1])
        delta_attention_scores = delta_alibi.baddbmm(
            batch1=query_layer.transpose(1, 2).unsqueeze(-1).reshape(query_layer.shape[0]*query_layer.shape[-1], query_layer.shape[-2], -1), 
            batch2=key_layer.transpose(1, 2).unsqueeze(-1).reshape(key_layer.shape[0]*key_layer.shape[-1], key_layer.shape[-2], -1).transpose(-2, -1),
            beta=self.beta,
            alpha=self.inv_norm_factor,
            ) # batch*n_heads, head_dim, seq_len, seq_len

        # change view to [batch_size, num_heads, q_length, kv_length]
        attn_weights = attention_scores.view(batch_size, self.num_heads, q_length, -1)

        # change view to [batch_size, num_heads, head_dim, q_length, kv_legnth]
        # breakpoint()
        delta_attn_weights = delta_attention_scores.view(batch_size, self.num_heads, self.head_dim, q_length, -1)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_layer.shape[-1]]
            attn_weights = attn_weights + causal_mask
        
        attn_diff_weights = (attn_weights.unsqueeze(2).expand(-1, -1, query_layer.shape[-1], -1, -1)-delta_attn_weights)/math.sqrt(self.head_dim)
        #breakpoint()

        attn_diff_weights = F.softmax(attn_diff_weights, dim=-1, dtype=torch.float32).to(query_layer.dtype)
        #breakpoint()

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_layer.dtype)
        attn_weights = self.attention_dropout(attn_weights)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
            attn_diff_weights = attn_diff_weights * head_mask.unsqueeze(2).expand(-1, -1, query_layer.shape[-1], -1, -1)
        #breakpoint()

        qk_importance = attn_diff_weights-attn_weights.unsqueeze(2).expand(-1,-1,query_layer.shape[-1],-1,-1) # batch, n_heads, head_dim, seq_len, seq_len
        qk_importance = qk_importance**2
        qk_importance = qk_importance.sum(dim=(-2,-1)).sum(dim=0).view(-1) # n_head*head_dim

        self.query_importance = qk_importance.detach().cpu().float().unsqueeze(0)
        self.key_importance = qk_importance.detach().cpu().float().unsqueeze(0)


        # change view [batch_size x num_heads, q_length, kv_length]
        attn_weights_reshaped = attn_weights.view(batch_size * self.num_heads, q_length, -1)

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = torch.bmm(attn_weights_reshaped, value_layer)
        # change view [batch_size, q_length, num_heads * head_dim]
        context_layer = self._merge_heads(context_layer)

        self.value_importance = torch.sum(torch.abs(context_layer), dim=1).sum(dim=0).detach().cpu().float().unsqueeze(0)

        
        output_tensor = self.dense(context_layer)

        output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)
        
        self.output_importance = torch.sum(torch.abs(output_tensor), dim=1).sum(dim=0).detach().cpu().float().unsqueeze(0)

        outputs = (output_tensor, layer_past)
        if output_attentions:
            outputs += (attention_probs,)
        torch.cuda.empty_cache()
        return outputs
