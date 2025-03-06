"""
Embarassingly simple (should I have written it in bash?) script
for turning conll-formatted files to sentence-per-line
whitespace-tokenized files.

Takes the filepath at sys.argv[1]; writes to stdout

John Hewitt, 2019
"""

import sys
import argparse
import re

argp = argparse.ArgumentParser()
argp.add_argument('input_conll_filepath')
argp.add_argument('--use_chinese', dest='use_chinese', action='store_true', default=False)
argp.add_argument('--use_translit', dest='use_translit', action='store_true', default=False)
args = argp.parse_args()

def generate_lines_for_sent(lines, skip_lines=False):
  '''Yields batches of lines describing a sentence in conllx.

  Args:
    lines: Each line of a conllx file.
  Yields:
    a list of lines describing a single sentence in conllx.
  '''
  buf = []
  obvs_idx = 0
  for index, line in enumerate(lines):
    if line.startswith('#'):
      # if 'sent_id' in line:
        # print(line)
      continue
    if not line.strip():
      if buf:
        yield buf
        buf = []
        obvs_idx += 1
      else:
        continue
    else:
      buf.append(line.strip())
  if buf:
    yield buf

def remove_ranges(lines, head_index):
  import copy
  lines = copy.deepcopy(lines)
  index_mappings = {'0': '0'}
  for index in range(len(lines)):
    if index >= len(lines): break
    line = lines[index]
    if '-' in line[0]:
      l, r = [int(x) for x in line[0].split('-')]
      width = r - l + 1
      newLine = lines[index+1]         # copy all data but lemma, index from the first word in the range
      newLine[0] = str(l)              # the new index is the first index of the range
      newLine[1] = line[1]             # copy the lemma of the entire fused range
      possibleIndices = [l[head_index] for l in lines[index+1:index+1+width]]

      # we only keep head indices that aren't within the range
      toUse = [x for x in possibleIndices if not (l <= int(x) <= r)]
      toUse = list(dict.fromkeys(toUse))
      if len(toUse) == 0:
        print(lines[index])
        raise AssertionError
      newLine[head_index] = toUse[0] if len(toUse) == 1 else toUse

      lines[index] = newLine
      del lines[index+1:index+1+width]
      for i in range(l, r + 1):
        index_mappings[str(i)] = str(index + 1)
    else:
      index_mappings[lines[index][0]] = str(index + 1)
    lines[index][0] = str(index + 1)

  def toMapping(x):
    if isinstance(x, list): return [index_mappings[y] for y in x]
    return index_mappings[x]

  for i, line in enumerate(lines):
    line[head_index] = toMapping(line[head_index])
  return lines


buf = []
toRemove = 0
joiningString = ' ' if not args.use_chinese else ''
sentences = []
with open(args.input_conll_filepath, 'r') as f:
  lines = f.readlines()
head_index = 6
for buf in generate_lines_for_sent(lines):
      conllx_lines = []
      for line in buf:
        conllx_lines.append(line.strip().split('\t'))
      conllx_lines = [x for x in conllx_lines if '.' not in x[0]]
      conllx_lines = remove_ranges(conllx_lines, head_index)

      data = list(zip(*conllx_lines))

      head_indices = list(data[head_index])

      # resolve ambiguities that arise with multiwords
      for i, indices in enumerate(head_indices, 1):
        if not isinstance(indices, list): continue # nothing to be resolved
        indices = list(dict.fromkeys(indices))    # remove duplicates; can't use set because want to preserve order
        if len(indices) == 0:
          raise AssertionError
        if '0' in indices:
          head_indices[i-1] = '0'
          continue
        for idx in indices:
          if (head_indices[int(idx)-1] == str(i) or
             (isinstance(head_indices[int(idx)-1], list) and str(i) in head_indices[int(idx)-1])):
            indices.remove(idx)
        if len(indices) == 1:
          head_indices[i-1] = indices[0]
        elif len(indices) == 0:
          raise AssertionError
        else:
          # print("Remaining ambiguity found", len(indices), conllx_lines[i-1])
          head_indices[i-1] = indices[-1]
      data[head_index] = tuple(head_indices)
      for x in head_indices:
        assert(isinstance(x, str)), (data, x)
      if len(data) > 1:
        sys.stdout.write(' '.join(data[1]) + '\n')
