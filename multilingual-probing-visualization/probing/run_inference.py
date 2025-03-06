"""Loads configuration yaml and runs an experiment."""
from argparse import ArgumentParser
import os
from datetime import datetime
import shutil
import yaml
from tqdm import tqdm
import torch
import numpy as np
import pickle
from reporter import UnionFind

import data
from data import GPTInferenceDataset
import model
import probe
import regimen
import reporter
import task
import loss
import string

def choose_task_classes(args):
  """Chooses which task class to use based on config.

  Args:
    args: the global config dictionary built by yaml.
  Returns:
    A class to be instantiated as a task specification.
  """
  if args['probe']['task_name'] == 'parse-distance':
    task_class = task.ParseDistanceTask
    reporter_class = reporter.WordPairReporter
    if args['probe_training']['loss'] == 'L1':
      loss_class = loss.L1DistanceLoss
    else:
      raise ValueError("Unknown loss type for given probe type: {}".format(
        args['probe_training']['loss']))
  elif args['probe']['task_name'] == 'parse-depth':
    task_class = task.ParseDepthTask
    reporter_class = reporter.WordReporter
    if args['probe_training']['loss'] == 'L1':
      loss_class = loss.L1DepthLoss
    else:
      raise ValueError("Unknown loss type for given probe type: {}".format(
        args['probe_training']['loss']))
  elif args['probe']['task_name'] == 'semantic-roles':
    task_class = task.SemanticRolesTask
    reporter_class = reporter.WordReporter
    if args['probe_training']['loss'] == 'CrossEntropy':
      loss_class = loss.CrossEntropyLoss
    else:
      raise ValueError("Unknown loss type for given probe type: {}".format(
        args['probe_training']['loss']))
  else:
    raise ValueError("Unknown probing task type: {}".format(
      args['probe']['task_name']))
  return task_class, reporter_class, loss_class

def choose_dataset_class(args):
  """Chooses which dataset class to use based on config.

  Args:
    args: the global config dictionary built by yaml.
  Returns:
    A class to be instantiated as a dataset.
  """
  if args['model']['model_type'] in {'ELMo-disk', 'ELMo-random-projection', 'ELMo-decay'}:
    dataset_class = data.ELMoDataset
  elif args['model']['model_type'] == 'BERT-disk':
    dataset_class = data.BERTDataset
  elif args['model']['model_type'] == 'GPT-disk':
    dataset_class = data.GPTDataset
  else:
    raise ValueError("Unknown model type for datasets: {}".format(
      args['model']['model_type']))

  return dataset_class

def choose_probe_class(args):
  """Chooses which probe and reporter classes to use based on config.

  Args:
    args: the global config dictionary built by yaml.
  Returns:
    A probe_class to be instantiated.
  """
  if args['probe']['task_signature'] == 'word':
    if args['probe']['psd_parameters']:
      return probe.OneWordPSDProbe
    else:
      return probe.OneWordNonPSDProbe
  elif args['probe']['task_signature'] == 'word_pair':
    if args['probe']['psd_parameters']:
      return probe.TwoWordPSDProbe
    else:
      return probe.TwoWordNonPSDProbe
  elif args['probe']['task_signature'] == 'word_label':
    if 'probe_spec' not in args['probe'] or args['probe']['probe_spec']['probe_hidden_layers'] == 0:
      return probe.OneWordLinearLabelProbe
    else:
      return probe.OneWordNNLabelProbe
  else:
    raise ValueError("Unknown probe type (probe function signature): {}".format(
      args['probe']['task_signature']))

def choose_model_class(args):
  """Chooses which reporesentation learner class to use based on config.

  Args:
    args: the global config dictionary built by yaml.
  Returns:
    A class to be instantiated as a model to supply word representations.
  """
  if args['model']['model_type'] == 'ELMo-disk':
    return model.DiskModel
  elif args['model']['model_type'] == 'BERT-disk':
    return model.DiskModel
  elif args['model']['model_type'] == 'GPT-disk':
    return model.DiskModel
  elif args['model']['model_type'] == 'ELMo-random-projection':
    return model.ProjectionModel
  elif args['model']['model_type'] == 'ELMo-decay':
    return model.DecayModel
  elif args['model']['model_type'] == 'pytorch_model':
    raise ValueError("Using pytorch models for embeddings not yet supported...")
  else:
    raise ValueError("Unknown model type: {}".format(
      args['model']['model_type']))

def run_train_probe(args, probe, dataset, model, loss, reporter, regimen):
  """Trains a structural probe according to args.

  Args:
    args: the global config dictionary built by yaml.
          Describes experiment settings.
    probe: An instance of probe.Probe or subclass.
          Maps hidden states to linguistic quantities.
    dataset: An instance of data.SimpleDataset or subclass.
          Provides access to DataLoaders of corpora.
    model: An instance of model.Model
          Provides word representations.
    reporter: An instance of reporter.Reporter
          Implements evaluation and visualization scripts.
  Returns:
    None; causes probe parameters to be written to disk.
  """
  regimen.train_until_convergence(probe, model, loss,
      dataset.get_train_dataloader(), dataset.get_dev_dataloader())


def run_report_results(args, probe, dataset, model, loss, reporter, regimen):
  """
  Reports results from a structural probe according to args.
  By default, does so only for dev set.
  Requires a simple code change to run on the test set.
  """
  probe_params_path = os.path.join(args['reporting']['root'],args['probe']['params_path'])

  dev_dataloader = dataset.get_dev_dataloader()
  try:
    probe.load_state_dict(torch.load(probe_params_path))
    probe.eval()
    dev_predictions = regimen.predict(probe, model, dev_dataloader)
  except FileNotFoundError:
    print("No trained probe found.")
    dev_predictions = None

  reporter(dev_predictions, probe, model, dev_dataloader, 'dev')

  #train_dataloader = dataset.get_train_dataloader(shuffle=False)
  #train_predictions = regimen.predict(probe, model, train_dataloader)
  #reporter(train_predictions, train_dataloader, 'train')

  # Uncomment to run on the test set
  test_dataloader = dataset.get_test_dataloader()
  test_predictions = regimen.predict(probe, model, test_dataloader, True)
  reporter(test_predictions, probe, model, test_dataloader, 'test')

def execute_experiment(args, train_probe, report_results):
  """
  Execute an experiment as determined by the configuration
  in args.

  Args:
    train_probe: Boolean whether to train the probe
    report_results: Boolean whether to report results
  """
  dataset_class = choose_dataset_class(args)
  task_class, reporter_class, loss_class = choose_task_classes(args)
  probe_class = choose_probe_class(args)
  model_class = choose_model_class(args)
  regimen_class = regimen.ProbeRegimen

  task = task_class()
  expt_dataset = dataset_class(args, task)
  expt_reporter = reporter_class(args)
  expt_probe = probe_class(args)
  expt_model = model_class(args)
  expt_regimen = regimen_class(args)
  expt_loss = loss_class(args)

  if train_probe:
    print('Training probe...')
    run_train_probe(args, expt_probe, expt_dataset, expt_model, expt_loss, expt_reporter, expt_regimen)
  if report_results:
    print('Reporting results of trained probe...')
    run_report_results(args, expt_probe, expt_dataset, expt_model, expt_loss, expt_reporter, expt_regimen)


def setup_new_experiment_dir(args, yaml_args, reuse_results_path):
  """Constructs a directory in which results and params will be stored.

  If reuse_results_path is not None, then it is reused; no new
  directory is constrcted.

  Args:
    args: the command-line arguments:
    yaml_args: the global config dictionary loaded from yaml
    reuse_results_path: the (optional) path to reuse from a previous run.
  """
  now = datetime.now()
  date_suffix = '-'.join((str(x) for x in [now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond]))
  model_suffix = '-'.join((yaml_args['model']['model_type'], yaml_args['probe']['task_name']))
  if reuse_results_path:
    new_root = reuse_results_path
    tqdm.write('Reusing old results directory at {}'.format(new_root))
    if args.train_probe == -1:
      args.train_probe = 0
      tqdm.write('Setting train_probe to 0 to avoid squashing old params; '
          'explicitly set to 1 to override.')
  else:
    new_root = os.path.join(yaml_args['reporting']['root'], model_suffix + '-' + date_suffix +'/' )
    tqdm.write('Constructing new results directory at {}'.format(new_root))
  yaml_args['reporting']['root'] = new_root
  os.makedirs(new_root, exist_ok=True)
  try:
    shutil.copyfile(args.experiment_config, os.path.join(yaml_args['reporting']['root'],
      os.path.basename(args.experiment_config)))
  except shutil.SameFileError:
    tqdm.write('Note, the config being used is the same as that already present in the results dir')

def prims_matrix_to_edges_simplified(matrix, words):
  '''
  Constructs a minimum spanning tree from the pairwise weights in matrix;
  returns the edges.

  Never lets punctuation-tagged words be part of the tree.
  '''
  pairs_to_distances = {}
  uf = UnionFind(len(matrix))
  punctuations_lst = [symb for symb in string.punctuation]
  punctuations_lst += ["''", ",", ".", ":", "``", "(", ")"]
  for i_index, line in enumerate(matrix):
    for j_index, dist in enumerate(line):
      if words[i_index] in punctuations_lst:
        continue
      if words[j_index] in punctuations_lst:
        continue
      pairs_to_distances[(i_index, j_index)] = dist
  edges = []
  for (i_index, j_index), distance in sorted(pairs_to_distances.items(), key = lambda x: x[1]):
    if uf.find(i_index) != uf.find(j_index):
      uf.union(i_index, j_index)
      edges.append((i_index, j_index))
  return edges

def run_inference(args):
    dataset = GPTInferenceDataset(args)
    dataloader = dataset.get_inference_dataloader()
    model_cls = choose_model_class(args)
    model = model_cls(args)
    probe_cls = choose_probe_class(args)
    probe = probe_cls(args)
    probe_params_path = os.path.join(args['reporting']['root'],args['probe']['params_path'])
    probe.load_state_dict(torch.load(probe_params_path))
    probe.eval()
    model.eval()
    all_edges = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            hidden_reps, lengths, words = batch
            batch_outputs = model(hidden_reps)
            batch_outputs = probe(batch_outputs).detach().cpu().float().numpy()
            for batch_output, length, word in tqdm(zip(batch_outputs, lengths, words)):
                mat = batch_output[:length, :length]
                sent_edges = prims_matrix_to_edges_simplified(mat, word)
                all_edges.append(sent_edges)
    output_path = os.path.join(args['dataset']['edges']['root'], args['dataset']['edges']['inference_path'])

    # merge edges and text altogether
    corpus_path = os.path.join(args['dataset']['corpus']['root'], args['dataset']['corpus']['inference_path'])
    corpus = []
    with open(corpus_path, 'r') as f:
      lines = f.readlines()
      for line in lines:
        corpus.append(line)
    
    texts_and_edges = []
    for text, edges in zip(corpus, all_edges):
      texts_and_edges.append({
        'text': text,
        'edges': edges
      })
    
    with open(output_path, 'wb') as f:
        pickle.dump(texts_and_edges, f)
      
    
    

    





if __name__ == '__main__':
  argp = ArgumentParser()
  argp.add_argument('experiment_config')
  cli_args = argp.parse_args()
  cli_args.seed = 42
  if cli_args.seed:
    np.random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


  yaml_args = yaml.load(open(cli_args.experiment_config), Loader=yaml.SafeLoader)
  #setup_new_experiment_dir(cli_args, yaml_args, cli_args.results_dir)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  yaml_args['device'] = device
  #yaml_args['train_probe'] = cli_args.train_probe
  run_inference(yaml_args)
