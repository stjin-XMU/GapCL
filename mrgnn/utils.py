from torch import nn
import numpy as np
from tqdm import tqdm
import torch
import deepchem as dc
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score,\
    roc_auc_score, accuracy_score, log_loss


def modify_config(config):
    """inplace change to config"""
    if 'featurizer' not in config['model']:
        config['model']['featurizer'] = ['set1','set1']


    if config['model']['model_type'] == 'GGNN':
        if config['model']['featurizer'][1] not in ['set2']:
            print('GGNN requires discrete edge feature')
            config['model']['featurizer'][1] = 'set2'
    elif config['model']['model_type'] == 'GAT':
        if 'n_head' not in config['model']:
            print('specify number of heads for GAT. Using default 4')
            config['model']['n_head'] = 4


def num_atom(mol):
    return mol.GetNumAtoms()


def size_split(df, size=[0.9, 0.1]):
    df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
    df['num_atoms'] = df['mol'].apply(num_atom)
    df = df.sort_values('num_atoms').reset_index(drop=True)
    n = len(df)
    s = ['train']*round(n*size[0])+['test']*round(n*size[1])
    df['set'] = s

    return list(df[df['set'] == 'train'].index), list(df[df['set'] == 'test'].index)


def generate_scaffold(mol, include_chirality: bool = False):
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=include_chirality)

    return scaffold


def scaffold_to_smiles(mols, use_indices):
    scaffolds = defaultdict(set)
    for i, mol in tqdm(enumerate(mols), total=len(mols)):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds


def scaffold_split(df, sizes=[0.8, 0.2]):
    df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
    assert sum(sizes) == 1

    # Split
    n = len(df)
    train_size, test_size = sizes[0] * n, sizes[1] * n
    train, test = [], []
    train_scaffold_count, test_scaffold_count = 0, 0

    # Map from scaffold to index in the data
    scaffold_to_indices = scaffold_to_smiles(df['mol'], use_indices=True)

    index_sets = sorted(list(scaffold_to_indices.values()),
                        key=lambda index_set: len(index_set),
                        reverse=True)

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1

    print(f'Total scaffolds = {len(scaffold_to_indices):,} | '
          f'train scaffolds = {train_scaffold_count:,} | '
          f'test scaffolds = {test_scaffold_count:,}')

    return train, test

def split(dataset, size=[0.8, 0.1, 0.1], split_type='size', seed=0):
    if split_type == 'size':
        train, test = size_split(dataset.data, [size[0]+size[1], size[2]])
    elif split_type == 'scaffold':
        train, test = scaffold_split(dataset.data, [size[0]+size[1], size[2]])
    elif split_type == 'random':
        train, test = train_test_split(
            range(len(dataset)), test_size=size[2], random_state=seed)
    elif split_type == 'stratified':
        assert dataset.num_task == 1, 'multi-task for stratified split'
        train, test = next(StratifiedShuffleSplit(n_splits=1,test_size=size[2],random_state=seed).split(range(len(dataset)),dataset.labels))
    else:
        print('supported split type: size, scaffold, random')
    train, valid = train_test_split(
        train, test_size=size[1]/(size[0]+size[1]), random_state=seed)

    return {"train": train, "valid": valid, "test": test}

def get_scaffold_split(dataset, size=[0.8, 0.1, 0.1], split_type='size', seed=0):
    # data = dataset.data
    smiles = dataset.smis
    # indx = 0
    # remove_idx = []
    idx_org = [i for i in range(len(smiles))]
    xs = np.array(idx_org)
    ws = np.zeros(len(smiles))
    y = np.zeros(len(smiles))
    dataset = dc.data.DiskDataset.from_numpy(X=xs,y=y,w=ws,ids=smiles)
    scaffoldsplitter = dc.splits.ScaffoldSplitter()
    train_dataset, valid_dataset, test_dataset = scaffoldsplitter.train_valid_test_split(dataset)
    return{
        'train': train_dataset.X.tolist(),
        'valid': valid_dataset.X.tolist(),
        'test': test_dataset.X.tolist(),
    }


def calculate_loss(preds, labels, loss_fn):
    labels.reshape(preds.shape)
    #print("labels", labels,labels.shape)
    #print("preds",preds,preds.shape )
    preds, truth = fill_nan_label(preds, labels)   # any loss_fn(0,0) = 0
    loss = loss_fn(preds, truth).mean(1)  # average on num_tasks // average on num_valid_labels ??
    return loss

def calculate_metrics(preds, labels, metric_fn):
    num_task = preds.size(1)
    if num_task > 1:
        loss = 0
        for i in range(num_task):
            loss += metric_fn(
                preds[:, i], labels[:, i])/num_task
    else:
        loss = metric_fn(preds, labels.reshape(preds.shape))

    return loss

def fill_nan_label(pred, truth):
    nan = torch.isnan(truth)
    truth = truth.masked_fill(nan,0)
    pred = pred.masked_fill(nan,0)

    return pred, truth

def remove_nan_label(pred, truth):
    """
    pred,truth: (Batch_Size,)
    """
    nan = torch.isnan(truth)
    truth = truth[~nan]
    pred = pred[~nan]

    return pred, truth


def roc_auc(pred, truth):
    pred, truth = remove_nan_label(pred, truth)
    if truth.sum() == len(truth) or truth.sum() == 0:
        return 1
    if torch.isnan(pred).any():
        print("Found NaN")
        print(pred)
        pred = torch.nan_to_num(pred, nan=0.0)
    return roc_auc_score(truth, pred)


def rmse(pred, truth):
    return nn.functional.mse_loss(pred, truth, reduction='mean')**0.5


def mae(pred, truth):
    return mean_absolute_error(truth, pred)

def acc(pred, truth):
    return accuracy_score(truth, pred)


LOSS_FUNCS = {
    'mse': nn.MSELoss(reduction='none'),
    'crossentropy': nn.CrossEntropyLoss(reduction='none'),
    'bce': nn.BCEWithLogitsLoss(reduction='none'),
    

}
METRICS_FUNCS = {
    'auc': roc_auc,
    'mse': nn.MSELoss(reduction='mean'),
    'rmse': rmse,
    'mae': mae,
    'acc': acc,
    'crossentropy': nn.CrossEntropyLoss(reduction='mean'),
    'bce': nn.BCEWithLogitsLoss(reduction='mean'),
}


def get_func(fn_name, task='loss'):
    fn_name = fn_name.lower()
    if task == 'loss':
        return LOSS_FUNCS[fn_name]
    elif task == 'metrics':
        return METRICS_FUNCS[fn_name]


class StandardScaler:
    """A StandardScaler normalizes a dataset.

    When fit on a dataset, the StandardScaler learns the mean and standard deviation across the 0th axis.
    When transforming a dataset, the StandardScaler subtracts the means and divides by the standard deviations.
    """

    def __init__(self, means: np.ndarray = None, stds: np.ndarray = None, replace_nan_token=None):
        """
        Initialize StandardScaler, optionally with means and standard deviations precomputed.

        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: The token to use in place of nans.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X):
        """
        Learns means and standard deviations across the 0th axis.

        :param X: A list of lists of floats.
        :return: The fitted StandardScaler.
        """
        X = torch.tensor(X)
        self.means = torch.mean(X, axis=0)
        self.stds = torch.std(X, axis=0)
        self.means = torch.where(torch.isnan(
            self.means), torch.zeros(self.means.shape), self.means)
        self.stds = torch.where(torch.isnan(self.stds),
                                torch.ones(self.stds.shape), self.stds)
        self.stds = torch.where(
            self.stds == 0, torch.ones(self.stds.shape), self.stds)

        return self

    def transform(self, X):
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.

        :param X: A list of lists of floats.
        :return: The transformed data.x
        """
        X = torch.tensor(X)
        transformed_with_nan = (X - self.means) / self.stds
#         transformed_with_none = torch.where(torch.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_nan

    def inverse_transform(self, X):
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.

        :param X: A list of lists of floats.
        :return: The inverse transformed data.
        """
#         X = torch.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
#         transformed_with_none = torch.where(torch.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_nan
