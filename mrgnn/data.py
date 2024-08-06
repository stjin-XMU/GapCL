import torch
from torch.utils.data import Dataset, Subset
import dgl
from dgl.dataloading import GraphDataLoader

from .featurize.atom import *
from .featurize.bond import *


def Mol2HomoGrpah(mol, af, bf):
    """homo graph"""
    if mol.GetNumAtoms() == 1:
        g = dgl.graph([], num_nodes=1)
    else:
        begin = []
        end = []
        for bond in mol.GetBonds():
            begin.append(bond.GetBeginAtomIdx())
            begin.append(bond.GetEndAtomIdx())
            end.append(bond.GetEndAtomIdx())
            end.append(bond.GetBeginAtomIdx())

        g = dgl.graph((begin, end))

    f_atom = []
    for idx in g.nodes():
        atom = mol.GetAtomWithIdx(idx.item())
        f_atom.append(af.featurize(atom))
    if af.descrete:
        g.ndata['x'] = torch.LongTensor(f_atom)
    else:
        g.ndata['x'] = torch.FloatTensor(f_atom)

    f_bond = []
    src, dst = g.edges()
    for i in range(g.num_edges()):
        f_bond.append(bf.featurize(
            mol.GetBondBetweenAtoms(src[i].item(), dst[i].item())))
    if bf.descrete:
        g.edata['x'] = torch.LongTensor(f_bond)
    else:
        g.edata['x'] = torch.FloatTensor(f_bond)

    return g


class MolGraphSet(Dataset):
    def __init__(self, df, target, af, bf, gtype=Mol2HomoGrpah, log=print):
        self.data = df
        self.mols = []
        self.smis = []
        self.labels = []
        self.graphs = []
        self.invalid = []
        self.num_task = len(target)
        for i, row in df.iterrows():
            smi = row['smiles']
            label = row[target].values.astype(float)
            try:
                mol = Chem.MolFromSmiles(smi)
                
                try:
                    Chem.SanitizeMol(mol)
                except:
                    log('invalid', smi)
                    self.invalid.append(i)

                if mol is None:
                    log('invalid', smi)
                    self.invalid.append(i)
                elif mol.GetNumAtoms() >= 100:
                    log('invalid', smi)
                    self.invalid.append(i)
                else:
                    g = gtype(mol, af, bf)
                    if g.num_nodes() == 0:
                        log('no edge in graph', smi)
                        self.invalid.append(i)
                    else:
                        self.mols.append(mol)
                        self.smis.append(smi)
                        self.graphs.append(g)
                        self.labels.append(label)
            except Exception as e:
                log('invalid', smi, e)
                self.invalid.append(i)

        self.perturb = [None]*len(self.mols)
        self.data = self.data.drop(self.invalid).reset_index(drop=True)

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):
        return idx, self.graphs[idx], self.labels[idx]
    
    def count(self,indices):
        labels = [self.labels[i] for i in indices]
        for idx,label in enumerate(list(zip(*labels))):
            print(f'task {idx}, num of positive samples: ',sum([1  if l==1 else 0  for l in label]))


def create_dataloader(df, target_names, af, bf,batch_size, shuffle=True, drop_last=False):
    dataset = MolGraphSet(df, target_names, af, bf)
    dataloader = GraphDataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    dataset.count(range(len(dataset)))
    return dataloader


def subset_loader(dataset, index, batch_size, shuffle=True, drop_last=False):
    dataloader = GraphDataLoader(
        Subset(dataset, index), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    # dataset.count(index)
    return dataloader
