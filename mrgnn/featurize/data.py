import torch
from torch.utils.data import Dataset,Subset
import dgl
from dgl.dataloading import GraphDataLoader

from .featurize.atom import *
from .featurize.bond import *
 
import torch
from mrgnn.utils import calculate_loss
import pynauty
from communities.algorithms import louvain_method
import dgl
import numpy as np 
import networkx as nx
from mrgnn.models.basic import *
def invert_dict(d):
    return {v: k for (k, v) in d.items()}

def invert_list(lst):
    ret = {}
    for (i, elem) in enumerate(lst):
        ret[elem] = i
    return ret

# Sort a set of orbits by the minimum canonical index
def sort_orbits(canonization_mapping, orbits):
    def min_canon_node(nodes):
        return min([canonization_mapping[n] for n in nodes])

    return sorted(orbits, key=min_canon_node)

# Map nodes to the index of their orbit
def canonical_orbits_mapping(sorted_orbits):
    ret = {}
    for (i, orb) in enumerate(sorted_orbits):
        for n in orb:
            ret[n] = i
    return ret
# Convert a NetworkX graph to a nauty graph
# Input should be a NetworkX digraph with node labels represented as strings, stored in the 'label'
# field of the NetworkX node attribute dictionary
 
def findcomindex(n,commun,com_num):
    lb=0
    print("n is {}".format(n))
    for i in range(0,com_num):
          if n in commun[i]:
            # print(commun[i])
             lb=i
    return lb
         
 
def nauty_graph(g):
    # Map each node to a natural number 0,...,n-1 in an arbitrary order. The number of a node is a node index
    node_to_idx = {n: i for (i, n) in enumerate(g.nodes)}
    # Convert the NetworkX adjacency information to use the node indices
    adj_dict = {node_to_idx[s]: [node_to_idx[t] for t in g.neighbors(s)] for s in g.nodes}

    # Dictionary mapping node labels to a set of node indices
    colorings_lookup = {}
    communities, _ = louvain_method(np.array(nx.adjacency_matrix(g).todense()))
    #print("communities is {} and lenth is {} ".format(communities ,len(communities)))
    #print("communities 0 is {}".format(communities[0][0]))
    com_num=len(communities)
    for n in g.nodes:
        #print(n)
        label =0#findcomindex(n,communities,com_num)#g.nodes[n]['label']
       # print(label)
        if label not in colorings_lookup:
            colorings_lookup[label] = set()
        colorings_lookup[label].add(node_to_idx[n])
    #print("colorings_lookup is {}".format(colorings_lookup))

    # It turns out that the order of the vertex_coloring passed to nauty is important
    ordered_labels = sorted(colorings_lookup.keys())

    # Convert the dictionary into a list of sets. Each set contains node indices with identical labels
    colorings = [colorings_lookup[label] for label in ordered_labels]

    # Construct the pynauty graph
    nauty_g = pynauty.Graph(g.order(), directed=True, adjacency_dict=adj_dict, vertex_coloring=colorings)

    # Return the node to index conversion function and the nauty graph
    return (node_to_idx, nauty_g)

# Returns a list of nodes, ordered in the canonical order
def canonize(idx_to_node, nauty_g):
    canon = pynauty.canon_label(nauty_g)
    return [idx_to_node[i] for i in canon]

def escape(s):
    # Replace backslashes with double backslash and quotes with escaped quotes
    return s.replace('\\', '\\\\').replace('"', '\\"')

def to_str(data):
    if isinstance(data, list):
        return "[{}]".format(",".join([to_str(elem) for elem in data]))
    elif isinstance(data, str):
        return '"{}"'.format(escape(data))
    elif isinstance(data, tuple):
        return "({})".format(",".join([to_str(elem) for elem in data]))
    elif isinstance(data, int):
        return str(data)
    else:
        raise TypeError("Unable to call to_str on " + str(data))

# Returns a list of lists of nodes, each list is an orbit
def orbits(idx_to_node, nauty_g):
    # orbs gives the orbits of the graph. Two nodes i,j are in the same orbit if and only if orbs[i] == orbs[j]
    (_, _, _, orbs, num_orbits) = pynauty.autgrp(nauty_g)

    # orbits_lookup maps an orbit identifier to a list of nodes in that orbit
    orbits_lookup = {}
    for i in range(len(orbs)):
        orb_label = orbs[i]
        if orb_label not in orbits_lookup:
            orbits_lookup[orb_label] = []
        orbits_lookup[orb_label].append(idx_to_node[i])

    # Now dispose of the orbit identifier, we are only interested in the orbit groupings
    return list(orbits_lookup.values())

# Analyze a NetworkX graph, returning a list of nodes in canonical order and a list of orbits
def analyze_graph(g, compute_orbits):
    (node_to_idx, nauty_g) = nauty_graph(g)
    idx_to_node = invert_dict(node_to_idx)
    if compute_orbits:
        orbs = orbits(idx_to_node, nauty_g)
    else:
        orbs = None
    return (canonize(idx_to_node, nauty_g), orbs)

def Mol2HomoGrpah(mol,af, bf):
    """homo graph"""
    if mol.GetNumAtoms() == 1:
        g = dgl.graph([],num_nodes=1)
    else:
        begin = []
        end = []
        for bond in mol.GetBonds():
            begin.append(bond.GetBeginAtomIdx())
            begin.append(bond.GetEndAtomIdx())
            end.append(bond.GetEndAtomIdx())
            end.append(bond.GetBeginAtomIdx())

        g = dgl.graph((begin,end))

    f_atom = []
    for idx in g.nodes():
        atom = mol.GetAtomWithIdx(idx.item())
        f_atom.append(af.featurize(atom))
    if af.descrete:
        g.ndata['x'] = torch.LongTensor(f_atom)
    else:
        g.ndata['x'] = torch.FloatTensor(f_atom)
    
    f_bond = []
    src,dst = g.edges()
    for i in range(g.num_edges()):
        f_bond.append(bf.featurize(mol.GetBondBetweenAtoms(src[i].item(),dst[i].item())))
    if bf.descrete:
        g.edata['x'] = torch.LongTensor(f_bond)
    else:
        g.edata['x'] = torch.FloatTensor(f_bond)
    
    return g


class MolGraphSet(Dataset):
    def __init__(self,df,target,af,bf,gtype=Mol2HomoGrpah,log=print):
        self.data = df
        self.mols = []
        self.labels = []
        self.graphs = []
        self.invalid = []
        self.orbits=[]
        self.redundance=[]
        for i,row in df.iterrows():
            smi = row['smiles']
            label = row[target].values.astype(float)
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    log('invalid',smi)
                    self.invalid.append(i)
                else:
                    g = gtype(mol,af,bf)
                    if g.num_nodes() == 0:
                        log('no edge in graph',smi)
                        self.invalid.append(i)
                    else:
                        self.mols.append(mol)
                        self.graphs.append(g)
                        self.labels.append(label)
                        nx=g.to_networkx().to_undirected()
                        node_to_id ,orbits = analyze_graph(nx,compute_orbits=True)
                        redundance = len(nx.nodes)
                       
                        self.redundance.append(redundance)
                        
                        self.orbits.append(orbits)
            except Exception as e:
                log('invalid',smi,e)
                self.invalid.append(i)
       
        self.perturb = [None]*len(self.mols)
                
    def __len__(self):
        return len(self.mols)
    
    def __getitem__(self,idx):
        
        return idx,self.graphs[idx],self.labels[idx] ,self.redundance[idx]
    

def create_dataloader(df,target_names,batch_size,shuffle=True,train=True):
    dataset = MolGraphSet(df,target_names)
    dataloader = GraphDataLoader(dataset,batch_size=batch_size,shuffle=shuffle)
    
    return dataloader
    
def subset_loader(dataset,index,batch_size,shuffle=True):
    dataloader = GraphDataLoader(Subset(dataset,index),batch_size=batch_size,shuffle=shuffle)
    return dataloader
