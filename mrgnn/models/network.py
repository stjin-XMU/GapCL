import torch
from torch import nn
#from dgl import function as fn

from .basic import *

# to be implemented:
# master node and virtual edge
# GIN

__all__ = [ 'GCN']


class BaseGNN(nn.Module):
    """framework for homogeneous dgl graph"""

    def __init__(self, args):
        super(BaseGNN, self).__init__()
        hid_dim = args['hid_dim']
        self.hid_dim = hid_dim
        self.node_dim = args['atom_dim']
        self.edge_dim = args['bond_dim']
        self.last_x = None
        if isinstance(args['num_task'],int):
            self.num_tasks = args['num_task']
            self.few_shot = False
        elif isinstance(args['num_task'], tuple):
            self.num_train_tasks, self.num_test_tasks = args['num_task']
            # self.num_tasks = max(args['num_task'])
            self.num_tasks = sum(args['num_task'])
            self.few_shot = True
            self.test = False

        self.depth = args['depth']
        self.w_atom = nn.Linear(self.node_dim, hid_dim)
        self.w_bond = nn.Linear(self.edge_dim, hid_dim)
        self.act = get_act_func(args['act'])
        self.outlayer = nn.Sequential(SinglerLayer(hid_dim, self.act),
                                 nn.Linear(hid_dim, self.num_tasks)
                                 )

        self.before_encoder = args.get('before_encoder',True)
        self.node_perturb = None
        self.edge_perturb = None
        self.graph_perturb = None

        # if args['augment']['target'] == 'node':
        #     self.attacker = nn.Linear(args['atom_dim'],args['atom_dim'])

    def initialize_weights(self):
        for param in self.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

    def init_feat(self, bg):
        
        if self.w_atom is not None:
            if self.node_perturb is not None:
                if self.before_encoder:
                    bg.ndata['h'] = self.w_atom(bg.ndata['x']+self.node_perturb)
                else:
                    bg.ndata['h'] = self.w_atom(bg.ndata['x'])+self.node_perturb
                self.node_perturb = None
            else:
                bg.ndata['h'] = self.w_atom(bg.ndata['x'])
            bg.ndata['h'] = self.act(bg.ndata['h'])

        if self.w_bond is not None:
            if self.edge_perturb is not None:
                if self.before_encoder:
                    bg.edata['h'] = self.w_bond(bg.edata['x']+self.edge_perturb)
                else:
                    bg.edata['h'] = self.w_bond(bg.edata['x'])+self.edge_perturb
                self.edge_perturb = None
            else:
                # print("Shape of bg.edata['x']:", bg.edata['x'].shape)
                # print("bg.edata['x']:", bg.edata['x'])
                # print("Shape of bg.ndata['x']:", bg.ndata['x'].shape)
                # print("bg.ndata['x']:", bg.ndata['x'])
                # print(bg)
                if bg.edata['x'].shape[1] == 0:
                    # TODO
                    bg.edata['h'] = torch.zeros((bg.num_edges(), self.w_bond.out_features))
                else:
                    bg.edata['h'] =  self.w_bond(bg.edata['x'])
            bg.edata['h'] = self.act(bg.edata['h'])

    def out(self,x):
        if self.graph_perturb is None:
            out =  self.outlayer(torch.sigmoid(x))
        else:
            x = torch.sigmoid(x)+self.graph_perturb
            self.graph_perturb = None
            out =  self.outlayer(x)

        if self.few_shot:
            # multi-tasks output
            if self.test:
                out = out[:,self.num_train_tasks:]
            else:
                out = out[:,:self.num_train_tasks]

            # if self.test:
            #     out = out[:,:self.num_test_tasks]
            # else:
            #     out = out[:,:self.num_train_tasks]
                
        self.last_x = x
        return out # return the graph readout embedding vector
    
    def get_last_x(self):
        return self.last_x

    def forward(self, bg):
        raise NotImplementedError


class GCN(BaseGNN):
    """https://arxiv.org/abs/1609.02907"""

    def __init__(self, args):
        super(GCN, self).__init__(args)
        self.depth = args['depth']
        self.w = nn.ModuleList([nn.Linear(self.hid_dim, self.hid_dim)
                               for _ in range(self.depth)])

        self.register_module('w_bond', None)
        self.readout = mean_pooling
        self.initialize_weights()

    def forward(self, bg):
        self.init_feat(bg)
        node_feats = bg.ndata['h']
        n = bg.num_nodes()
        self_loop = torch.diag(torch.ones(n)).to(bg.device)
        a = (bg.adjacency_matrix().to_dense() + self_loop).to(bg.device)
        d = torch.diag(1/a.sum(0)**0.5)
        for layer in self.w:
            node_feats = self.act(d@a@d@layer(node_feats))
        bg.ndata['h'] = node_feats
        graph_feats = self.readout(*split_batch(bg))
        return self.out(graph_feats)

