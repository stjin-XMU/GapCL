import torch
from torch import nn
import torch.nn.functional as F


ACT_FUNC = {'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            }


def get_act_func(fn_name):
    fn_name = fn_name.lower()
    return ACT_FUNC[fn_name]


def split_batch(bg):
    feats = bg.ndata['h']
    node_size = bg.batch_num_nodes()
    max_num_node = node_size.max()
    hid_dim = feats.size(1)
    start_index = torch.cat(
        [torch.tensor([0], device=bg.device), torch.cumsum(node_size, 0)[:-1]])
    feat_lst = []
    mask_lst = []
    for i in range(bg.batch_size):
        start, size = start_index[i], node_size[i]
        assert size != 0
        cur_hidden = feats.narrow(0, start, size)
        pad = torch.zeros(max_num_node-size, hid_dim).to(bg.device)
        feat = torch.vstack([cur_hidden, pad]).unsqueeze(0)
        feat_lst.append(feat)
        mask = (feat.sum(-1, keepdim=True) == 0)
        mask_lst.append(mask)

    batch_feats = torch.cat(feat_lst, 0)
    batch_mask = torch.cat(mask_lst, 0)

    return batch_feats, batch_mask


class SinglerLayer(nn.Module):
    def __init__(self, dim, act):
        super(SinglerLayer, self).__init__()
        self.layer = nn.Linear(dim, dim)
        self.act = act

    def forward(self, x):
        return self.act(self.layer(x))


# node/edge functions
def copy_src(edge):
    return {'src': edge.src['h']}


def agg_msg(node):
    return {'h': node.data['h']+node.data['neigh']}


# def reverse_edge(bg, field='h', etype='_E'):
#     rev = torch.zeros_like(bg.edata[field])
#     src, dst = bg.edges()
#     for i in range(bg.num_edges(etype=etype)):
#         rev[i] = bg.edata[field][bg.edge_ids(dst[i], src[i])]
#     bg.edata[f'rev_{field}'] = rev
def reverse_edge_map(bg, etype='_E'):
    # rev_map = []
    # src, dst = bg.edges()
    # for i in range(bg.num_edges(etype=etype)):
    #     rev_map.append(bg.edge_ids(dst[i], src[i]))
    n = bg.num_edges(etype=etype)
    delta = torch.ones(n).type(torch.long)
    delta[torch.arange(1,n,2)] = -1
    rev_map = delta + torch.tensor(range(n))

    return rev_map


# message
def edge_network(edge):
    d2 = edge.data['h'].size(-1)
    d = int(d2**0.5)
    assert d**2 == d2
    return {'m': torch.einsum('nd,nde->ne', (edge.src['h'], edge.data['h'].reshape([-1, d, d])))}

# reduce


def attn_reduce(node):
    a = F.softmax(node.mailbox['m'][:, :, :, 0], dim=1) # n_node * n_neighbor * n_head
    h = node.mailbox['m'][:, :,:, 1:]  # n_node * n_neighbor * n_head * hid_dim
    return {'h': torch.einsum('bnh,bnhd->bhd', a, h).mean(1)}

# update


# readout
def sum_pooling(node_feats):
    """padding has no influence"""
    return node_feats.sum(1)


def mean_pooling(node_feats, mask):
    return node_feats.masked_fill(mask,torch.nan).nanmean(1)


class attention_pooling(nn.Module):
    def __init__(self, node_dim, hid_dim):
        super(attention_pooling, self).__init__()
        self.project_i = nn.Linear(node_dim + hid_dim, 1)
        self.project_j = nn.Linear(node_dim + hid_dim, hid_dim)

    def forward(self, bg):
        bg.ndata['h'] = torch.hstack([bg.ndata['h'], bg.ndata['x']])
        feats, masks = split_batch(bg)
        h1 = F.softmax(self.project_i(feats).masked_fill(masks, -1e5), dim=1)
        h2 = self.project_j(feats)
        # (h1.transpose(2,1) @ h2).squeeze(1) equivalent
        return torch.einsum('bni,bnj->bj', h1, h2)  # i = 1


class Set2Set(nn.Module):
    def __init__(self, input_dim, hidden_dim, act_fn=nn.ReLU, num_layers=1):
        super(Set2Set, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        if hidden_dim <= input_dim:
            print('ERROR: Set2Set output_dim should be larger than input_dim')
        self.lstm_output_dim = hidden_dim - input_dim
        self.lstm = nn.LSTM(hidden_dim, input_dim,
                            num_layers=num_layers, batch_first=True)

        # convert back to dim of input_dim
        self.pred = nn.Linear(hidden_dim, input_dim)
        self.act = act_fn

    def forward(self, embedding, mask):
        batch_size = embedding.size()[0]
        n = embedding.size()[1]
        device = embedding.device

        hidden = (torch.zeros(self.num_layers, batch_size, self.lstm_output_dim).to(device),
                  torch.zeros(self.num_layers, batch_size, self.lstm_output_dim).to(device))

        q_star = torch.zeros(batch_size, 1, self.hidden_dim).to(device)
        for i in range(n):
            q, hidden = self.lstm(q_star, hidden)
            e = embedding @ torch.transpose(q, 1, 2)  # bacth * n * 1
            a = F.softmax(e.masked_fill(mask, -1e5), dim=1)
            r = torch.sum(a * embedding, dim=1, keepdim=True)
            q_star = torch.cat((q, r), dim=2)
        q_star = torch.squeeze(q_star, dim=1)
        out = self.pred(q_star)

        return out
