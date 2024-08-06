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
    print("communities is {} and lenth is {} ".format(communities ,len(communities)))
    #print("communities 0 is {}".format(communities[0][0]))
    com_num=len(communities)
    for n in g.nodes:
        #print(n)
        label =0#findcomindex(n,communities,com_num)#g.nodes[n]['label']
        print(label)
        if label not in colorings_lookup:
            colorings_lookup[label] = set()
        colorings_lookup[label].add(node_to_idx[n])
    print("colorings_lookup is {}".format(colorings_lookup))

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

def augment(bg, labels, model, loss_fn, args, device, init=None,ob=None,idx=None,rn=None):
    """ generate gradient-based adversial perturbation (GAP)
    Args:
        bg: batched dgl graph. 
        labels: ground truth of the batch. 
        args (dict):
            depth: how many times the GAP is updated. 
            size: bound of the GAP. 
            constrain: method to constrain the GAP. Possible values: `inf`. Default to 'inf'. 
            target: where to add the GAP, `node` or `edge`. Default to 'node'. 
            origin: whether keep the unperturbed data. Default to True. 
    Returns:
        a list of GAPs of length `depth+1`
    """
    perturbs = []
    depth = args['depth']
    size = args['size']
    step_size = args.get('step_size',size/depth)
    constrain = args.get('constrain', 'inf')

    perturb = init.to(device)
 
    if args.get('origin', True):
        perturbs.append(torch.zeros_like(perturb))

    for i in range(depth):
        perturb.requires_grad_()
        model.zero_grad()
        if args.get('target', 'node') == 'node':
            model.node_perturb = perturb
        elif args['target'] == 'edge':
            model.edge_perturb = perturb
        elif args['target'] == 'graph':
            model.graph_perturb = perturb
        else:
            raise f'Not supported perturb target {args["target"]}, you can only perturb the `node` or the `edge`.'

      
        preds = model(bg)
        loss = calculate_loss(preds, labels, loss_fn).mean()
        loss.backward()
        grad = perturb.grad
    
        if constrain == 'inf':
            perturb = (perturb.detach() +
                       perturb.grad.sign()).clip(-size, size)
            #Add symmetry information to Nodes
            nodes_idx=0
            for index in range(0,rn.shape[0]):
                for ob_g in ob[index]:
                    if len(ob_g)>1:
                       #  Add symmetry information based on orbits
                        mean=0
                        for j in range(len(ob_g)):
                            mean=mean+perturb[nodes_idx+ob_g[j]].detach()
                        mean=mean/len(ob_g)
                        for j in range(len(ob_g)):
                            perturb[nodes_idx+ob_g[j]]=(perturb[nodes_idx+ob_g[j]]+mean ).clip(-size, size)
                nodes_idx=nodes_idx+rn[index]-1
            #Add symmetry information to Nodes
        elif constrain == 'l2':
            norm_grad = torch.einsum('ij,i->ij',grad,step_size/grad.norm(dim=1)) #对行求2的范式
            norm_grad = torch.masked_fill(norm_grad, torch.isinf(norm_grad), 0)
            norm_grad = torch.masked_fill(norm_grad, torch.isnan(norm_grad), 0)
            perturb = perturb.detach() + norm_grad        
        perturbs.append(perturb.clone())

    return perturbs
def symmetry_augment(bg, labels, length, model, loss_fn, args, device):
    """ generate gradient-based adversial perturbation (GAP)
    Args:
        bg: batched dgl graph. 
        labels: ground truth of the batch. 
        args (dict):
            depth: how many times the GAP is updated. 
            size: bound of the GAP. 
            constrain: method to constrain the GAP. Possible values: `inf`. Default to 'inf'. 
            target: where to add the GAP, `node` or `edge`. Default to 'node'. 
            origin: whether keep the unperturbed data. Default to True. 
    Returns:
        a list of GAPs of length `depth+1`
    """
    perturbs = []
    depth = args['depth']
    size = args['size']
    step_size = args.get('step_size',size/depth)
    constrain = args.get('constrain', 'inf')

    perturb = init.to(device)
    if args.get('origin', True):
        perturbs.append(torch.zeros_like(perturb))

    for i in range(depth):
        perturb.requires_grad_()
        model.zero_grad()
        if args.get('target', 'node') == 'node':
            model.node_perturb = perturb
        elif args['target'] == 'edge':
            model.edge_perturb = perturb
        elif args['target'] == 'graph':
            model.graph_perturb = perturb
        else:
            raise f'Not supported perturb target {args["target"]}, you can only perturb the `node` or the `edge`.'
        preds = model(bg)
        loss = calculate_loss(preds, labels, loss_fn).mean()
        loss.backward()
        grad = perturb.grad
        if constrain == 'inf':
            perturb = (perturb.detach() +
                       perturb.grad.sign()).clip(-size, size)
        elif constrain == 'l2':
            norm_grad = torch.einsum('ij,i->ij',grad,step_size/grad.norm(dim=1)) #对行求2的范式
            norm_grad = torch.masked_fill(norm_grad, torch.isinf(norm_grad), 0)
            norm_grad = torch.masked_fill(norm_grad, torch.isnan(norm_grad), 0)
            perturb = perturb.detach() + norm_grad        
        perturbs.append(perturb.clone())

    return perturbs

def random_augment(bg, labels, length, model, loss_fn, args, device):
    perturbs = []
    losses = []
    size = args['size']
    depth = args['depth']
    if args.get('origin', True):
        preds = model(bg)
        loss = calculate_loss(preds, labels, loss_fn).mean()
        loss.backward()
        losses.append(loss)
        
    for i in range(depth):
        perturb = torch.zeros([length, model.gap_dim]).to(
                device).uniform_(-size, size)
        if args.get('target', 'node') == 'node':
            model.node_perturb = perturb
        elif args['target'] == 'edge':
            model.edge_perturb = perturb
        elif args['target'] == 'graph':
            model.graph_perturb = perturb
        else:
            raise ValueError(f'Not supported perturb target {args["target"]}, you can only perturb the `node` or the `edge`.')
        preds = model(bg)
        loss = calculate_loss(preds, labels, loss_fn).mean()/depth
        loss.backward()
        losses.append(loss)
        perturbs.append(perturb.clone())

    return perturbs, losses

def augment_loss(aug_arg, idx, bg, labels, loss_fn, trainloader, model, epoch,rn):
    """
    Compute the loss with GAP
    Args:
        args (dict):
            init_method: how the GAP is initialized. Possible choices: `mean`, default to `mean`. 
                `mean`: use the mean of the perturbations from the last epoch; 
            depth: how many times the GAP is updated. 
            size: bound of the GAP. 
            constrain: method to constrain the GAP. Possible values: `inf`. Default to 'inf'. `inf` constains the l_{inf}-norm le to `size` that each scaler of the GAP is in [-size, size]. 
            target: where to add the GAP, `node` or `edge`. Default to 'node'. 
            origin: whether keep the unperturbed data. Default to True. 
        idx: the index of a single graph in the dataset.
    """
    device = bg.device
    
    if aug_arg['target'] == 'graph':
        model.gap_dim = model.hid_dim
        length = bg.batch_size
    elif aug_arg['target'] == 'node':
        length = bg.num_nodes()
        if model.before_encoder:
            model.gap_dim = model.node_dim
        else:
            model.gap_dim = model.hid_dim
    elif aug_arg['target'] == 'edge':
        length = bg.num_edges()
        if model.before_encoder:
            model.gap_dim = model.edge_dim
        else:
            model.gap_dim = model.hid_dim
            
    init_method = aug_arg.get('init_method', 'zero')     
    if epoch == 0:
        if init_method == 'zero':
            init = torch.zeros([length, model.gap_dim]).to(device)
        elif init_method == 'uniform':
            init = torch.zeros([length, model.gap_dim]).to(
                    device).uniform_(-aug_arg['size'], aug_arg['size'])        
    else:
        init = torch.cat([trainloader.dataset.dataset.perturb[i]
                          for i in idx], 0)
   # ob=[]
   # for i in idx:
   #     ob.append(trainloader.dataset.dataset.orbits[i])
    ob=trainloader.dataset.dataset.orbits                    
    perturbs= augment(bg, labels, model, loss_fn, aug_arg, device, init,ob,idx,rn)
    
    loss_aug = []
    for perturb in perturbs:
        model.zero_grad()
        if aug_arg['target'] == 'node':
            model.node_perturb = perturb
        elif aug_arg['target'] == 'edge':
            model.edge_perturb = perturb
        elif aug_arg['target'] == 'graph':
            model.graph_perturb = perturb
        preds = model(bg)
        loss_aug.append(calculate_loss(preds, labels, loss_fn))

    
    lc_method = aug_arg.get('lc_method', 'mean')
    select = None
    save = False
    if lc_method == 'mean':
        save_perturb = torch.stack(perturbs).mean(0).cpu()
        save = True
    elif lc_method == 'last':
        save_perturb = perturbs[-1].cpu()
        save = True
    elif init_method == 'max':
        raise ValueError('init method max is conflict with the training process now.')
        save_perturb = torch.stack(perturbs).cpu()
        select = torch.stack(loss_aug).max(0)[1]
        save = True
    if save:
        node_size = bg.batch_num_nodes()
        start_index = torch.cat(
            [torch.tensor([0], device=device), torch.cumsum(node_size, 0)[:-1]])
        for i in range(bg.batch_size):
            start, size = start_index[i], node_size[i]
            assert size != 0
            if select is None:
                cur_perturb = save_perturb.narrow(0, start, size)
            else:
                cur_perturb = save_perturb[select[i]].narrow(0, start, size)
            trainloader.dataset.dataset.perturb[idx[i]] = cur_perturb

    loss = sum(loss_aug)/len(perturbs)

    return loss

