import torch
from mrgnn.utils import calculate_loss
from losses import SupConLoss
import time

def gap_augment(bg, labels, model, loss_fn, args, device, ob, rn, init=None):
    """ generate gradient-based adversial perturbation (GAP)
    Args:
        bg: batched dgl graph. 
        labels: ground truth of the batch. 
        args (dict):
            depth: how many times the GAP is updated. 
            size: bound of the GAP. 
            constrain: method to constrain the GAP. Possible values: `inf`, `l2`. Default to 'inf'. 
            target: where to add the GAP, `node` or `edge`. Default to 'node'. 
            origin: whether keep the unperturbed data. Default to True. 
            step_size: learning rate to update gap
    Returns:
        a list of GAPs of length `depth+1`
    """
    perturbs = []
    losses = []
    depth = args['depth']
    size = args['size']
    is_comm = args.get('is_comm', True)
    constrain = args.get('constrain', 'inf')
    step_size = args.get('step_size',size/depth)
    lambda_cl=0.01
    cl_fn = SupConLoss().to(device)

    perturb = init.to(device)
    preds = model(bg)
    x_org = model.get_last_x()


    for i in range(depth):
        perturb.requires_grad_()
        if args.get('target', 'node') == 'node':
            model.node_perturb = perturb
        elif args['target'] == 'edge':
            model.edge_perturb = perturb
        elif args['target'] == 'graph':
            model.graph_perturb = perturb
        else:
            raise ValueError(f'Not supported perturb target {args["target"]}, you can only perturb the `node` or the `edge`.')
        preds = model(bg)
        x_gap = model.get_last_x() #last x, last feature of x(before linear layer)
        loss = calculate_loss(preds, labels, loss_fn).mean()/depth

        loss.backward()
        grad = perturb.grad
        
        if constrain == 'inf':
            perturb = (perturb.detach() +
                       perturb.grad.sign()).clip(-size, size)
            # Add symmetry information to Nodes
            nodes_idx = 0
            for index in range(0, rn.shape[0]):
                for ob_g in ob[index]:
                    if len(ob_g) > 1:
                        #  Add symmetry information based on orbits
                        mean = 0
                        for j in range(len(ob_g)):
                            mean = mean + perturb[nodes_idx + ob_g[j]].detach()
                        mean = mean / len(ob_g)
                        for j in range(len(ob_g)):
                            perturb[nodes_idx + ob_g[j]] = (perturb[nodes_idx + ob_g[j]] + mean).clip(-size, size)
                nodes_idx = nodes_idx + rn[index] - 1
            # Add symmetry information to Nodes

        elif constrain == 'l2':
            norm_grad = torch.einsum('ij,i->ij', grad, step_size / grad.norm(dim=1))  # 对行求2的范式
            norm_grad = torch.masked_fill(norm_grad, torch.isinf(norm_grad), 0)
            norm_grad = torch.masked_fill(norm_grad, torch.isnan(norm_grad), 0)
            perturb = perturb.detach() + norm_grad
            # Add symmetry information to Nodes
            nodes_idx = 0
            for index in range(0, rn.shape[0]):
                for ob_g in ob[index]:
                    if len(ob_g) > 1:
                        #  Add symmetry information based on orbits
                        mean = 0
                        for j in range(len(ob_g)):
                            mean = mean + perturb[nodes_idx + ob_g[j]].detach()
                        mean = mean / len(ob_g)
                        for j in range(len(ob_g)):
                            perturb[nodes_idx + ob_g[j]] = (perturb[nodes_idx + ob_g[j]] + mean).clip(-size, size)
                nodes_idx = nodes_idx + rn[index] - 1
            # Add symmetry information to Nodes
        else:
            raise ValueError('Unknow constrain')
        
        perturbs.append(perturb.clone())
    
    # update theta
    preds = model(bg)
    x_gap = model.get_last_x() #last x, last feature of x(before linear layer)
    scloss = cl_fn(x_org, x_gap) * lambda_cl
    loss = calculate_loss(preds, labels, loss_fn).mean()
    if is_comm:
        losses.append(loss + scloss)
    else:
        losses.append(loss)

    return perturbs, losses

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

def augment_loss(aug_arg, idx, bg, labels, loss_fn, trainloader, model, epoch, rn):
    """
    Compute the loss with GAP
    Args:
        args (dict):
            init_method: how the GAP is initialized. Possible choices: `last`, `mean`, `max`, `zero`, `uniform`, default to `zero`. 
                `last`: use the last perturbation from the last epoch; 
                `mean`: use the mean of the perturbations from the last epoch; 
                `max`: use the perturbation that maximize the loss; 
                `zero`: init the perturbation with zeros; 
                `uniform`: sample a random perturbation from a uniform distributioin( U(-size,size) ); 
            depth: how many times the GAP is updated. 
            size: bound of the GAP. 
            constrain: method to constrain the GAP. Possible values: `inf`, `l2`. Default to 'inf'. `inf` constains the l_{inf}-norm le to `size` that each scaler of the GAP is in [-size, size]. 
            target: where to add the GAP, `node` or `edge`. Default to 'node'. 
            origin: whether keep the unperturbed data. Default to True. 
        idx: the index of a single graph in the dataset.
    """
    aug_method = aug_arg.get('method','gap')
    device = bg.device
    # print(aug_method)
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
            

    if aug_method == 'gap':
        init_method = aug_arg.get('init_method', 'zero')
        assert init_method in ['last', 'mean', 'max', 'zero', 'uniform'], f'unknorw init method {init_method}'
        if init_method == 'zero' or epoch == 0:
            init = torch.zeros([length, model.gap_dim]).to(device)
        elif init_method == 'uniform':
            init = torch.zeros([length, model.gap_dim]).to(
                    device).uniform_(-aug_arg['size'], aug_arg['size'])
        else:
            if(aug_arg['target'] == 'graph'):
                init = torch.stack([trainloader.dataset.dataset.perturb[i] for i in idx], dim = 0)
            else:
                init = torch.cat([trainloader.dataset.dataset.perturb[i] for i in idx], dim = 0) #拼接perturb
               
        ob = trainloader.dataset.dataset.orbits
        perturbs, loss_aug = gap_augment(bg, labels, model, loss_fn, aug_arg, device, ob, rn, init)

    elif aug_method == 'random':
       
        perturbs, loss_aug = random_augment(bg,labels,length,model,loss_fn, aug_arg, device)

    # watch the perturbations
    cos_aug = torch.tensor(0.)
    norm_inf = torch.tensor(0.)
    norm_2 = torch.tensor(0.)
    for perturb in perturbs:
        if aug_arg['target'] == 'node' and model.before_encoder:
            cos_aug += torch.cosine_similarity(perturb,bg.ndata['x']).mean().cpu()/len(perturbs) 
            norm_inf += perturb.abs().max(1)[0].mean().cpu()/len(perturbs)
            norm_2 += perturb.norm(dim=1).mean().cpu()/len(perturbs)

    if aug_method == 'gap':
        select = None
        save = False
        if init_method == 'mean':
            save_perturb = torch.stack(perturbs).mean(0).cpu()
            save = True
        elif init_method == 'last':
            save_perturb = perturbs[-1].cpu()
            save = True
        elif init_method == 'max':
            raise ValueError('init method max is conflict with the training process now.')
            save_perturb = torch.stack(perturbs).cpu()
            select = torch.stack(loss_aug).max(0)[1]
            save = True
        if save:
            node_size = bg.batch_num_nodes()
            edge_size = bg.batch_num_edges()
            if(aug_arg['target'] == 'edge'):
                start_index = torch.cat([torch.tensor([0], device=device), torch.cumsum(edge_size, 0)[:-1]])
                type_size = edge_size
            else:
                start_index = torch.cat([torch.tensor([0], device=device), torch.cumsum(node_size, 0)[:-1]])
                type_size = node_size
            for i in range(bg.batch_size):
                start, size = start_index[i], type_size[i]
                # assert size != 0
                if aug_arg['target'] == 'graph':
                    if select is None:
                        cur_perturb = save_perturb[i]
                    else:
                        cur_perturb = save_perturb[select[i]]
                    trainloader.dataset.dataset.perturb[idx[i]] = cur_perturb
                else:
                    if select is None:
                        cur_perturb = save_perturb.narrow(0, start, size)
                    else:
                        cur_perturb = save_perturb[select[i]].narrow(0, start, size)
                    trainloader.dataset.dataset.perturb[idx[i]] = cur_perturb

    loss = sum(loss_aug)/2
    perturb_metrics = {'cos':cos_aug,'l2-norm':norm_2, 'inf-norm':norm_inf}

    return loss, perturb_metrics
