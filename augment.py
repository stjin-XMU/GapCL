import torch
from torch import autograd
from mrgnn.utils import calculate_loss
from losses import SupConLoss
import time

def gap_augment(bg, labels, model, loss_fn, args, device, init=None):
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
    if args.get('origin', True):
        preds = model(bg)
        x_org = model.get_last_x()
        loss = calculate_loss(preds, labels, loss_fn).mean()
        loss.backward()
        losses.append(loss)

    for i in range(depth):
        perturb.requires_grad_()
        if args.get('target', 'node') == 'node':
            model.node_perturb = perturb
        else:
            raise ValueError(f'Not supported perturb target {args["target"]}, you can only perturb the `node` or the `edge`.')
        preds = model(bg)
        x_gap = model.get_last_x() #last x, last feature of x(before linear layer)
        loss = calculate_loss(preds, labels, loss_fn).mean()/depth
        scloss = cl_fn(x_org, x_gap) * lambda_cl
        if is_comm:
            losses.append(loss + scloss)
        else:
            losses.append(loss)
        # grad = autograd.grad(loss, perturb)[0]
        loss.backward()
        grad = perturb.grad
        if constrain == 'inf':
            perturb = (perturb.detach() +
                       step_size * grad.sign()).clip(-size, size)
        elif constrain == 'l2':
            norm_grad = torch.einsum('ij,i->ij',grad,step_size/grad.norm(dim=1))
            norm_grad = torch.masked_fill(norm_grad, torch.isinf(norm_grad), 0)
            norm_grad = torch.masked_fill(norm_grad, torch.isnan(norm_grad), 0)
            perturb = perturb.detach() + norm_grad
        else:
            raise ValueError('Unknow constrain')
        
        perturbs.append(perturb.clone())

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
            print(model.node_perturb)
        else:
            raise ValueError(f'Not supported perturb target {args["target"]}, you can only perturb the `node` or the `edge`.')
        preds = model(bg)
        loss = calculate_loss(preds, labels, loss_fn).mean()/depth
        loss.backward()
        losses.append(loss)
        perturbs.append(perturb.clone())

    return perturbs, losses

def augment_loss(aug_arg, idx, bg, labels, loss_fn, trainloader, model, epoch):
    """
    Compute the loss with GAP
    Args:
        args (dict):
            init_method: how the GAP is initialized. Possible choices: `mean`. 
                `mean`: use the mean of the perturbations from the last epoch; 
            depth: how many times the GAP is updated. 
            size: bound of the GAP. 
            constrain: method to constrain the GAP. Possible values: `inf`, `l2`. Default to 'inf'. `inf` constains the l_{inf}-norm le to `size` that each scaler of the GAP is in [-size, size]. 
            target: where to add the GAP, `node` or `edge`. Default to 'node'. 
            origin: whether keep the unperturbed data. Default to True. 
        idx: the index of a single graph in the dataset.
    """
    aug_method = aug_arg.get('method','gap')
    device = bg.device

    if aug_arg['target'] == 'node':
        length = bg.num_nodes()
        if model.before_encoder:
            model.gap_dim = model.node_dim
        else:
            model.gap_dim = model.hid_dim
            
    if aug_method == 'gap':
        init_method = aug_arg.get('init_method', 'zero')
        assert init_method in ['mean'], f'unknorw init method {init_method}'
        #print(trainloader.dataset.dataset.perturb)
        if epoch == 0:
            init = torch.zeros([length, model.gap_dim]).to(device)
        else:
            init = torch.cat([trainloader.dataset.dataset.perturb[i] for i in idx], dim = 0) #拼接perturb
        perturbs, loss_aug = gap_augment(bg, labels, model, loss_fn, aug_arg, device, init)

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
        else:
            raise ValueError('init method is not provided now.')

        if save:
            node_size = bg.batch_num_nodes()
            start_index = torch.cat([torch.tensor([0], device=device), torch.cumsum(node_size, 0)[:-1]])
            type_size = node_size
            for i in range(bg.batch_size):
                start, size = start_index[i], type_size[i]
                # assert size != 0
                if select is None:
                    cur_perturb = save_perturb.narrow(0, start, size)
                else:
                    cur_perturb = save_perturb[select[i]].narrow(0, start, size)
                trainloader.dataset.dataset.perturb[idx[i]] = cur_perturb

    loss = sum(loss_aug)/2
    perturb_metrics = {'cos':cos_aug,'l2-norm':norm_2, 'inf-norm':norm_inf}

    return loss, perturb_metrics
