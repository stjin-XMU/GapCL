import os
import numpy as np
import pandas as pd
import json
import time
import operator
from tqdm import tqdm
import torch
from torch.optim import Adam


from augment import augment_loss
from mrgnn.featurize import get_featurizer
from mrgnn.utils import get_func, get_scaffold_split, split, calculate_loss, calculate_metrics, modify_config
from mrgnn.models import build_model
from mrgnn.data import MolGraphSet, subset_loader, create_dataloader
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))


def evaluate(dataloader, model, device, metric_fn, metric_dtype, task, scaler=None):
    preds = []
    labels = []
    for idx, bg, label in dataloader:
        bg, label = bg.to(device), label.type(metric_dtype)
        pred = model(bg).cpu().detach()
        if scaler is not None:
            pred = scaler.inverse_transform(pred)
        if task == 'classification':
            pred = torch.sigmoid(pred)
        elif task == 'multiclass':
            pred = torch.softmax(pred, dim=1)
        #if regression, no extra actions taken
        preds.append(pred.cpu())
        labels.append(label)

    preds = torch.cat(preds)
    labels = torch.cat(labels)
    m = calculate_metrics(preds, labels, metric_fn)

    return m


def train(data_args, train_args, model_args, seed, model_type, split_type='size', device='cuda', save=False):
    epochs = train_args['epochs']
    num_fold = train_args['num_fold']
    device = device if torch.cuda.is_available() else 'cpu'
    batch_size = data_args['batch_size']
    task = data_args['task']

    af, bf = get_featurizer(model_args['featurizer'])
    model_args['atom_dim'] = af.dim
    model_args['bond_dim'] = bf.dim

    df = pd.read_csv(os.path.join(ROOT, data_args['path']))
    dataset = MolGraphSet(df, data_args['tasks'], af, bf)
    save_path=None
    if save:
        save_path = 'ckpt_'+train_args['group_name']
        os.makedirs(save_path, exist_ok=True)

    torch.manual_seed(seed)
    loss_fn = get_func(train_args['loss_fn'], 'loss')
    metric_fn = get_func(train_args['metric_fn'], 'metrics')
    loss_dtype = torch.float32
    metric_dtype = torch.float32

    for fold in range(num_fold):
     
        if split_type == 'defined':
            trainloader = create_dataloader(df,data_args['tasks'], af, bf,batch_size, shuffle=True, drop_last=True) 
            df_valid = pd.read_csv(os.path.join(ROOT, data_args['valid']))
            valloader = create_dataloader(df_valid,data_args['tasks'], af, bf,batch_size, shuffle=False, drop_last=False)
            df_test = pd.read_csv(os.path.join(ROOT, data_args['test']))
            testloader = create_dataloader(df_test,data_args['tasks'], af, bf,batch_size, shuffle=False, drop_last=False)
        else:
            if split_type == 'chem_scaffold':
                index = get_scaffold_split(dataset, size=[0.7, 0.1, 0.2],
                            seed=seed+fold, split_type=split_type)
            else:
                index = split(dataset, size=[0.7, 0.1, 0.2],
                            seed=seed+fold, split_type=split_type)
            trainloader = subset_loader(
                dataset, index['train'], batch_size, shuffle=True, drop_last=False)
            valloader = subset_loader(dataset, index['valid'],
                batch_size, shuffle=False)
            testloader = subset_loader(dataset, index['test'],batch_size, shuffle=False)

        scaler = None

        print(f'dataset size, train: {len(trainloader.dataset)}, \
                val: {len(valloader.dataset)}, \
                test: {len(testloader.dataset)}')
        model = build_model(model_type, model_args).to(device)
        
        print(f"--model_type:{model_type}------")
        optimizer = Adam(model.parameters(), train_args['lr'])


        if train_args['metric_fn'] in ['auc', 'acc']:
            best = 0
            best_test = 0
            op = operator.gt
        else:
            best = np.inf
            best_test = np.inf
            op = operator.lt
        best_epoch = 0
        best_epoch_test = 0
        test_on_best_valid = 0

        for epoch in tqdm(range(epochs)):
            model.train()
            total_loss = 0
            pm_avg = {'cos': 0, 'l2-norm': 0, 'inf-norm': 0}
            # print(f"--epoch:{epoch}--model_type:{model_type}------>model: {model.state_dict()}")
            for idx, bg, labels in trainloader:
                #print("label in loader", labels, labels.shape)
                bg, labels = bg.to(device), labels.type(
                    loss_dtype).to(device)
                optimizer.zero_grad()
                
                if train_args.get('augment', None) is None:
                    preds = model(bg)
                   
                    loss = calculate_loss(preds, labels, loss_fn).mean()
                    loss.backward()
                else:
                    aug_arg = train_args['augment']
                    
                    loss, p_metrics = augment_loss(
                        aug_arg, idx, bg, labels, loss_fn, trainloader, model, epoch)

                    for k, v in pm_avg.items():
                        pm_avg[k] += p_metrics[k].item()/len(trainloader)

                total_loss += loss.item()*bg.batch_size
                optimizer.step()
            total_loss = total_loss / len(trainloader.dataset)

            # val
            model.eval()
            val_metric = evaluate(
                valloader, model, device, metric_fn, metric_dtype, task, scaler).item()
            test_metric = evaluate(
                testloader, model, device, metric_fn, metric_dtype, task, scaler).item()

            if op(val_metric, best):
                best = val_metric
                test_on_best_valid = test_metric
                best_epoch = epoch
                if save_path is not None:
                    if train_args['augment']:
                        aug_method = train_args['augment']['method']
                        aug_size = train_args['augment']['size']
                    else:
                        aug_method = 'no_aug'
                        aug_size = ''
                    torch.save(model.cpu().state_dict(),os.path.join(save_path,f'./{split_type}_{model_type}_seed{seed}_best_fold{fold}_{aug_method}_{aug_size}.pt'))
                        
                    model.to(device)
                print(f'{val_metric},{test_metric},save checkpoint at epoch {epoch}')

            if op(test_metric, best_test):
                best_test = test_metric
                best_epoch_test = epoch
                print('better test metric: ', epoch, best_test)

            logs = {f'train {train_args["loss_fn"]} loss': round(total_loss, 4),
                    f'valid {train_args["metric_fn"]}': round(val_metric, 4),
                    f'test {train_args["metric_fn"]}': round(test_metric, 4),
                    }
            
            if pm_avg['cos'] != 0:
                logs.update(pm_avg)


        print(f'best epoch {best_epoch} for fold {fold}, val {train_args["metric_fn"]}:{best}, test: {test_on_best_valid} \
                best test_epoch {best_epoch_test} for fold {fold} test {train_args["metric_fn"]}:{best_test}')

        #wandb.finish()
        
        
def run(args):
   
    config_path =   args.config_path
    model_type  =   args.model_type
    device  =   args.device
    split_type  =   args.split_type
    target  =   args.target
    before_encoder  =   args.before_encoder
    init_method =   args.init_method
    aug_size    =   args.aug_size
    aug_method  =   args.aug_method
    save_model  =   args.save_model 
    aug_flag    =   args.aug_flag
    
    
    config = json.load(open(config_path, 'r'))
    train_args = config['train']
    train_args['group_name'] = config_path.split('/')[-1].split('.')[0]
    data_args = config['data']
    model_args = config['model']
    model_args['num_task'] = len(data_args['tasks'])
    seed = config['seed']
       
    model_args['model_type'] = model_type
    model_args['before_encoder'] = before_encoder
    #print("aug_flag", aug_flag, aug_flag=='True')
    print("aug_flag", aug_flag)
    print("save_model", save_model)
    if aug_flag:
        print("--------training with augmentation------->config")
        train_args['augment'] = {
        "method":aug_method,
        "init_method": init_method,
        "depth": 3,
        "size": aug_size,
        "constrain": "inf",
        "target": target,
        "lr":0.01
        }

    else:
        print("-----training without augmentation---------->config")
        # 不带augment参数的训练
        train_args['augment'] = None


    modify_config(config)
    print(config)
  
    print(model_args)
    train(data_args, train_args, model_args, seed, model_type, split_type, device, save_model)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Script to load configuration and parameters.")
    parser.add_argument('--config_path', type=str, default='configs/bbbp.json', choices=["configs/bbbp.json", "configs/esol.json", "configs/sider.json", "configs/tox21_base.json", "configs/bace.json", "configs/freesolv.json", "configs/lipophilicity.json"])
    parser.add_argument('--model_type', type=str, default='GCN', choices=['GCN']) 
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cuda:0', 'cuda:1', 'cpu'])
    parser.add_argument('--split_type', type=str, default='chem_scaffold', choices=['chem_scaffold', 'defined'])
    parser.add_argument('--target', type=str, default='node', choices=['node'])
    parser.add_argument('--before_encoder', type=bool, default=True) #TODO: action='store_true'?
    parser.add_argument('--init_method', type=str, default='mean', choices=['mean'])
    parser.add_argument('--aug_size', type=float, default=0.1) #0.01, 0.1, 0.5
    parser.add_argument('--aug_method', type=str, default='gap') #gap, random
    parser.add_argument('--save_model', action='store_true') 
    parser.add_argument('--aug_flag', action='store_false', default=True)  #default aug; adding --aug_flag means no aug
    args=parser.parse_args()
    
   
    run(args)


