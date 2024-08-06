import os
import numpy as np
import pandas as pd
import json
import operator
from tqdm import tqdm
import torch
from torch.optim import Adam


from augment_im import augment_loss
from mrgnn.featurize import get_featurizer
from mrgnn.utils import get_func, split, calculate_loss, calculate_metrics
from mrgnn.models import build_model
from mrgnn.data import MolGraphSet, subset_loader
from losses import SupConLoss

ROOT = os.path.dirname(os.path.abspath(__file__))


def evaluate(dataloader, model, device, metric_fn, metric_dtype, task, scaler=None):
    preds = []
    labels = []
    for idx, bg, label,rn in dataloader:
        bg, label = bg.to(device), label.type(metric_dtype)
        pred = model(bg).cpu().detach()
        if scaler is not None:
            pred = scaler.inverse_transform(pred)
        if task == 'classification':
            pred = torch.sigmoid(pred)
        elif task == 'multiclass':
            pred = torch.softmax(pred, dim=1)

        preds.append(pred.cpu())
        labels.append(label)

    preds = torch.cat(preds)
    labels = torch.cat(labels)
    m = calculate_metrics(preds, labels, metric_fn)

    return m


def train(data_args, train_args, model_args, seed, model_type, split_type='size', device='cuda'):
    epochs = train_args['epochs']
    num_fold = train_args['num_fold']
    device = device if torch.cuda.is_available() else 'cpu'
    batch_size = data_args['batch_size']
    task = data_args['task']
    lr = train_args['lr']

    df = pd.read_csv(os.path.join(ROOT, data_args['path']))

    af, bf = get_featurizer(model_args['featurizer'])
    model_args['atom_dim'] = af.dim
    model_args['bond_dim'] = bf.dim
    dataset = MolGraphSet(df, data_args['tasks'], af, bf)
    df = df.drop(dataset.invalid).reset_index(drop=True)

    best_result_picked = []
    best_result_infer = []

    torch.manual_seed(seed)
    for fold in range(num_fold):
        index = split(df, size=[0.7, 0.1, 0.2], seed=seed+fold, split_type=split_type)
        
        trainloader = subset_loader(dataset, index['train'], batch_size, shuffle=True)
        valloader = subset_loader(dataset, index['valid'], min(batch_size*10, len(index['valid'])), shuffle=False)
        testloader = subset_loader(dataset, index['test'], min(batch_size*10, len(index['test'])), shuffle=False)

        scaler = None

        print(f'dataset size, train: {len(trainloader.dataset)}, val: {len(valloader.dataset)}, test: {len(testloader.dataset)}')
        print(model_type)
        model = build_model(model_type, model_args).to(device)
        optimizer = Adam(model.parameters(), lr)

        temperature = train_args["temperature"]
        loss_fn = get_func(train_args['loss_fn'], 'loss')
        metric_fn = get_func(train_args['metric_fn'], 'metrics')
        cl_fn = SupConLoss(temperature=temperature, base_temperature=temperature).to(device)

        loss_dtype = torch.float32
        metric_dtype = torch.float32

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
            for idx, bg, labels, rn in trainloader:
                bg, labels = bg.to(device), labels.type(loss_dtype).to(device)
              
                optimizer.zero_grad()
               # print("train bg is {}".format(bg))

                if train_args.get('augment', None) is None:  # no perturb
                    preds = model(bg)
                    loss = calculate_loss(preds, labels, loss_fn).mean()
                else:
                    # loss with perturb
                    aug_arg = train_args['augment']
                    loss = augment_loss(aug_arg, idx, bg, labels, loss_fn, cl_fn, trainloader, model, epoch, rn, use_cl=train_args["use_cl"], lambda_cl=train_args["lambda_cl"]).mean()

                    # # loss with no perturb
                    # loss_wo_perturb = calculate_loss(model(bg), labels, loss_fn).mean()
                    # # loss with perturb
                    # aug_arg = train_args['augment']
                    # loss_w_perturb = augment_loss(aug_arg, idx, bg, labels, loss_fn, trainloader, model, epoch, rn).mean()
                    # loss = (loss_wo_perturb + loss_w_perturb) / 2

                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            total_loss = total_loss / len(trainloader.dataset)

            # val
            model.eval()
            val_metric = evaluate(valloader, model, device, metric_fn, metric_dtype, task, scaler).item()
            test_metric = evaluate(testloader, model, device, metric_fn, metric_dtype, task, scaler).item()

            if op(val_metric, best):
                best = val_metric
                test_on_best_valid = test_metric
                best_epoch = epoch
                print(f'{val_metric},{test_metric},save checkpoint at epoch {epoch}')

            if op(test_metric, best_test):
                best_test = test_metric
                best_epoch_test = epoch
                print('better test metric: ', epoch, best_test)

        print(f'best epoch {best_epoch} for fold {fold}, val {train_args["metric_fn"]}:{best}, test: {test_on_best_valid} \
                best test_epoch {best_epoch_test} for fold {fold} test {train_args["metric_fn"]}:{best_test}')
        best_result_picked.append(test_on_best_valid)
        best_result_infer.append(best_test)
    print(best_result_picked)
    print(best_result_infer)
    print(f"best picked:{np.mean(best_result_picked)}+/-{np.std(best_result_picked)}, best infered:{np.mean(best_result_infer)}+/-{np.std(best_result_infer)}")


if __name__ == '__main__':

    import sys
    config_path = sys.argv[1]
    model_type = sys.argv[2]
    device = sys.argv[3]
    split_type = sys.argv[4]

    config = json.load(open(config_path, 'r'))
    data_args = config['data']
    train_args = config['train']
    model_args = config['model']
    model_args['num_task'] = len(data_args['tasks'])
    model_args['model_type'] = model_type
    seed = config['seed']
    train_args['group_name'] = config_path.split('/')[-1].split('.')[0]

    print(config)
    train(data_args, train_args, model_args, seed, model_type, split_type, device)


