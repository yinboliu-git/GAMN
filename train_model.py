import datetime
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from global_args import device
from data_utils import get_metrics
import copy
import torch.nn as nn
from model import DiGAMN_Model

class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking Loss for optimizing in recommendation systems."""
    def __init__(self, lamb_reg):
        super(BPRLoss, self).__init__()
        self.lamb_reg = lamb_reg

    def forward(self, pos_preds, neg_preds, *reg_vars):
        batch_size = pos_preds.size(0)
        bpr_loss = -0.5 * (pos_preds - neg_preds).sigmoid().log().sum() / batch_size
        reg_loss = torch.tensor([0.], device=bpr_loss.device)
        for var in reg_vars:
            reg_loss += self.lamb_reg * 0.5 * var.pow(2).sum()
        reg_loss /= batch_size
        loss = bpr_loss + reg_loss
        return loss, [bpr_loss.item(), reg_loss.item()]

def train_model(graph_init, x_encode, y, train_idx, test_idx, param):
    """Function to train the model based on provided indices and parameters."""
    graph = graph_init.clone()
    model = DiGAMN_Model(param).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0002)

    src_train, tgt_train = y['y_edge'][0][train_idx], y['y_edge'][1][train_idx]
    src_test, tgt_test = y['y_edge'][0][test_idx], y['y_edge'][1][test_idx]

    # Edge removal for test edges
    true_src = y['y_edge'][0][test_idx[(y['y'][test_idx] == 1).reshape(-1)]]
    true_tgt = y['y_edge'][1][test_idx[(y['y'][test_idx] == 1).reshape(-1)]]
    for _src, _tgt in zip(true_src, true_tgt):
        graph.remove_edges((_src, _tgt))

    auc_list = []
    start_time = datetime.datetime.now()
    for epoch in range(1, param.epochs + 1):
        optimizer.zero_grad()
        rep = model(graph, x_encode).to(device)
        preds = model.predict(rep[src_train], rep[tgt_train]).to(device)
        loss = F.binary_cross_entropy_with_logits(preds, y['y'][train_idx].reshape(-1,).to(device))

        loss.backward()
        optimizer.step()

        if epoch % param.print_epoch == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            model.eval()
            with torch.no_grad():
                preds = model.predict(model(graph, x_encode)[src_test], model(graph, x_encode)[tgt_test])
                out_pred = preds.to('cpu').detach().numpy()
                y_true = y['y'][test_idx].to('cpu').detach().numpy()
                auc = roc_auc_score(y_true, out_pred)
                print('AUC:', auc)
                auc_idx, auc_name = get_metrics(y_true, out_pred)
                auc_idx.extend(param.other_args['arg_value'])
                auc_idx.append(epoch)
                auc_list.append(auc_idx)
                print_execution_time(start_time, param.print_epoch)
            model.train()

    auc_name.extend(param.other_args['arg_name'])
    return auc_list, auc_name

def print_execution_time(start_time, epoch_interval):
    """Print the execution time for a set of epochs."""
    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total execution time for {epoch_interval} epochs: {hours} hours, {minutes} minutes, {seconds} seconds")

def CV_train(param, args_tuple=()):
    """Cross-validation training setup."""
    graph, x_encode, y = args_tuple
    k_fold = param.kfold
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=param.globel_random)
    kf_auc_list = []

    for train_idx, test_idx in kf.split(np.arange(y['y'].shape[0])):
        print(f'Running fold {len(kf_auc_list) + 1} of {k_fold}...')
        auc_idx,auc_name = train_model(graph, x_encode, y, train_idx, test_idx, param)
        kf_auc_list.append(auc_idx)

    data_idx = np.array(kf_auc_list)
    # Optionally, save the cross-validation results for further analysis
    # np.save(os.path.join(param.save_file, 'data_idx_mean.npy'), data_idx.mean(axis=0))
    return data_idx, auc_name
