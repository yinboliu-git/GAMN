import torch, dgl
import numpy as np
from global_args import device
import pandas as pd

def get_metrics(real_score, predict_score):
    real_score, predict_score = real_score.flatten(), predict_score.flatten()
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num*np.arange(1, 1000)/1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN

    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]

    # np.savetxt(roc_path.format(i), ROC_dot_matrix)

    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])

    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]

    # np.savetxt(pr_path.format(i), PR_dot_matrix)

    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])

    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    print( ' auc:{:.4f} ,aupr:{:.4f},f1_score:{:.4f}, accuracy:{:.4f}, recall:{:.4f}, specificity:{:.4f}, precision:{:.4f}'.format( auc[0, 0],aupr[0, 0], f1_score, accuracy, recall, specificity, precision))
    return [real_score, predict_score,auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision], \
           ['y_true', 'y_score', 'auc', 'prc', 'f1_score', 'acc', 'recall', 'specificity', 'precision']


def get_y_data(file_pair, params):
    adj_matrix = pd.read_csv(file_pair.md, header=None, index_col=None).values

    # 将邻接矩阵转换为张量，这里要将列索引加上总节点数，使得列节点的索引不与行节点重复
    edge_index_pos = np.column_stack(np.argwhere(adj_matrix != 0))
    edge_index_pos = torch.tensor(edge_index_pos, dtype=torch.long)

    # 获取不相关连接 为了构建训练数据
    edge_index_neg = np.column_stack(np.argwhere(adj_matrix == 0))
    edge_index_neg = torch.tensor(edge_index_neg, dtype=torch.long)

    # 创建数据

    num_pos_edges_number = edge_index_pos.shape[1]
    selected_neg_edge_indices = torch.randint(high=edge_index_neg.shape[1], size=(num_pos_edges_number,),
                                               dtype=torch.long)
    edge_index_neg_selected = edge_index_neg[:, selected_neg_edge_indices]
    edg_index_all = torch.cat((edge_index_pos, edge_index_neg_selected), dim=1)
    y = torch.cat((torch.ones((edge_index_pos.shape[1], 1)),
                   torch.zeros((edge_index_neg_selected.shape[1], 1))), dim=0)  # 将所有y值设置为1,0
    # 获取平衡样本

    if params.xr == 'randn':
        xe_1 = torch.randn((adj_matrix.shape[0], params.self_encode_len))  # 没有节点特征，所以设置为1 m
        xe_2 = torch.randn((adj_matrix.shape[1], params.self_encode_len))  # 没有节点特征，所以设置为1 d
    elif params.xr=='self_encode':
        xe_1 = []
        xe_2 = []

        if len(file_pair.mm) + len(file_pair.dd) > 0:
            for i in range(0, len(file_pair.mm)):
                xe_1.append(pd.read_csv(file_pair.mm[i], header=None, index_col=None).values)

            for j in range(0, len(file_pair.dd)):
                xe_2.append(pd.read_csv(file_pair.dd[i], header=None, index_col=None).values)

            xe_1 = torch.tensor(np.array(xe_1).mean(0), dtype=torch.float32)
            xe_2 = torch.tensor(np.array(xe_2).mean(0), dtype=torch.float32)
    else:
        xe_1, xe_2 = None, None

    edg_index_all[-1] = edg_index_all[-1] + adj_matrix.shape[0]
    if xe_1 == None:
        return None, {'y': y, 'y_edge': edg_index_all}
    return {'x1':xe_1,'x2':xe_2}, {'y':y, 'y_edge':edg_index_all}



def read_data(parms, file_pair):
    """
        etype:
            0 m-d
            1 d-m
            2 m-c
            3 c-m
            4 d-c
            5 c-d
    """
    adj_matrix = pd.read_csv(file_pair.md, header=None, index_col=None).values
    adj_matrix_mc = pd.read_csv(file_pair.mc, header=None, index_col=None).values
    adj_matrix_dc = pd.read_csv(file_pair.dc, header=None, index_col=None).values

    adj_matrix = torch.tensor(adj_matrix)
    adj_matrix_mc = torch.tensor(adj_matrix_mc)
    adj_matrix_dc = torch.tensor(adj_matrix_dc)
    bm_numb = parms.m_numb = adj_matrix.shape[0]
    bd_numb = parms.d_numb = adj_matrix.shape[1]
    bcm_numb = parms.cm_numb = adj_matrix_mc.shape[1]
    bcd_numb = parms.cd_numb = adj_matrix_dc.shape[1]

    src, dst, etype = [], [], []

    """ m_d.csv """
    bm = 0
    bd = bm + bm_numb
    temp = adj_matrix.nonzero()
    mids, dids = temp[:,0], temp[:,1]
    src += (bm + mids).tolist()
    dst += (bd + dids).tolist()
    etype += [parms.num_rels] * (mids.shape[0])
    parms.num_rels += 1

    src += (bd + dids).tolist()
    dst += (bm + mids).tolist()
    etype += [parms.num_rels] * (mids.shape[0])
    parms.num_rels +=1
    parms.num_nodes = bd + bd_numb

    """ m_c.csv """
    bcm = bd + bd_numb
    temp = adj_matrix_mc.nonzero()
    mids, cmids = temp[:,0], temp[:,1]
    src += (bm + mids).tolist()
    dst += (bcm + cmids).tolist()
    etype += [parms.num_rels] * (mids.shape[0])
    parms.num_rels +=1

    src += (bcm + cmids).tolist()
    dst += (bm + mids).tolist()
    etype += [parms.num_rels] * (mids.shape[0])
    parms.num_rels+=1
    parms.num_nodes = bcm + bcm_numb


    """ d_c.csv """
    bcd = bcm + bcm_numb
    temp = adj_matrix_dc.nonzero()
    dids, cdids = temp[:,0], temp[:,1]
    src += (bd + dids).tolist()
    dst += (bcd + cdids).tolist()
    etype += [parms.num_rels] * (dids.shape[0])
    parms.num_rels+=1

    src += (bcd + cdids).tolist()
    dst += (bd + dids).tolist()
    etype += [parms.num_rels] * (dids.shape[0])
    parms.num_rels+=1
    parms.num_nodes = bcd + bcd_numb

    graph = dgl.graph((src, dst), num_nodes=parms.num_nodes).to(device)
    graph.edata['type'] = torch.LongTensor(etype).to(device)

    # 使用add_edges方法添加自环边
    self_loop_src = torch.arange(parms.num_nodes).to(device)
    self_loop_dst = torch.arange(parms.num_nodes).to(device)
    graph.add_edges(self_loop_src, self_loop_dst)
    x_encode, y = get_y_data(file_pair,parms)

    return graph, x_encode, y
