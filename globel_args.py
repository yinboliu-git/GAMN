import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os

def set_seed(seed):
    torch.manual_seed(seed)
    #进行随机搜索的这个要注释掉
    # random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

dev_id = 0
torch.cuda.set_device(dev_id)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

