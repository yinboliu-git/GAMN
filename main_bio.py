import copy
import setproctitle
import multiprocessing
import torch
import numpy as np
from sklearn.model_selection import ParameterGrid
import os
from train_model import CV_train
from data_utils import read_data
import warnings
import datetime
from kMeans_Get_Di import get_di
import warnings
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Set the process title for identification purposes
setproctitle.setproctitle("DiGAMN_Bio_gsearch")


def set_seed(seed):
    """Set the seed for all relevant random number generators to ensure reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Config:
    """Configuration class to store model and training parameters."""

    def __init__(self):
        self.datapath = './data/'
        self.save_file = './save_file/'
        self.kfold = 5
        self.self_encode_len = 256
        self.globel_random = 100
        self.other_args = {'arg_name': [], 'arg_value': []}
        self.epochs = 5000
        self.print_epoch = 20
        self.lr = 0.01
        self.reg = 0.0005
        self.decay = 0.985
        self.decay_step = 1
        self.patience = 30
        self.num_rels = 0
        self.dropout = 0.1
        self.MGAT_layer = 2
        self.GAT_layer = 4
        self.n_hid = 16
        self.Memory_size = 8
        self.xr = None
        self.di_num = 5


def set_attr(config, param_search):
    """Generate configurations based on the parameter grid for hyperparameter search."""
    param_grid_list = list(ParameterGrid(param_search))
    for param in param_grid_list:
        new_config = copy.deepcopy(config)
        new_config.other_args = {'arg_name': [], 'arg_value': []}
        for key, value in param.items():
            setattr(new_config, key, value)
            new_config.other_args['arg_name'].append(key)
            new_config.other_args['arg_value'].append(value)
            print(f"{key}: {value}")
        yield new_config


class Data_paths:
    """Class to manage data paths for different datasets."""

    def __init__(self, DATAFIELD,di):
        base_path = f'./data{di}/{DATAFIELD}/'
        self.md = base_path + 'm_d.csv'
        self.mc = base_path + 'm_sc.csv'
        self.dc = base_path + 'd_sc.csv'
        self.mm = [base_path + 'm_gs.csv', base_path + 'm_ss.csv']
        self.dd = [base_path + 'd_gs.csv', base_path + 'd_ss.csv']


def mul_func(data_tuple):
    """Function to handle model training and evaluation for a single configuration."""
    start_time = datetime.datetime.now()
    warnings.filterwarnings('ignore', message='TypedStorage is deprecated.')
    data_name, params, data_tuple = data_tuple
    set_seed(521)

    key_file = ''.join(f"{key}{getattr(params, key)}" for key in best_param_search.keys())
    save_file = f"{data_name}_5cv_data_{params.epochs}_{key_file}"
    print(f'-----Starting task {save_file}-----')

    data_idx, auc_name = CV_train(params, data_tuple)  # Cross-validation training

    if not os.path.exists(params.save_file):
        os.makedirs(params.save_file, exist_ok=True)
    np.save(os.path.join(params.save_file, f"{save_file}.npy"), data_idx)

    data_mean = data_idx[:, :, 2:].mean(axis=0)
    idx_max = data_mean[:, 1].argmax()
    print(f'\nMax value for {data_name}: {data_mean[idx_max, :]}')
    print(f'-----Task {save_file} completed-----')

    print_execution_time(start_time)


def print_execution_time(start_time):
    """Print the execution time of the task."""
    execution_time = datetime.datetime.now() - start_time
    hours, remainder = divmod(execution_time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total execution time: {hours} hours, {minutes} minutes, {seconds} seconds")

best_param_search = {
    'MGAT_layer': [4],
    'GAT_layer': [4, ],
    'n_hid': [8, ],
    'Memory_size': [8, ],
}

# Define the parameter search space
best_param_search = {
    'MGAT_layer': [1, 2, 3, 4, 5, 6, 7],  # 2
    'GAT_layer': [2, 4, 6, 8],  # 4
    'n_hid': [4, 8, 16, 32],  # emb_number # 16
    'Memory_size': [4, 8, 16, 32],  # 8
}



use_multiprocessing = False
if __name__ == '__main__':
    data_names = ['AMHMDA', 'DAmiRLocGNet', 'MDA-CF', 'VGAMF']
    print(best_param_search)
    params_all = Config()
    param_generator = set_attr(params_all, best_param_search)
    params_list = []
    data_id = 0

    get_di(params_all.di_num)
    # Load the dataset for the specified data ID
    data_tuple = read_data(params_all, file_pair=Data_paths(data_names[data_id],di=params_all.di_num))

    # Generate configurations from the parameter search grid
    for params in param_generator:
        params_list.append((data_names[data_id], copy.deepcopy(params), data_tuple))
        print(f"Configuration set {len(params_list)} prepared...")

    print(f"A total of {len(params_list)} configurations will be processed...")

    # Determine if multiprocessing should be used

    if use_multiprocessing:
        multiprocessing.set_start_method('spawn')
        with multiprocessing.Pool(processes=min(len(params_list), os.cpu_count())) as pool:
            pool.map(mul_func, params_list)
    else:
        # Process each configuration sequentially without multiprocessing
        for config in params_list:
            mul_func(config)

    print("All tasks completed.")
