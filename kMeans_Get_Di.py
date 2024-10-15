import warnings
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
# Ignore all warnings
warnings.filterwarnings("ignore")
import os
import shutil
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder



def copy_files(src_folder, dest_folder):
    """
    Copies files from a source folder to a destination folder.

    Parameters:
    - src_folder (str): The path to the source folder.
    - dest_folder (str): The path to the destination folder.
    """
    # Check if the destination folder exists, create it if not
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Iterate over all files in the source folder
    for filename in os.listdir(src_folder):
        src_file = os.path.join(src_folder, filename)
        dest_file = os.path.join(dest_folder, filename)

        # Copy file to the destination folder if it is a file
        if os.path.isfile(src_file):
            shutil.copy2(src_file, dest_file)


def get_k_id(k, df):
    """
    Applies KMeans clustering to the given dataset and saves the one-hot encoded cluster IDs.

    Parameters:
    - k (int): The number of clusters for KMeans.
    - df (str): The dataset name.
    """
    print('Kmeans get Di sim of {df}')
    # Load m_gs.csv and m_ss.csv files
    m_gs = pd.read_csv(f'./data/{df}/m_gs.csv', header=None, index_col=None)
    m_ss = pd.read_csv(f'./data/{df}/m_ss.csv', header=None, index_col=None)

    # Load p_gs.csv and p_ss.csv files (p_gs and p_ss naming seems to refer to proteins, consider renaming for clarity)
    p_gs = pd.read_csv(f'./data/{df}/d_gs.csv', header=None, index_col=None)
    p_ss = pd.read_csv(f'./data/{df}/d_ss.csv', header=None, index_col=None)

    # Cluster m_gs using KMeans and encode cluster IDs as one-hot vectors
    kmeans_m_gs = KMeans(n_clusters=k)
    m_gc = kmeans_m_gs.fit_predict(m_gs)
    m_gc_onehot = OneHotEncoder(sparse=False).fit_transform(m_gc.reshape((-1, 1)))

    # Repeat clustering and encoding for m_ss, p_gs, and p_ss
    kmeans_m_ss = KMeans(n_clusters=k)
    m_sc = kmeans_m_ss.fit_predict(m_ss)
    m_sc_onehot = OneHotEncoder(sparse=False).fit_transform(m_sc.reshape((-1, 1)))

    kmeans_p_gs = KMeans(n_clusters=k)
    p_gc = kmeans_p_gs.fit_predict(p_gs)
    p_gc_onehot = OneHotEncoder(sparse=False).fit_transform(p_gc.reshape((-1, 1)))

    kmeans_p_ss = KMeans(n_clusters=k)
    p_sc = kmeans_p_ss.fit_predict(p_ss)
    p_sc_onehot = OneHotEncoder(sparse=False).fit_transform(p_sc.reshape((-1, 1)))

    # Save the one-hot encoded vectors to new files
    base_path = f'./data{k}/{df}'
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    np.savetxt(f'{base_path}/m_gc.csv', m_gc_onehot, delimiter=',')
    np.savetxt(f'{base_path}/m_sc.csv', m_sc_onehot, delimiter=',')
    np.savetxt(f'{base_path}/d_gc.csv', p_gc_onehot, delimiter=',')
    np.savetxt(f'{base_path}/d_sc.csv', p_sc_onehot, delimiter=',')

    # Copy other files from the original dataset folder to the new folder
    copy_files(f'./data/{df}/', base_path)



def get_di(di_num):
    """
    Applies get_k_id function across multiple datasets.

    Parameters:
    - di_num (int): The number of clusters for KMeans.
    """
    for df in ['AMHMDA', 'DAmiRLocGNet', 'MDA-CF', 'VGAMF']:
        get_k_id(di_num, df)


if __name__ == '__main__':
    get_di(2)
