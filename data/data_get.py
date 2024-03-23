import numpy as np
import pandas as pd


class Data_paths:
    def __init__(self, DATAFIEL):
        self.paths = '../data/'+DATAFIEL+'/'
        self.md = self.paths + 'm_d.csv'
        self.mc = self.paths + 'm_sc.csv'
        self.dc = self.paths + 'd_sc.csv'
        self.mm = [self.paths + 'm_gs.csv', self.paths + 'm_ss.csv']
        self.dd = [self.paths + 'd_gs.csv', self.paths + 'd_ss.csv']

data_list = ['AMHMDA', 'DAmiRLocGNet', 'MDA-CF', 'VGAMF']
file_pair = Data_paths('AMHMDA')

adj_matrix = pd.read_csv(file_pair.md, header=None, index_col=None).values


print(adj_matrix)