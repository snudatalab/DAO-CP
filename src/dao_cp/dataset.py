"""
DAO-CP: Data-adaptive online CP decomposition (PLOS ONE 2021)

Authors:
- Sangjun Son      (lucetre@snu.ac.kr), Seoul National University
- Yongchan Park (wjdakf3948@snu.ac.kr), Seoul National University
- Minyong Cho   (chominyong@gmail.com), Seoul National University
- U Kang             (ukang@snu.ac.kr), Seoul National University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""
import csv
import os
import wget
import pandas as pd
import numpy as np
import tensorly as tl

from scipy.io import loadmat

def get_dataset(name):
    
    if name == 'synthetic':
        synthetic = tl.tensor(np.zeros([1000, 10, 20, 30], dtype='f'))
        for i in range(200):
            with open('../data/synthetic/data{}.tensor'.format(i)) as file:
                reader = csv.reader(file, delimiter='\t')    
                for row in reader:
                    indices = [[index] for index in np.int64(np.asarray(row[:-1]))-1]
                    synthetic[tuple(indices)] = np.double(row[-1])
        return synthetic
    
    elif name == 'video':
        video = tl.tensor(np.zeros([205, 240, 320, 3], dtype='d'))
        for i in range(41):
            with open('../data/video/video{}.tensor'.format(i)) as file:
                reader = csv.reader(file, delimiter='\t')    
                for row in reader:
                    indices = [[index] for index in np.int64(np.asarray(row[:-1]))-1]
                    video[tuple(indices)] = np.double(row[-1])
        return video
    
    elif name == 'stock':
        stock = tl.tensor(np.zeros([3089, 140, 5], dtype='d'))
        with open('../data/stock/KOSPI140.tensor') as file:
            reader = csv.reader(file, delimiter='\t')    
            for row in reader:
                indices = np.asarray([index for index in np.int64(np.asarray(row[:-1]))])[[1, 0, 2]]
                stock[tuple(indices)] = np.double(row[-1])
        return stock
    
    elif name == 'korea':
        fname = '../data/korea/air_quality.tensor'
        if not os.path.isfile(fname):
            os.mkdir('../data/korea/')
            print('Online download...')
            url = 'https://github.com/snudatalab/DAO-CP/releases/download/v0.1/air_quality.tensor'
            wget.download(url, fname)
        
        df = pd.read_csv(fname, delimiter='\t', header=None)
        dims = df[[0,1,2]].max()+1
        korea = np.empty(dims) * np.nan
        for i, row in df.iterrows():
            indices = [[index] for index in np.int64(np.asarray(row[:-1]))]
            korea[tuple(indices)] = np.double(row[3])
        avg = []
        for i in range(korea.shape[2]):
            avg.append(np.nanmean(korea[:,:,i]))
        inds = np.where(np.isnan(korea))
        for ind in zip(inds[0], inds[1], inds[2]):
            korea[ind] = avg[ind[-1]]
        return korea

    elif name == 'hall':
        hall = loadmat('../data/hall/hall1-200.mat')['XO']
        hall = np.moveaxis(hall, -1, 0)
        hall = hall.reshape(200, 144, 176, order='F')
        return tl.tensor(hall, dtype='f')
    