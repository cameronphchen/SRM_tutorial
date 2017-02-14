"""Distributed Searchlight Example
example usage: mpirun -n 8 python3 searchlightsrm_time_segment_matching.py

Authors: Hejia Zhang (Princeton)
"""

import numpy as np
from mpi4py import MPI
import sys
import scipy.io as sio
from scipy.stats import stats
import warnings

sys.path.append('/Users/ChimatChen/brainiak')
from brainiak.searchlight.searchlight import Searchlight
from brainiak.funcalign.srm import SRM


# parameters
sl_rad = 1 #searchlight length (of each edge) will be 1+2*sl_rad
max_blk_edge = 1 #won't change computational results, has effect on MPI processing
nfeature = 10
n_iter = 10

# sanity check
if sl_rad <= 0 or max_blk_edge <= 0:
  raise ValueError('sl_rad and max_blk_edge must be positive')
  #return None
if nfeature > (1+2*sl_rad)**3:
  print ('nfeature truncated')
  nfeature = int((1+2*sl_rad)**3)

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# load data
movie_file = sio.loadmat('data/sherlock/movie_data.mat')
movie_data = movie_file['data']

# Dataset size parameters
dim1,dim2,dim3,ntr,nsubj = movie_data.shape

# preprocess data
all_data = [] # first half train, second half test
for s in range(nsubj):
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    train_tmp = np.nan_to_num(stats.zscore(movie_data[:,:,:,:int(ntr/2),s],axis=3,ddof=1))
    test_tmp = np.nan_to_num(stats.zscore(movie_data[:,:,:,int(ntr/2):,s],axis=3,ddof=1))
  all_data.append(np.concatenate((train_tmp,test_tmp),axis=3))

# Generate mask
mask = np.ones((dim1,dim2,dim3), dtype=np.bool)

# Create searchlight object
sl = Searchlight(sl_rad=sl_rad, max_blk_edge=max_blk_edge)

# Distribute data to processes
sl.distribute(all_data, mask)
sl.broadcast([n_iter,nfeature])

# time segment matching experiment
def timesegmentmatching_accuracy(data, win_size=6): 
  nsubjs = len(data)
  (ndim, nsample) = data[0].shape
  accu = np.zeros(shape=nsubjs)

  nseg = nsample - win_size 
  # mysseg prediction prediction
  trn_data = np.zeros((ndim*win_size, nseg))

  # the trn data also include the tst data, but will be subtracted when 
  # calculating A
  for m in range(nsubjs):
      for w in range(win_size):
          trn_data[w*ndim:(w+1)*ndim,:] += data[m][:,w:(w+nseg)]

  for tst_subj in range(nsubjs):
    tst_data = np.zeros((ndim*win_size, nseg))
    for w in range(win_size):
      tst_data[w*ndim:(w+1)*ndim,:] = data[tst_subj][:,w:(w+nseg)]

    A =  np.nan_to_num(stats.zscore((trn_data - tst_data),axis=0, ddof=1))
    B =  np.nan_to_num(stats.zscore(tst_data,axis=0, ddof=1))
    # normalize A and B
    A = A/np.linalg.norm(A,axis=0)
    B = B/np.linalg.norm(B,axis=0)

    corr_mtx = B.T.dot(A)
    for i in range(nseg):
      for j in range(nseg):
        if abs(i-j)<win_size and i != j :
          corr_mtx[i,j] = -np.inf

    max_idx =  np.argmax(corr_mtx, axis=1)
    accu[tst_subj] = sum(max_idx == range(nseg)) / float(nseg)

  return accu

# Define voxel function
def sfn(l, msk, myrad, bcast_var):
  # extract training and testing data
  train_data = []
  test_data = []
  d1,d2,d3,ntr = l[0].shape
  nvx = d1*d2*d3
  for s in l:
    train_data.append(np.reshape(s[:,:,:,:int(ntr/2)],(nvx,int(ntr/2))))
    test_data.append(np.reshape(s[:,:,:,int(ntr/2):],(nvx,ntr-int(ntr/2))))
  # train an srm model 
  srm = SRM(bcast_var[0],bcast_var[1])
  srm.fit(train_data)
  # transform test data
  shared_data = srm.transform(test_data)
  for s in range(len(l)):
    shared_data[s] = np.nan_to_num(stats.zscore(shared_data[s],axis=1,ddof=1))
  # experiment
  accu = timesegmentmatching_accuracy(shared_data,6)

  return np.mean(accu),stats.sem(accu) # multiple outputs will be saved as tuples

# Run searchlight
global_outputs= sl.run_searchlight(sfn) # output is in shape (dim1,dim2,dim3)

# Unpack and save result
if rank == 0:
  acc = np.zeros((dim1,dim2,dim3))
  se = np.zeros((dim1,dim2,dim3))
  for i in range(dim1):
    for j in range(dim2):
      for k in range(dim3):
        if global_outputs[i][j][k] is not None:
          acc[i][j][k] = global_outputs[i][j][k][0]
          se[i][j][k] = global_outputs[i][j][k][1]
  print (acc)
  np.savez_compressed('data/sherlock/searchlight_srm_tsm_acc.npz',acc=acc,se=se)  




