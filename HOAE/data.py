import networkx as nx
import numpy as np
import scipy.sparse
import pickle
import torch


def load_ACM_data(prefix='data/preprocessed/ACM_processed'):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1_onehot = np.eye(7167)
    features_2_onehot = np.eye(60)
    features_1 = features_1_onehot
    features_2 = features_2_onehot
    adjM = scipy.sparse.load_npz(prefix + '/adjM.npz').toarray()
    pap = np.load(prefix + '/0/0-1-0.npy')
    psp = np.load(prefix + '/0/0-2-0.npy')
    papsp = np.load(prefix + '/0/0-1-0-2-0.npy')
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    return [features_0, features_1, features_2], \
           adjM, \
           type_mask, \
           labels, \
           train_val_test_idx, [pap, psp, papsp]
