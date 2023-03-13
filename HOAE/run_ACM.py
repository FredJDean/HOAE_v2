import os
import time
import argparse
import random
import dgl
import psutil
import scipy.sparse
import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from pytorchtools import EarlyStopping
from data import load_ACM_data
from tools import index_generator, evaluate_results_nc, parse_minibatch, parse_mask
from models import GC_HAN_AC

ap = argparse.ArgumentParser(description='HOAE testing for the ACM dataset')
ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
ap.add_argument('--epoch', type=int, default=300, help='Number of epochs. Default is 100.')
ap.add_argument('--patience', type=int, default=30, help='Patience. Default is 5.')
ap.add_argument('--aggregator', type=str, default="mean", help='Heterogeneous information aggregate layer, att or mean')
ap.add_argument('--feats-drop-rate', type=float, default=0.2, help='The dropout of attribute completion.')
ap.add_argument('--ac_layers', type=int, default=1, help='layers of attribute completion. Default is 1.')
ap.add_argument('--save-postfix', default='ACM', help='Postfix for the saved model and result. Default is ACM.')
ap.add_argument('--cuda', action='store_true', default=True, help='Using GPU or not.')
ap.add_argument('--slope', type=float, default=0.05)
ap.add_argument('--schedule_step', type=int, default=400)
args = ap.parse_args()
print(args)
hidden_dim = args.hidden_dim
num_heads = args.num_heads
num_epochs = args.epoch
patience = args.patience
save_postfix = args.save_postfix
ac_drop = args.feats_drop_rate
ac_layers = args.ac_layers
agg = args.aggregator
is_cuda = args.cuda
slope = args.slope
# random seed
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if is_cuda:
    print('Using CUDA')
    torch.cuda.manual_seed(seed)

# Params
GCN_Hidden_dim = 64
out_dim = 16
# DROPOUT
dropout_rate = 0.5
lr = 0.001
weight_decay = 0.001
device = torch.device('cuda:0')

features_list, adjM, type_mask, labels, train_val_test_idx, meta_data = load_ACM_data()
labels_plt =labels
features_list = [torch.FloatTensor(features).to(device) for features in features_list]
in_dims = [features.shape[1] for features in features_list]
num_classes = int(labels.max()) + 1
adjM = torch.FloatTensor(adjM).to(device)
adjMX = adjM.data.cpu().numpy()
adjMX = scipy.sparse.csr_matrix(adjMX)
g = dgl.DGLGraph(adjMX + (adjMX.T))
g = dgl.remove_self_loop(g)
g = dgl.add_self_loop(g)
g = g.to(device)
pap = meta_data[0]
psp = meta_data[1]
papsp = meta_data[2]
remove_self_loop = False
if (remove_self_loop):
    num_nodes = labels.shape[0]
    pap = scipy.sparse.csr_matrix(pap - np.eye(num_nodes))
    psp = scipy.sparse.csr_matrix(psp - np.eye(num_nodes))
    papsp = scipy.sparse.csr_matrix(papsp - np.eye(num_nodes))
    pass
else:
    pap = scipy.sparse.csr_matrix(pap)
    psp = scipy.sparse.csr_matrix(psp)
    papsp = scipy.sparse.csr_matrix(papsp)
pap_g = dgl.DGLGraph(pap).to(device)
psp_g = dgl.DGLGraph(psp).to(device)
papsp_g = dgl.DGLGraph(papsp).to(device)
meta_graph = [pap_g, psp_g]
num_metapaths = len(meta_graph)
labels = torch.LongTensor(labels).to(device)
train_idx = train_val_test_idx['train_idx']
train_idx = np.sort(train_idx)
val_idx = train_val_test_idx['val_idx']
val_idx = np.sort(val_idx)
test_idx = train_val_test_idx['test_idx']
test_idx = np.sort(test_idx)
print('data load finish')
net = GC_HAN_AC(num_metapaths, in_dims, hidden_dim, out_dim, num_classes, num_heads, g,
                meta_graph, GCN_Hidden_dim, device, ac_drop, ac_layers, agg, dropout_rate, is_cuda)
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
print('model init finish\n')

# training loop
print('training...')
net.train()
early_stopping = EarlyStopping(patience=patience, verbose=True,
                               save_path='checkpoint/checkpoint_{}.pt'.format(save_postfix))
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.schedule_step, max_lr=1e-3,
                                                pct_start=0.05)
train_step = 0
for epoch in range(num_epochs):
    # training
    t = time.time()
    net.train()
    train_loss_avg = 0

    logits, _= net((adjM, features_list, type_mask))
    logp = F.log_softmax(logits, 1)
    loss_classification = F.nll_loss(logp[train_idx], labels[train_idx])
    train_loss = loss_classification
    # auto grad
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    train_time = time.time() - t
    train_step += 1
    print(u'当前进程的内存使用:%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
    scheduler.step(train_step)
    # validation
    t = time.time()
    net.eval()
    val_loss = 0
    with torch.no_grad():
        logits, _ = net((adjM, features_list,type_mask))
        logp = F.log_softmax(logits, 1)
        val_loss = F.nll_loss(logp[val_idx], labels[val_idx])
        val_time = time.time() - t

    print(
        'Epoch {:05d} | Train_Loss {:.4f} | Train_Time(s) {:.4f} | Val_Loss {:.4f} | Val_Time(s) {:.4f}'.format(
            epoch, train_loss, train_time, val_loss, val_time))

    # early stopping
    early_stopping(val_loss, net)
    if early_stopping.early_stop:
        print('Early stopping!')
        break

# testing with evaluate_results_nc
print('\ntesting...')
net.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format(save_postfix)))
net.eval()
test_embeddings = []
t_start = time.time()
with torch.no_grad():
    _, embeddings = net(
        (adjM, features_list, type_mask))
    t_end = time.time()
    print("Time: {:.4f}".format(t_end - t_start))
    test_embeddings.append(embeddings)
    test_embeddings = torch.cat(test_embeddings, 0)
    embeddings = test_embeddings.detach().cpu().numpy()
    svm_macro, svm_micro, nmi, ari = evaluate_results_nc(embeddings[test_idx], labels[test_idx].cpu().numpy(), num_classes)
