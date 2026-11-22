'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from local_code.base_class.dataset import dataset
import torch
import numpy as np
import scipy.sparse as sp

# Per-class sizes from the stage 5 ReadMe.
SPLIT_SIZES = {
    'cora':     {'train_per_class': 20, 'test_per_class': 150},
    'citeseer': {'train_per_class': 20, 'test_per_class': 200},
    'pubmed':   {'train_per_class': 20, 'test_per_class': 200},
}
# Validation is a random sample of the nodes not used in train or test.
# It is not class-balanced.
VAL_SIZE = 500


class Dataset_Loader(dataset):
    data = None
    dataset_name = None

    def __init__(self, dName=None, dDescription=None, seed=None):
        super(Dataset_Loader, self).__init__(dName, dDescription)
        self.seed = seed

    def adj_normalize(self, mx):
        """normalize sparse matrix: D^-1/2 M D^-1/2"""
        rowsum = np.array(mx.sum(1)).flatten()
        r_inv = np.power(rowsum, -0.5)
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        return r_mat_inv.dot(mx).dot(r_mat_inv)

    def row_normalize(self, mx):
        """row-normalize sparse matrix so each row sums to 1"""
        rowsum = np.array(mx.sum(1)).flatten()
        r_inv = np.power(rowsum, -1.0)
        r_inv[np.isinf(r_inv)] = 0.
        return sp.diags(r_inv).dot(mx)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse COO tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def sample_split(self, labels, train_per_class, test_per_class, val_size, rng):
        """Class-balanced train/test sample, then a random validation sample from the rest."""
        train, test = [], []
        for c in np.unique(labels):
            idx = rng.permutation(np.where(labels == c)[0])
            need = train_per_class + test_per_class
            if len(idx) < need:
                raise ValueError('class {} has {} nodes, needs {}'.format(c, len(idx), need))
            train.extend(idx[:train_per_class])
            test.extend(idx[train_per_class:need])

        used = np.zeros(len(labels), dtype=bool)
        used[np.array(train, dtype=np.int64)] = True
        used[np.array(test, dtype=np.int64)] = True
        remaining = rng.permutation(np.where(~used)[0])
        if len(remaining) < val_size:
            raise ValueError('only {} nodes left for validation, needs {}'.format(len(remaining), val_size))
        val = remaining[:val_size]

        return (np.sort(np.array(train, dtype=np.int64)),
                np.sort(val.astype(np.int64)),
                np.sort(np.array(test, dtype=np.int64)))

    def load(self):
        """Load citation network dataset"""
        print('Loading {} dataset...'.format(self.dataset_name))

        if self.dataset_name not in SPLIT_SIZES:
            raise ValueError('Unknown dataset name: {}'.format(self.dataset_name))

        folder = self.dataset_source_folder_path

        # node file: <node_id> <features...> <label>
        raw = np.genfromtxt("{}/node".format(folder), dtype=np.dtype(str))
        node_ids = np.array(raw[:, 0], dtype=np.int32)
        features = sp.csr_matrix(raw[:, 1:-1].astype(np.float32))
        classes, labels = np.unique(raw[:, -1], return_inverse=True)

        id_to_idx = {j: i for i, j in enumerate(node_ids)}
        reverse_idx_map = {i: j for i, j in enumerate(node_ids)}

        # link file: each line "A B" means B points to A
        edges_raw = np.loadtxt("{}/link".format(folder), dtype=np.int32, ndmin=2)
        missing = ~np.isin(edges_raw, node_ids)
        if missing.any():
            raise ValueError('{} link endpoints are not in the node file'.format(int(missing.sum())))
        edges = np.array([[id_to_idx[a], id_to_idx[b]] for a, b in edges_raw], dtype=np.int64)

        # build a symmetric, binary adjacency matrix without self-links
        n = len(node_ids)
        adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
        adj.data[:] = 1.0            # duplicate lines collapse to one edge
        adj = adj.maximum(adj.T)     # symmetric
        adj.setdiag(0)               # drop self-links so each node gets exactly one self-loop
        adj.eliminate_zeros()
        norm_adj = self.adj_normalize(adj + sp.eye(n))

        # row-normalize features
        features = self.row_normalize(features)

        # convert to torch
        features = torch.from_numpy(features.toarray().astype(np.float32))
        labels_t = torch.LongTensor(labels)
        adj_t = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

        # class-balanced train/test, random validation
        sizes = SPLIT_SIZES[self.dataset_name]
        rng = np.random.RandomState(self.seed)
        idx_train, idx_val, idx_test = self.sample_split(
            labels, sizes['train_per_class'], sizes['test_per_class'], VAL_SIZE, rng)

        train_test_val = {
            'idx_train': torch.LongTensor(idx_train),
            'idx_test': torch.LongTensor(idx_test),
            'idx_val': torch.LongTensor(idx_val),
        }
        graph = {
            'node': id_to_idx,
            'edge': edges,
            'X': features,
            'y': labels_t,
            'classes': classes,
            'utility': {'A': adj_t, 'reverse_idx': reverse_idx_map},
        }
        return {'graph': graph, 'train_test_val': train_test_val}