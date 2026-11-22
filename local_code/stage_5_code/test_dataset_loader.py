'''
Unit test for the stage 5 dataset loader on the real Cora, Citeseer and PubMed files.
Run:  python test_dataset_loader.py
'''
import os
import sys
import unittest
import torch

torch.sparse.check_sparse_tensor_invariants.enable()

def _find_project_root():
    """Walk up from this file until the folder containing data/stage_5_data is found."""
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isdir(os.path.join(d, 'data', 'stage_5_data')):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise FileNotFoundError('could not find data/stage_5_data above this file')
        d = parent


PROJECT_ROOT = _find_project_root()
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'stage_5_data')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Adjust this import to match where you saved the loader.
from local_code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader

VAL_SIZE = 500

# Sizes and class counts from the ReadMe.
EXPECTED = {
    'cora':     {'n': 2708,  'features': 1433, 'classes': 7, 'train': 140, 'test': 1050,
                 'train_per_class': 20, 'test_per_class': 150},
    'citeseer': {'n': 3312,  'features': 3703, 'classes': 6, 'train': 120, 'test': 1200,
                 'train_per_class': 20, 'test_per_class': 200},
    'pubmed':   {'n': 19717, 'features': 500,  'classes': 3, 'train': 60,  'test': 600,
                 'train_per_class': 20, 'test_per_class': 200},
}


def make_loader(name, description="node classification dataset", seed=0):
    """Create a Dataset_Loader and set its source folder the way the course scripts do."""
    ld = Dataset_Loader(name, description, seed=seed)
    ld.dataset_source_folder_path = os.path.join(DATA_ROOT, name)
    return ld


def _max_abs(sparse_t):
    v = sparse_t.coalesce().values()
    return float(v.abs().max()) if v.numel() else 0.0


class TestDatasetLoader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.results = {name: make_loader(name).load() for name in EXPECTED}

    def test_shapes(self):
        for name, exp in EXPECTED.items():
            with self.subTest(dataset=name):
                g = self.results[name]['graph']
                self.assertEqual(g['X'].shape, (exp['n'], exp['features']))
                self.assertEqual(g['y'].shape, (exp['n'],))
                self.assertEqual(g['utility']['A'].shape, (exp['n'], exp['n']))
                self.assertEqual(len(g['node']), exp['n'])

    def test_number_of_classes(self):
        for name, exp in EXPECTED.items():
            with self.subTest(dataset=name):
                y = self.results[name]['graph']['y']
                self.assertEqual(len(torch.unique(y)), exp['classes'])

    def test_split_sizes(self):
        for name, exp in EXPECTED.items():
            with self.subTest(dataset=name):
                tt = self.results[name]['train_test_val']
                self.assertEqual(len(tt['idx_train']), exp['train'])
                self.assertEqual(len(tt['idx_test']), exp['test'])
                self.assertEqual(len(tt['idx_val']), VAL_SIZE)

    def test_split_is_class_balanced(self):
        for name, exp in EXPECTED.items():
            with self.subTest(dataset=name):
                res = self.results[name]
                y = res['graph']['y']
                tt = res['train_test_val']
                for c in range(exp['classes']):
                    train_count = int((y[tt['idx_train']] == c).sum())
                    test_count = int((y[tt['idx_test']] == c).sum())
                    self.assertEqual(train_count, exp['train_per_class'], msg='train class {}'.format(c))
                    self.assertEqual(test_count, exp['test_per_class'], msg='test class {}'.format(c))

    def test_splits_are_disjoint(self):
        for name in EXPECTED:
            with self.subTest(dataset=name):
                tt = self.results[name]['train_test_val']
                train = set(tt['idx_train'].tolist())
                test = set(tt['idx_test'].tolist())
                val = set(tt['idx_val'].tolist())
                self.assertEqual(train & test, set())
                self.assertEqual(train & val, set())
                self.assertEqual(test & val, set())

    def test_adjacency_is_symmetric_with_self_loops(self):
        for name, exp in EXPECTED.items():
            with self.subTest(dataset=name):
                A = self.results[name]['graph']['utility']['A']
                self.assertLess(_max_abs(A - A.t()), 1e-6)

                A_c = A.coalesce()
                rows, cols = A_c.indices()
                diag_count = int((rows == cols).sum())
                self.assertEqual(diag_count, exp['n'])

    def test_adjacency_values_in_range(self):
        for name in EXPECTED:
            with self.subTest(dataset=name):
                vals = self.results[name]['graph']['utility']['A'].coalesce().values()
                self.assertGreaterEqual(float(vals.min()), 0.0)
                self.assertLessEqual(float(vals.max()), 1.0 + 1e-6)

    def test_diagonal_matches_degree_normalization(self):
        # With self-links removed, node i has exactly one self-loop, so after
        # D^-1/2 (A + I) D^-1/2 its diagonal entry must be 1 / (1 + number of neighbors).
        for name, exp in EXPECTED.items():
            with self.subTest(dataset=name):
                g = self.results[name]['graph']
                n = exp['n']

                neighbors = [set() for _ in range(n)]
                for a, b in g['edge']:
                    a, b = int(a), int(b)
                    if a != b:
                        neighbors[a].add(b)
                        neighbors[b].add(a)
                expected = torch.tensor([1.0 / (1 + len(s)) for s in neighbors])

                A = g['utility']['A'].coalesce()
                rows, cols = A.indices()
                mask = rows == cols
                diag = torch.zeros(n)
                diag[rows[mask]] = A.values()[mask]

                self.assertTrue(torch.allclose(diag, expected, atol=1e-6),
                                msg='{} nodes have a wrong diagonal'.format(
                                    int((~torch.isclose(diag, expected, atol=1e-6)).sum())))

    def test_features_are_row_normalized(self):
        for name in EXPECTED:
            with self.subTest(dataset=name):
                X = self.results[name]['graph']['X']
                rowsum = X.sum(dim=1)
                nonzero = rowsum > 0
                self.assertTrue(torch.allclose(rowsum[nonzero],
                                               torch.ones(int(nonzero.sum())), atol=1e-5))

    def test_same_seed_gives_same_split(self):
        a = make_loader('cora', seed=123).load()['train_test_val']
        b = make_loader('cora', seed=123).load()['train_test_val']
        self.assertTrue(torch.equal(a['idx_train'], b['idx_train']))
        self.assertTrue(torch.equal(a['idx_test'], b['idx_test']))
        self.assertTrue(torch.equal(a['idx_val'], b['idx_val']))

    def test_unknown_dataset_raises(self):
        ld = make_loader('not_a_dataset')
        with self.assertRaises(ValueError):
            ld.load()


if __name__ == '__main__':
    unittest.main(verbosity=2)