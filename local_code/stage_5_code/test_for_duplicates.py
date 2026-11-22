'''
Checks the raw Cora, Citeseer and PubMed files for duplicate and bad lines.
Run:  python test_for_duplicates.py
'''
import os
import sys
import unittest
from collections import Counter


def _find_data_root():
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        candidate = os.path.join(d, 'data', 'stage_5_data')
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(d)
        if parent == d:
            raise FileNotFoundError('could not find data/stage_5_data above this file')
        d = parent


DATA_ROOT = _find_data_root()
DATASETS = ['cora', 'citeseer', 'pubmed']


def read_node_ids(folder):
    with open(os.path.join(folder, 'node'), encoding='utf-8') as f:
        return [int(line.split()[0]) for line in f if line.strip()]


def read_links(folder):
    with open(os.path.join(folder, 'link'), encoding='utf-8') as f:
        return [tuple(int(x) for x in line.split()) for line in f if line.strip()]


class TestForDuplicates(unittest.TestCase):

    def test_no_duplicate_node_ids(self):
        for name in DATASETS:
            with self.subTest(dataset=name):
                ids = read_node_ids(os.path.join(DATA_ROOT, name))
                dupes = [i for i, c in Counter(ids).items() if c > 1]
                print('\n[{}] node ids: {} lines, {} duplicated ids'.format(name, len(ids), len(dupes)))
                self.assertEqual(dupes, [], msg='duplicated node ids, e.g. {}'.format(dupes[:5]))

    def test_no_duplicate_directed_links(self):
        for name in DATASETS:
            with self.subTest(dataset=name):
                links = read_links(os.path.join(DATA_ROOT, name))
                counts = Counter(links)
                dupes = {k: v for k, v in counts.items() if v > 1}
                extra = sum(v - 1 for v in dupes.values())
                print('\n[{}] links: {} lines, {} repeated pairs, {} extra copies'.format(
                    name, len(links), len(dupes), extra))
                self.assertEqual(len(dupes), 0,
                                 msg='{} repeated directed links, e.g. {}'.format(
                                     len(dupes), list(dupes.items())[:5]))

    def test_info_reciprocal_and_self_links(self):
        # Informational only: these are not failures.
        for name in DATASETS:
            with self.subTest(dataset=name):
                links = read_links(os.path.join(DATA_ROOT, name))
                unique = set(links)
                reciprocal = sum(1 for a, b in unique if a != b and (b, a) in unique) // 2
                self_links = sum(1 for a, b in unique if a == b)
                unordered_dupes = len(links) - len({tuple(sorted(p)) for p in links})
                print('\n[{}] reciprocal pairs (A B and B A): {}, self-links: {}, '
                      'lines repeated ignoring direction: {}'.format(
                          name, reciprocal, self_links, unordered_dupes))

    def test_all_link_endpoints_exist_in_node_file(self):
        for name in DATASETS:
            with self.subTest(dataset=name):
                node_ids = set(read_node_ids(os.path.join(DATA_ROOT, name)))
                missing = {x for link in read_links(os.path.join(DATA_ROOT, name)) for x in link
                           if x not in node_ids}
                print('\n[{}] link endpoints missing from node file: {}'.format(name, len(missing)))
                self.assertEqual(missing, set(), msg='missing endpoints, e.g. {}'.format(list(missing)[:5]))


if __name__ == '__main__':
    unittest.main(verbosity=2)