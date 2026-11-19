'''
Build an embedding matrix for the joke vocabulary from pretrained GloVe vectors.

The GloVe file is read straight from its zip (no need to unzip 1.7 GB). Row i of the
matrix is the vector for itos[i]; words GloVe does not know get small random vectors,
and <PAD> gets zeros. The result is cached next to the zip, so later runs skip the read.

Optional fallback for words GloVe does not have: GloVe splits contractions and possessives
into pieces ("baker's" -> baker 's, "can't" -> ca n't) while our tokenizer keeps them whole,
so a missing word with an apostrophe gets the average of its GloVe pieces.
'''

import os
import io
import zipfile
import torch

DEFAULT_GLOVE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'pretrained_glove', 'glove.2024.wikigiga.100d.zip'))
GLOVE_DIM = 100


def glove_pieces(word):
    '''Split a contraction / possessive the way GloVe does: "can't" -> [ca, n't], "baker's" -> [baker, 's]'''
    if word.endswith("n't") and len(word) > 3:
        return [word[:-3], "n't"]
    if "'" in word.strip("'"):
        head, tail = word.split("'", 1)
        return [head, "'" + tail]
    return []


def load_glove_matrix(itos, pad_idx=0, glove_path=DEFAULT_GLOVE_PATH, fallback=False, seed=2):
    '''
    Returns (matrix, info):
        matrix  float tensor [len(itos), GLOVE_DIM]
        info    coverage numbers for the report: found, fallback, missing, total
    '''
    cache_path = os.path.splitext(glove_path)[0] + ('_jokes_fallback.pt' if fallback else '_jokes.pt')
    if os.path.isfile(cache_path):
        cached = torch.load(cache_path, weights_only=True)
        if cached['itos'] == list(itos):
            print('loaded cached GloVe matrix:', cache_path)
            return cached['matrix'], cached['info']

    print('reading GloVe vectors from', glove_path, '...')
    special = {i for i, w in enumerate(itos) if w.startswith('<') and w.endswith('>')}
    needed = {w for i, w in enumerate(itos) if i not in special}
    if fallback:
        needed |= {piece for w in list(needed) for piece in glove_pieces(w)}
    vectors = {}
    with zipfile.ZipFile(glove_path) as z:
        with io.TextIOWrapper(z.open(z.namelist()[0]), encoding='utf-8') as f:
            for line in f:
                word, rest = line.rstrip('\n').split(' ', 1)
                if word in needed:
                    vectors[word] = torch.tensor([float(v) for v in rest.split(' ')])

    # missing words start as small random vectors on the same scale as the GloVe ones
    known = torch.stack(list(vectors.values()))
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(len(itos), GLOVE_DIM, generator=generator) * known.std()
    matrix[pad_idx] = 0
    found = from_fallback = 0
    for i, word in enumerate(itos):
        if i in special:
            continue
        if word in vectors:
            matrix[i] = vectors[word]
            found += 1
        elif fallback:
            pieces = [vectors[p] for p in glove_pieces(word) if p in vectors]
            if pieces:
                matrix[i] = torch.stack(pieces).mean(0)
                from_fallback += 1
    total = len(itos) - len(special)
    info = {'found': found, 'fallback': from_fallback, 'missing': total - found - from_fallback, 'total': total}
    print(f"GloVe coverage: {found}/{total} words found, {from_fallback} from fallback, {info['missing']} random")

    torch.save({'itos': list(itos), 'matrix': matrix, 'info': info}, cache_path)
    return matrix, info
