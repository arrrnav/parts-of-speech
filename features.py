"""
features.py
-----------
Feature extraction for CRF-based POS tagging.

Four nested feature sets of increasing richness:
  FS1 - word form + case flags          (baseline)
  FS2 - FS1 + suffixes/prefixes
  FS3 - FS2 + ±2 context window
  FS4 - FS3 + word shape + hyphen/digit flags  (full)
"""


def word_shape(word):
    """Collapse a word to its abstract shape: A→X, a→x, 0→9, else keep."""
    shape = []
    for ch in word:
        if ch.isupper():
            shape.append('X')
        elif ch.islower():
            shape.append('x')
        elif ch.isdigit():
            shape.append('9')
        else:
            shape.append(ch)
    # Compress runs: "Xxxxx" → "Xx", "999" → "9"
    compressed = []
    for ch in shape:
        if not compressed or compressed[-1] != ch:
            compressed.append(ch)
    return ''.join(compressed)


def base_features(word):
    """FS1: word form and simple case flags."""
    lw = word.lower()
    return {
        'word.lower':    lw,
        'word.isupper':  word.isupper(),
        'word.istitle':  word.istitle(),
        'word.isdigit':  word.isdigit(),
    }


def affix_features(word):
    """FS2 additions: character n-gram suffixes and prefixes."""
    lw = word.lower()
    feats = {}
    for n in range(1, 5):           # suffixes 1–4
        feats[f'word.suffix{n}'] = lw[-n:] if len(lw) >= n else lw
    for n in range(1, 4):           # prefixes 1–3
        feats[f'word.prefix{n}'] = lw[:n]  if len(lw) >= n else lw
    return feats


def shape_features(word):
    """FS4 additions: abstract word shape, hyphen, and digit flags."""
    return {
        'word.shape':    word_shape(word),
        'word.has_hyphen': '-' in word,
        'word.has_digit':  any(c.isdigit() for c in word),
    }


def token_features(sent, i, feature_set='FS4'):
    """
    Return a feature dict for token i in sentence `sent`
    (a list of (word, tag) tuples or just words).

    feature_set : 'FS1' | 'FS2' | 'FS3' | 'FS4'
    """
    word = sent[i][0] if isinstance(sent[i], tuple) else sent[i]

    feats = base_features(word)

    if feature_set in ('FS2', 'FS3', 'FS4'):
        feats.update(affix_features(word))

    if feature_set in ('FS4',):
        feats.update(shape_features(word))

    # BOS / EOS markers
    if i == 0:
        feats['BOS'] = True
    if i == len(sent) - 1:
        feats['EOS'] = True

    # FS3 / FS4: ±2 context window
    if feature_set in ('FS3', 'FS4'):
        for offset, prefix in [(-2, 'm2'), (-1, 'm1'), (1, 'p1'), (2, 'p2')]:
            j = i + offset
            if 0 <= j < len(sent):
                w = sent[j][0] if isinstance(sent[j], tuple) else sent[j]
                lw = w.lower()
                feats[f'{prefix}:word.lower'] = lw
                feats[f'{prefix}:word.istitle'] = w.istitle()
                feats[f'{prefix}:word.isupper'] = w.isupper()
                if feature_set == 'FS4':
                    feats[f'{prefix}:word.suffix2'] = lw[-2:] if len(lw) >= 2 else lw
                    feats[f'{prefix}:word.suffix3'] = lw[-3:] if len(lw) >= 3 else lw
            else:
                feats[f'{prefix}:OOB'] = True   # out-of-bounds

    return feats


def sent_to_features(sent, feature_set='FS4'):
    """Convert a tagged sentence to a list of feature dicts."""
    return [token_features(sent, i, feature_set) for i in range(len(sent))]


def sent_to_labels(sent):
    """Extract the gold tag sequence from a tagged sentence."""
    return [tag for _, tag in sent]


def sent_to_tokens(sent):
    """Extract the word sequence from a tagged sentence."""
    return [word for word, _ in sent]
