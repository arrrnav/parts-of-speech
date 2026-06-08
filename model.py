"""
model.py
--------
CRF model wrapper: train, predict, and serialize.

Uses sklearn-crfsuite (L-BFGS training, Viterbi decoding).
"""

import sklearn_crfsuite
from features import sent_to_features, sent_to_labels


def build_crf(c1=0.0, c2=0.01, max_iterations=150, feature_set='FS4'):
    """
    Instantiate a CRF model.

    Parameters
    ----------
    c1 : float   L1 regularisation coefficient
    c2 : float   L2 regularisation coefficient (default tuned on dev)
    max_iterations : int
    feature_set : str  Passed through to feature extraction at fit time.
    """
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=c1,
        c2=c2,
        max_iterations=max_iterations,
        all_possible_transitions=True,   # learn weights even for unseen tag bigrams
    )
    crf._feature_set = feature_set      # store for reference
    return crf


def fit(crf, train_sents, feature_set=None):
    """
    Fit the CRF on a list of tagged sentences.

    Parameters
    ----------
    crf : sklearn_crfsuite.CRF
    train_sents : list of list of (word, tag) tuples
    feature_set : str or None  (if None, uses crf._feature_set)
    """
    fs = feature_set or getattr(crf, '_feature_set', 'FS4')
    X_train = [sent_to_features(s, fs) for s in train_sents]
    y_train = [sent_to_labels(s) for s in train_sents]
    crf.fit(X_train, y_train)
    return crf


def predict(crf, sents, feature_set=None):
    """
    Return predicted tag sequences for a list of sentences.

    Parameters
    ----------
    crf : fitted sklearn_crfsuite.CRF
    sents : list of list of (word, tag) tuples  OR  list of list of str
    feature_set : str or None

    Returns
    -------
    y_pred : list of list of str
    """
    fs = feature_set or getattr(crf, '_feature_set', 'FS4')
    X = [sent_to_features(s, fs) for s in sents]
    return crf.predict(X)


def gold_labels(sents):
    """Extract gold tag sequences from tagged sentences."""
    return [sent_to_labels(s) for s in sents]
