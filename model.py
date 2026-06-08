import sklearn_crfsuite
from features import sent_to_features, sent_to_labels

def build_crf(c2=0.01, max_iterations=150, feature_set='FS4'):
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.0,
        c2=c2,
        max_iterations=max_iterations,
        all_possible_transitions=True,
    )
    crf._feature_set = feature_set
    return crf

def fit(crf, train_sents, feature_set=None):
    fs = feature_set or getattr(crf, '_feature_set', 'FS4')
    X_train = [sent_to_features(s, fs) for s in train_sents]
    y_train = [sent_to_labels(s) for s in train_sents]
    crf.fit(X_train, y_train)
    return crf

def predict(crf, sents, feature_set=None):
    fs = feature_set or getattr(crf, '_feature_set', 'FS4')
    X = [sent_to_features(s, fs) for s in sents]
    return crf.predict(X)

def gold_labels(sents):
    return [sent_to_labels(s) for s in sents]
