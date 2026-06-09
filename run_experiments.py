import sys, os, time, warnings
warnings.filterwarnings('ignore', category=UserWarning)

sys.path.insert(0, os.path.dirname(__file__))

import nltk
nltk.download('treebank', quiet=True)
from nltk.corpus import treebank

from model import build_crf, fit, predict, gold_labels
from eval  import (token_accuracy, per_tag_report,
                   plot_learning_curve, plot_regularisation,
                   plot_feature_ablation, plot_confusion,
                   print_summary)

print('\nLoading Penn Treebank')
all_sents = [s for s in treebank.tagged_sents() if any(t != '-NONE-' for _, t in s)]
n = len(all_sents)
n_train = int(0.80 * n)
n_dev = int(0.10 * n)

train_sents = all_sents[:n_train]
dev_sents = all_sents[n_train: n_train + n_dev]
test_sents = all_sents[n_train + n_dev:]
train_vocab = set(w for s in train_sents for w, _ in s)

print(f'Sentences: {n} total')
print(f'Split: {len(train_sents)} train / {len(dev_sents)} dev / {len(test_sents)} test')

MAX_ITER = 80

print('\nExperiment 1: Feature-Set Ablation')
ablation_accs = {}
for fs in ['FS1', 'FS2', 'FS3', 'FS4']:
    t0  = time.time()
    crf = build_crf(c2=0.01, max_iterations=MAX_ITER, feature_set=fs)
    fit(crf, train_sents, feature_set=fs)
    y_pred = predict(crf, test_sents, feature_set=fs)
    y_true = gold_labels(test_sents)
    acc    = token_accuracy(y_true, y_pred, test_sents, train_vocab)
    ablation_accs[fs] = acc['overall']
    print(f'{fs}: overall={acc["overall"]*100:.2f}%  '
          f'iv={acc.get("in_vocab",0)*100:.2f}%  '
          f'oov={acc.get("oov",0)*100:.2f}%  '
          f'({time.time()-t0:.1f}s)')

plot_feature_ablation(list(ablation_accs.keys()), list(ablation_accs.values()))

print('\nExperiment 2: Training Set Size Curve')
fractions = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
overall_accs = []
oov_accs = []
n_train_counts = []

for frac in fractions:
    n_sub = max(1, int(len(train_sents) * frac))
    sub = train_sents[:n_sub]
    sub_vocab = set(w for s in sub for w, _ in s)
    
    crf = build_crf(c2=0.01, max_iterations=MAX_ITER, feature_set='FS4')
    fit(crf, sub, feature_set='FS4')
    
    y_pred = predict(crf, test_sents, feature_set='FS4')
    y_true = gold_labels(test_sents)
    acc = token_accuracy(y_true, y_pred, test_sents, sub_vocab)
    
    overall_accs.append(acc['overall'])
    oov_accs.append(acc.get('oov', 0))
    n_train_counts.append(n_sub)
    print(f'{frac:.0%} (n={n_sub}): overall={acc["overall"]*100:.2f}%'
          f'oov={acc.get("oov",0)*100:.2f}%')

plot_learning_curve(fractions, overall_accs, oov_accs, n_train_counts)

print('\nExperiment 3: L2 Regularisation Sweep (dev set)')
c2_values = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
dev_accs  = []

for c2 in c2_values:
    crf = build_crf(c2=c2, max_iterations=MAX_ITER, feature_set='FS4')
    fit(crf, train_sents, feature_set='FS4')
    y_pred = predict(crf, dev_sents, feature_set='FS4')
    y_true = gold_labels(dev_sents)
    acc = token_accuracy(y_true, y_pred, dev_sents, train_vocab)
    dev_accs.append(acc['overall'])
    print(f'c2={c2}: dev={acc["overall"]*100:.2f}%')

best_c2 = c2_values[dev_accs.index(max(dev_accs))]
print(f'Best c2: {best_c2}')
plot_regularisation(c2_values, dev_accs)

print(f'\nExperiment 4: Best Model (c2={best_c2}, FS4) — Test Evaluation')
crf_best = build_crf(c2=best_c2, max_iterations=MAX_ITER, feature_set='FS4')
t0 = time.time()
fit(crf_best, train_sents, feature_set='FS4')
train_time = time.time() - t0

t1 = time.time()
y_pred_test = predict(crf_best, test_sents, feature_set='FS4')
decode_time = time.time() - t1
y_true_test = gold_labels(test_sents)

acc_test = token_accuracy(y_true_test, y_pred_test, test_sents, train_vocab)
print_summary('Test set', acc_test)
print(f'Train time : {train_time:.1f}s')
print(f'Decode time: {decode_time:.3f}s  ({len(test_sents)} sentences)')

print('\nPer-tag classification report:')
print(per_tag_report(y_true_test, y_pred_test))

plot_confusion(y_true_test, y_pred_test, n_worst=12)

print('\nAll experiments complete. Figures saved to pos_crf/figures/')
