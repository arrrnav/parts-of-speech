"""
eval.py
-------
Evaluation utilities for the CRF POS tagger.

Functions
---------
token_accuracy        : overall / in-vocab / OOV accuracy
per_tag_report        : precision, recall, F1 per tag
plot_learning_curve   : accuracy vs training fraction
plot_regularisation   : dev accuracy vs c2
plot_feature_ablation : bar chart of feature-set accuracies
plot_confusion        : confusion matrix heat-map for worst-N tags
save_fig              : helper to save figures consistently
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ── Style ──────────────────────────────────────────────────────────────────
BLUE   = '#2E75B6'
LIGHT  = '#BDD7EE'
ORANGE = '#C55A11'
GREY   = '#595959'
sns.set_theme(style='whitegrid', font='Arial', font_scale=1.05)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_fig(name, dpi=150):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f'  → saved {path}')
    return path


# ── Core metrics ───────────────────────────────────────────────────────────

def flatten(y_lists):
    return [t for seq in y_lists for t in seq]


def token_accuracy(y_true_seqs, y_pred_seqs, sents=None, train_vocab=None):
    """
    Compute overall accuracy plus in-vocab / OOV split.

    Parameters
    ----------
    y_true_seqs, y_pred_seqs : list of list of str
    sents       : list of tagged sentences (used for OOV split)
    train_vocab : set of words seen during training

    Returns
    -------
    dict with keys: overall, in_vocab, oov
    """
    y_true = flatten(y_true_seqs)
    y_pred = flatten(y_pred_seqs)
    total   = len(y_true)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    result = {'overall': correct / total}

    if sents is not None and train_vocab is not None:
        words = [w for s in sents for w, _ in s]
        iv_c = iv_t = oov_c = oov_t = 0
        for w, t, p in zip(words, y_true, y_pred):
            if w in train_vocab:
                iv_t  += 1;  iv_c  += (t == p)
            else:
                oov_t += 1;  oov_c += (t == p)
        result['in_vocab'] = iv_c  / iv_t  if iv_t  else None
        result['oov']      = oov_c / oov_t if oov_t else None
        result['oov_count']= oov_t
    return result


def per_tag_report(y_true_seqs, y_pred_seqs, output_dict=False):
    """Thin wrapper around sklearn classification_report."""
    y_true = flatten(y_true_seqs)
    y_pred = flatten(y_pred_seqs)
    return classification_report(y_true, y_pred,
                                 output_dict=output_dict,
                                 zero_division=0)


# ── Plots ──────────────────────────────────────────────────────────────────

def plot_learning_curve(fractions, overall_accs, oov_accs, n_train_counts):
    """Line plot of accuracy vs training set fraction."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(fractions, [a * 100 for a in overall_accs],
            'o-', color=BLUE,   lw=2, ms=6, label='Overall accuracy')
    ax.plot(fractions, [a * 100 for a in oov_accs],
            's--', color=ORANGE, lw=2, ms=6, label='OOV accuracy')

    ax.set_xlabel('Training fraction', fontsize=11)
    ax.set_ylabel('Token accuracy (%)', fontsize=11)
    ax.set_title('CRF Accuracy vs. Training Set Size\n(Penn Treebank, FS4)', fontsize=12)
    ax.set_xticks(fractions)
    ax.set_xticklabels([f'{int(f*100)}%\n(n={n})' for f, n in zip(fractions, n_train_counts)],
                       fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.set_ylim(60, 100)
    ax.legend(frameon=True)
    ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    return save_fig('learning_curve.png')


def plot_regularisation(c2_values, dev_accs):
    """Line plot of dev accuracy vs c2."""
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(range(len(c2_values)), [a * 100 for a in dev_accs],
            'o-', color=BLUE, lw=2, ms=7)
    ax.set_xticks(range(len(c2_values)))
    ax.set_xticklabels([str(c) for c in c2_values], fontsize=9)
    ax.set_xlabel('L2 regularisation coefficient (c2)', fontsize=11)
    ax.set_ylabel('Dev accuracy (%)', fontsize=11)
    ax.set_title('Dev Accuracy vs. L2 Regularisation Strength', fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax.grid(axis='y', alpha=0.4)
    # annotate best
    best_i = int(np.argmax(dev_accs))
    ax.annotate(f'best: {dev_accs[best_i]*100:.2f}%',
                xy=(best_i, dev_accs[best_i]*100),
                xytext=(best_i + 0.3, dev_accs[best_i]*100 - 0.5),
                arrowprops=dict(arrowstyle='->', color=ORANGE),
                color=ORANGE, fontsize=9)
    plt.tight_layout()
    return save_fig('regularisation_curve.png')


def plot_feature_ablation(feature_sets, accuracies):
    """Horizontal bar chart of feature-set accuracies."""
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    colors = [LIGHT, LIGHT, LIGHT, BLUE]   # highlight full model
    bars = ax.barh(feature_sets, [a * 100 for a in accuracies],
                   color=colors, edgecolor='white', height=0.55)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{acc*100:.1f}%', va='center', fontsize=10)
    ax.set_xlabel('Token accuracy (%)', fontsize=11)
    ax.set_title('Feature Set Ablation (Penn Treebank test set)', fontsize=12)
    ax.set_xlim(80, 100)
    ax.grid(axis='x', alpha=0.4)
    plt.tight_layout()
    return save_fig('feature_ablation.png')


def plot_confusion(y_true_seqs, y_pred_seqs, n_worst=12):
    """
    Confusion matrix restricted to the N tags with lowest recall,
    so the plot stays readable.
    """
    y_true = flatten(y_true_seqs)
    y_pred = flatten(y_pred_seqs)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    # pick N tags with lowest recall (excluding micro/macro avg keys)
    tag_recalls = {k: v['recall'] for k, v in report.items()
                   if isinstance(v, dict) and 'recall' in v}
    worst_tags = sorted(tag_recalls, key=tag_recalls.get)[:n_worst]

    cm = confusion_matrix(y_true, y_pred, labels=worst_tags, normalize='true')

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=worst_tags, yticklabels=worst_tags,
                linewidths=0.3, linecolor='#dddddd',
                vmin=0, vmax=1, ax=ax,
                annot_kws={'size': 7})
    ax.set_xlabel('Predicted tag', fontsize=11)
    ax.set_ylabel('True tag', fontsize=11)
    ax.set_title(f'Confusion Matrix — {n_worst} Lowest-Recall Tags', fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0,  labelsize=8)
    plt.tight_layout()
    return save_fig('confusion_matrix.png')


def print_summary(label, acc_dict):
    ov  = acc_dict.get('overall', None)
    iv  = acc_dict.get('in_vocab', None)
    oov = acc_dict.get('oov', None)
    n   = acc_dict.get('oov_count', None)
    print(f'\n  {label}')
    if ov  is not None: print(f'    Overall  : {ov*100:.2f}%')
    if iv  is not None: print(f'    In-vocab : {iv*100:.2f}%')
    if oov is not None: print(f'    OOV      : {oov*100:.2f}%  ({n} tokens)')
