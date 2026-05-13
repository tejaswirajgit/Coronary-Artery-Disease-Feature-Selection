"""
Vendored plotting helpers for the CAD feature-selection notebook.

Re-creates the Kaggle sibling kernel `visualize_py` so the notebook can run
locally. Confusion-matrix metric layout matches the notebook convention,
which calls confusion_matrix(y_pred, y_true) — so cm[0,1] is FN and
cm[1,0] is FP (transposed from sklearn's default y_true-first ordering).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc


def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    cm = np.asarray(cm, dtype=float)
    if normalize:
        cm = cm / cm.sum()
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    TN, FN = cm[0, 0], cm[0, 1]
    FP, TP = cm[1, 0], cm[1, 1]

    def safe(num, den):
        return num / den if den else float('nan')

    accuracy    = TN + TP if normalize else safe(TN + TP, cm.sum())
    precision   = safe(TP, TP + FP)
    sensitivity = safe(TP, TP + FN)
    specificity = safe(TN, TN + FP)
    npv         = safe(TN, TN + FN)
    fpr         = safe(FP, FP + TN)
    fnr         = safe(FN, FN + TP)
    fdr         = safe(FP, FP + TP)

    print()
    print('accuracy:\t\t\t%0.3f  ' % accuracy)
    print('precision:\t\t\t%0.3f ' % precision)
    print('sensitivity:\t\t\t%0.3f' % sensitivity)
    print()
    print('specificity:\t\t\t%0.3f ' % specificity)
    print('negative predictive value:\t%0.3f' % npv)
    print()
    print('false positive rate:\t\t%0.3f  ' % fpr)
    print('false negative rate:\t\t%0.3f ' % fnr)
    print('false discovery rate:\t\t%0.3f ' % fdr)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title, ylabel='Predicted', xlabel='True')
    thresh = cm.max() / 2.
    fmt = '.2f' if normalize else 'd'
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    plt.show()


def plot_roc_curve(fpr, tpr):
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label='ROC curve (AUC = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], lw=1, linestyle='--', color='grey')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


def _print_and_bar(scores, features, title):
    scores = np.asarray(scores, dtype=float)
    features = np.asarray(features)
    order = np.argsort(-np.abs(scores))
    for idx in order:
        print('Feature: %20s\tScore:\t%0.5f' % (features[idx], scores[idx]))

    fig, ax = plt.subplots(figsize=(8, 6))
    sorted_features = features[order]
    sns.barplot(x=np.abs(scores)[order], y=sorted_features, ax=ax,
                hue=sorted_features, palette='viridis', legend=False)
    ax.set_xlabel('Importance')
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def plot_feature_importance(fit, features):
    """For sklearn SelectKBest fits — uses .scores_."""
    _print_and_bar(fit.scores_, features, 'Univariate feature scores')


def plot_feature_importance_log(fit, features):
    """For statsmodels Logit — accepts a fitted Results or an unfit model."""
    if not hasattr(fit, 'params'):
        fit = fit.fit()
    print(fit.summary2())
    print()
    coefs = np.asarray(fit.params, dtype=float)
    stderr = np.asarray(fit.bse, dtype=float)
    z = np.where(stderr > 0, np.abs(coefs / stderr), 0.0)
    _print_and_bar(z, features, 'Logistic regression |z| scores')


def plot_feature_importance_dec(fit, features):
    """For tree-based feature_importances_ arrays."""
    _print_and_bar(fit, features, 'Tree feature importances')


def plotAge(df, axes, single_plot=False):
    ax = axes[0, 0] if hasattr(axes, 'shape') and axes.ndim == 2 else axes
    for label, sub in df.groupby('ca_disease'):
        sns.kdeplot(sub['age'], ax=ax, fill=True, alpha=0.4,
                    label='CA disease' if label else 'No CA disease')
    ax.set_xlabel('age')
    ax.set_title('Age density by coronary artery disease')
    ax.legend()


def plotVar(isCategorical, categorical, continuous, df, axes):
    """
    Continuous mode (isCategorical=False): per continuous variable (except
    the target at the end of `continuous`), render a histogram (col 0) and
    a violin plot split by ca_disease (col 1).
    """
    if isCategorical:
        vars_ = [name for name, _ in categorical if name != 'ca_disease']
        for i, name in enumerate(vars_):
            r, c = divmod(i, 2)
            ax = axes[r, c] if axes.ndim == 2 else axes[i]
            sns.countplot(data=df, x=name, hue='ca_disease', ax=ax)
            ax.set_title(name)
    else:
        vars_ = [name for name, _ in continuous if name != 'ca_disease']
        for i, name in enumerate(vars_):
            ax_hist = axes[i, 0]
            ax_viol = axes[i, 1]
            for label, sub in df.groupby('ca_disease'):
                sns.histplot(sub[name], ax=ax_hist, kde=True, alpha=0.4,
                             label='CA disease' if label else 'No CA disease')
            ax_hist.set_title(f'{name} — distribution')
            ax_hist.legend()
            sns.violinplot(data=df, x='ca_disease', y=name, ax=ax_viol)
            ax_viol.set_title(f'{name} — by CA disease')
    plt.tight_layout()


def plotContinuous(*args, **kwargs):
    """Stub — imported by the notebook but not invoked."""
    return None


def plotCategorical(*args, **kwargs):
    """Stub — imported by the notebook but not invoked."""
    return None
