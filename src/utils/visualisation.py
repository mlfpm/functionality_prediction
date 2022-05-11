# @author semese

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, roc_curve, auc
from scikitplot.metrics import plot_confusion_matrix
from sklearn.preprocessing import label_binarize

sns.set(font_scale=1.25)
sns.set_palette('muted')
sns.set_style('ticks')


# -----
# Cross-validation
# -----

def plot_roc_and_cm(y_true, y_scores, n_classes=2, figsize=(14, 7)):
    '''Create plot of the ROC and confusion matrix given the true class labels and prediction
    scres. 
    
    :param y_true: target outcomes
    :type y_true: array_like
    :param y_pred: predicted outcomes
    :type y_pred: array_like
    :param n_classes: number of classes, defaults to 2 (binary classification)
    :type n_classes: int
    :param figsize: figure size, defaults to (14, 7)
    :type figsize: tuple, optional
    '''
    if n_classes == 2:
        false_positive_rate, recall, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(false_positive_rate, recall)
        y_pred = np.rint(y_scores)
    else:
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        y_pred = np.argmax(y_scores, axis=1)
        # Compute micro-average ROC curve and ROC area
        false_positive_rate, recall, _ = roc_curve(
            y_true_bin.ravel(), y_scores.ravel())
        roc_auc = auc(false_positive_rate, recall)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.plot(false_positive_rate, recall, label='AUC = %0.3f' %
             roc_auc if n_classes == 2 else 'Micro-avg AUC = %0.3f' % roc_auc)
    legend = ax1.legend(loc='lower right', shadow=True, frameon=True)

    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('whitesmoke')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_ylabel('Sensitivity')
    ax1.set_xlabel('1-Specificity')

    plot_confusion_matrix(
        y_true, y_pred, labels=np.arange(n_classes), normalize=False,
        hide_zeros=False, hide_counts=False, ax=ax2,
        cmap='Blues', text_fontsize='small'
    )


def plot_cv_cm(cv_results, labels, figsize=(20, 5)):
    '''Plot the confusion matrix of the test set for each split
    of the CV:

    :param cv_results: CV outputs
    :type cv_results: dict
    :param labels: list of labels 
    :type labels: list
    :param figsize: [description], defaults to (20, 5)
    :type figsize: tuple, optional
    '''
    n_splits = len(cv_results)
    fig, axs = plt.subplots(1, n_splits, figsize=figsize)
    for i, ax in enumerate(axs):
        # compute the CM
        if len(labels) > 2:
            y_true, y_pred = cv_results[i]['eval'][1], np.argmax(
                cv_results[i]['eval'][0], axis=1)
        else:
            y_true, y_pred = cv_results[i]['eval'][1], np.rint(
                cv_results[i]['eval'][0])
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')

        # plot the CM
        res = sns.heatmap(
            cm, vmin=0, annot=True, cmap='Blues', cbar=False, ax=ax
        )
        ax.set(title='Split ' + str(i+1))

        # make frame visible
        for _, spine in res.spines.items():
            spine.set_visible(True)

    fig.supxlabel('Predicted label')
    fig.supylabel('True label')
    fig.tight_layout()


def plot_cv_labels_per_patient(cv_results):
    '''Visualise the number of labels per patient in each fold.

    :param cv_results: CV outputs
    :type cv_results: dict
    '''
    fig, axs = plt.subplots(1, len(cv_results), figsize=(13, 4))
    for i, ax in enumerate(axs):
        df_dict = {'No.entries': [], 'Dataset': []}
        for j, ds in zip(range(3), ['Train', 'Validation', 'Test']):
            unique, counts = np.unique(
                cv_results[i]['pids'][j], return_counts=True)
            df_dict['No.entries'].extend(list(counts))
            df_dict['Dataset'].extend([ds]*len(counts))

        sns.countplot(data=pd.DataFrame(df_dict),
                      y='No.entries', hue='Dataset', ax=ax)

        if i < 4:
            ax.get_legend().remove()

        ax.set(xlabel='', ylabel='')

    fig.supxlabel('Count')
    fig.supylabel('No. entries per patient')
    fig.tight_layout()


def plot_cv_diff_labels_per_patient(cv_results, df, y_col):
    '''Visualise the number of label changes per patient in each fold.

    :param cv_results: CV outputs
    :type cv_results: dict
    :param df: dataframe of patient ids and scores
    :type df: pd.DataFrame
    :param y_col: output column of interest
    :type y_col: list
    '''
    fig, axs = plt.subplots(1, len(cv_results), figsize=(13, 4))
    for i, ax in enumerate(axs):
        df_dict = {'No.entries': [], 'Dataset': []}
        for j, ds in zip(range(3), ['Train', 'Validation', 'Test']):
            change = []
            unique, unique_counts = np.unique(
                cv_results[i]['pids'][j], return_counts=True)
            for pid, pid_cnt in zip(unique, unique_counts):
                if pid_cnt == 1:
                    change.append(1)
                else:
                    change.append(df[df.user == pid][y_col].nunique())

            df_dict['No.entries'].extend(list(change))
            df_dict['Dataset'].extend([ds]*len(change))

        sns.countplot(data=pd.DataFrame(df_dict),
                      y='No.entries', hue='Dataset', ax=ax)

        if i < 4:
            ax.get_legend().remove()

        ax.set(xlabel='', ylabel='')

    fig.supxlabel('Count')
    fig.supylabel('No. different labels per patient')
    fig.tight_layout()


def plot_cv_losses_and_roc(cv_results, n_classes=2, figsize=(14, 7)):
    '''Plot the evolution of losses and the achieved ROC for the CV. 
    
    :param cv_results: CV outputs
    :type cv_results: dict
    :param n_classes: number of classes, defaults to 2 (binary classification)
    :type n_classes: int
    :param figsize: figure size, defaults to (14, 7)
    :type figsize: tuple, optional
    '''
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # plot training and validation losses
    plot_cv_losses(cv_results, ax=ax1)

    # plot ROC
    plot_cv_roc(cv_results, n_classes=n_classes, ax=ax2)


def plot_cv_losses(cv_results, figsize=(14, 7), ax=None):
    '''Plot the evolution of losses for the CV. 
    
    :param cv_results: CV outputs
    :type cv_results: dict
    :param figsize: figure size, defaults to (14, 7)
    :type figsize: tuple, optional
    :param ax: figure axes, in case of subplots, defaults to None
    :type ax: matplotlib.axes.Axes
    '''
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    # create dataframes for plotting
    loss_df = {'Epoch': [], 'Losses': [], 'Loss': []}
    for _, cv_r in cv_results.items():
        # store losses
        n_epochs = len(cv_r['losses'][0])
        loss_df['Epoch'].extend(list(range(n_epochs))*2)
        loss_df['Losses'].extend(list(cv_r['losses'][0]))
        loss_df['Losses'].extend(list(cv_r['losses'][1]))
        loss_df['Loss'].extend(['Training']*n_epochs + ['Validation']*n_epochs)

    # plot training and validation losses
    sns.lineplot(data=pd.DataFrame(loss_df), x='Epoch',
                 y='Losses', hue='Loss', ax=ax)


def plot_cv_roc(cv_results, n_classes=2, figsize=(14, 7), ax=None):
    '''Plot the ROC curves from the cross-validation results.
    
    :param cv_results: CV outputs
    :type cv_results: dict
    :param n_classes: number of classes, defaults to 2 (binary classification)
    :type n_classes: int
    :param figsize: figure size, defaults to (14, 7)
    :type figsize: tuple, optional
    :param ax: figure axes, in case of subplots, defaults to None
    :type ax: matplotlib.axes.Axes
    '''
    def compute_fpr_tpr_auc(y_scores, y_true):
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
        else:
            y_test = label_binarize(y_true, classes=np.arange(n_classes))
            # Compute micro-average ROC curve and ROC area
            fpr, tpr, _ = roc_curve(y_test.ravel(), y_scores.ravel())
        roc_auc = auc(fpr, tpr)
        return list(fpr), list(tpr), roc_auc

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    roc_df = {'Sensitivity': [], '1-Specificity': [], 'Split': []}
    roc_auc = []
    for cv_i, cv_r in cv_results.items():
        # compute ROC
        y_scores, y_true = cv_r['eval']
        fpr, tpr, auc_i = compute_fpr_tpr_auc(y_scores, y_true)
        roc_df['1-Specificity'].extend(fpr)
        roc_df['Sensitivity'].extend(tpr)
        roc_df['Split'].extend(
            [str(cv_i)+'- AUROC = ' + '%.3f' % auc_i]*len(tpr))

        roc_auc.append(auc_i)

    sns.lineplot(data=pd.DataFrame(roc_df),
                 x='1-Specificity', y='Sensitivity', hue='Split', ax=ax)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
            label='Chance', alpha=.8)
    ax.set_title('AUC = %.3f (SD %.3f)' % (
        np.mean(np.asarray(roc_auc)), np.std(np.asarray(roc_auc))))

# -----
# Predictions
# -----


def plot_autoregression_pred(y_pred, y_true, labels, figsize=(10, 5)):
    '''Create plot of the predictions from the regression problem.

    :param y_pred: predicted outcomes
    :type y_pred: array_like
    :param y_true: target outcomes
    :type y_true: array_like
    :param labels: list of variable labels
    :type labels: list
    :param figsize: figure size, defaults to (10, 5)
    :type figsize: tuple, optional
    '''
    fig, axs = plt.subplots(1, len(labels), figsize=figsize)

    for i, ax in enumerate(axs):
        ax.plot(y_pred[:100, i], c='darkorange', label='Predicted')
        ax.plot(y_true[:100, i], c='grey', label='True')
        ax.set_ylabel(labels[i])

        if i == len(labels)-1:
            ax.legend()

    fig.supxlabel('Sequence index')
    fig.tight_layout()


def plot_attentions(daily_atts, monthly_atts):
    '''Visualise the attention weights assigned to the days and months
    as 2D heatmaps.

    :param daily_atts: attention weights for the daily sequences
    :type daily_atts: array_like
    :param monthly_atts: attention weights for the monthly sequences
    :type monthly_atts: array_like
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    sns.heatmap(data=np.mean(daily_atts, axis=1),
                cmap='Blues', cbar=True, ax=ax1)
    ax1.set(xlabel='', ylabel='')

    sns.heatmap(data=monthly_atts.reshape(-1, 30),
                cmap='Blues', cbar=True, ax=ax2)
    ax2.set(xlabel='', ylabel='')

    fig.supxlabel('Timesteps')
    fig.supylabel('Attention weights')
    fig.tight_layout()
