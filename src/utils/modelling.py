# @author semese

from src.utils.data_loader import MobilityDataModule
from src.models.pred_model import WHODASPredictor
from src.utils.optimisation import ClassifierTrainer, RegressorTrainer
from src.utils.constants import demogr_cols
import torch
from sklearn.model_selection import StratifiedGroupKFold

import pickle
import numpy as np
import pandas as pd

pd.options.display.float_format = '{:.3f}'.format


# -----------------------
# Transfer learning utils
# -----------------------


def set_parameter_requires_grad(model):
    '''
    This helper function sets the .requires_grad attribute of the parameters in the model to False when we are feature extracting. 

    :param model: a defined model
    :type model: nn.Model
    '''
    for param in model.parameters():
        param.requires_grad = False


def get_params_to_update(model, feature_extract, verbose=False):
    '''
    Gather the parameters to be optimized/updated in this run. If we are
    finetuning we will be updating all parameters. However, if we are
    doing feature extract method, we will only update the parameters
    that we have just initialized, i.e. the parameters with requires_grad
    is True.

    :param model: a defined model
    :type model: nn.Model
    :param feature_extract: flag for feature extracting
    :type feature_extract: boolean
    '''
    params_to_update = model.parameters()
    if verbose:
        print('Params to learn:')
    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                if verbose:
                    print('\t', name)
    elif verbose:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print('\t', name)

    return params_to_update


def init_model_and_optim(pre_trained_path, hparams_tl, output_dim, task, feature_extract=False):
    '''Initialise model with or without pre-trained weights and optimiser according to the task.

    :param pre_trained_path: path to the pre-trained model for transfer learning 
    :type pre_trained_path: str
    :param hparams_tl: model hyperparameters
    :type hparams_tl: dict
    :param output_dim: dimension of the output
    :type output_dim: int
    :param task: whether it's a classification or regression model, defaults to 'classification'
    :type task: str, optional
    :param feature_extract: indicator for feature extraction type transfer learning, defaults to False
    :type feature_extract: bool, optional
    :return: the initialised model and optimiser
    :rtype: tuple
    '''
    # initialise the prediction model
    model = WHODASPredictor(hparams_tl)

    if pre_trained_path is not None:
        # load pre-trained weights
        pre_trained_kvpairs = torch.load(pre_trained_path)['state_dict']

        # set model weights with the pretrained ones
        model_tl_kvpairs = model.state_dict()
        for key in pre_trained_kvpairs.keys():
            model_tl_kvpairs[key] = pre_trained_kvpairs[key]

        model.load_state_dict(model_tl_kvpairs)

        # set .requires_grad to False if feature extraction is used
        if feature_extract:
            set_parameter_requires_grad(model)

    # adjust the last layer
    model.fc = torch.nn.Linear(
        model.fc.in_features + len(demogr_cols), output_dim)

    # initialise the optimiser
    if task == 'classification':
        loss_fn = torch.nn.BCEWithLogitsLoss(
        ) if output_dim == 1 else torch.nn.CrossEntropyLoss()

        optim = ClassifierTrainer(
            model,
            loss_fn,
            torch.optim.Adam(get_params_to_update(
                model, feature_extract), lr=0.001),
            clip=100,
            n_classes=2 if output_dim == 1 else output_dim
        )
    else:
        optim = RegressorTrainer(
            model,
            torch.nn.MSELoss(reduction='mean'),
            torch.optim.Adam(get_params_to_update(
                model, feature_extract), lr=0.001),
            clip=100
        )

    return model, optim

# ----------------------
# Cross-validation utils
# ----------------------


def cross_validate(pre_trained_path, chk_path, hparams_tl, hparams_d, n_splits=5, n_epochs=50,
                   task='classification', feature_extract=False, strat_col=None,
                   ckp_to_load='min_val_model.pt', verbose=True):
    '''Run k-fold cross validation in order to assess the model performance in function 
    of the split in the training data.

    :param pre_trained_path: path to the pre-trained temporal model
    :type pre_trained_path: str
    :param chk_path: path to save the checkpoints
    :type chk_path: str
    :param hparams_tl: hyperparameters of the model
    :type hparams_tl: dict
    :param hparams_d: hyperparameters of the dataloader
    :type hparams_d: dict
    :param n_splits: number of CV splits to do, defaults to 5
    :type n_splits: int, optional
    :param n_epochs: number of epochs to train the model, defaults to 50
    :type n_epochs: int, optional
    :param task: whether it's a classification or regression model, defaults to 'classification'
    :type task: str, optional
    :param feature_extract: indicator for feature extraction type transfer learning, defaults to False
    :type feature_extract: bool, optional
    :param strat_col: name of column to use for stratification in case of regression, defaults to None
    :type strat_col: str, optional
    :param ckp_to_load: name of checkpoint to load to evaluate, defaults to 'min_val_model.pt'
    :type ckp_to_load: str, optional
    :param verbose: indicator whether to print progress information, defaults to True
    :type verbose: bool, optional
    :return: dictionary of cross-validation results
    :rtype: dict

    '''
    # necessary X,y,groups initialisations for the splitting
    df = pd.read_csv(hparams_d['data_path']).dropna(
        subset=[hparams_d['y_col']], how='any')
    X = df[hparams_d['x_d_cols']].values
    y = df[hparams_d['y_col']].values.ravel(
    ) if strat_col is None else df[strat_col].values.ravel()
    groups = df.user.values.ravel()

    if task == 'classification':
        output_dim = 1 if df[hparams_d['y_col']].nunique(
        ) == 2 else df[hparams_d['y_col']].nunique()
    else:
        output_dim = 1

    # dictionary to store cv results
    cv_results = {}

    # define cross-validator
    cv = StratifiedGroupKFold(n_splits=n_splits)
    for cv_i, (train_idxs, test_idxs) in enumerate(cv.split(X, y, groups)):

        if verbose:
            print('Fold {}/{}'.format(cv_i, n_splits))
            print('...........')
        else:
            hparams_d['verbose'] = False

        # get patient ids for train-valid-test sets
        train_idxs, val_idxs = train_idxs[:int(
            0.9*len(train_idxs))], train_idxs[int(0.9*len(train_idxs)):]
        hparams_d['train_test_split'] = (
            groups[train_idxs], groups[val_idxs], groups[test_idxs])

        # create dataloaders
        data_module_tl = MobilityDataModule(hparams_d)
        train_loader = data_module_tl.train_dataloader()
        val_loader = data_module_tl.val_dataloader()
        test_loader = data_module_tl.test_dataloader()

        # initialise model and optimiser
        _, optim = init_model_and_optim(
            pre_trained_path, hparams_tl, output_dim, task, feature_extract)

        # train the model
        checkpoint_path = chk_path + 'fold_' + str(cv_i) + '_'
        optim.train(
            train_loader, val_loader, n_epochs=n_epochs,
            checkpoint_path=checkpoint_path
        )

        # store losses
        losses = (optim.train_losses, optim.val_losses)

        # load best model
        optim.load_ckp(checkpoint_path + ckp_to_load)

        # evaluate the model
        y_scores, y_true = optim.evaluate(test_loader)

        # store cv step results
        cv_results[cv_i] = {
            'pids': hparams_d['train_test_split'],
            'losses': losses,
            'eval': (y_scores, y_true)
        }

        # save results
        pickle.dump(cv_results, open(chk_path + 'cv_results', 'wb'))

    return cv_results


def get_best_ckp_idx(cv_results, n_splits, n_epochs):
    '''Compute the minimum average loss over the epochs.

    :param cv_results: CV outputs
    :type cv_results: dict
    :param n_splits: number of CV splits
    :type n_splits: int
    :param n_epochs: number of epochs that were performed during training
    :type n_epochs: int
    '''
    mean_losses = np.zeros(n_epochs)

    for i in range(n_splits):
        mean_losses += np.asarray(cv_results[i]['losses'][1][:-1])

    return np.argmin(mean_losses / n_splits)


def get_cv_missingness(hparams_d, n_splits=5, strat_col=None):
    ''' Compute percentage of missingness in the features based on the CV splits.

    :param hparams_d: hyperparameters to create the dataloaders
    :type hparams_d: dict
    :param n_splits: number of CV splits, defaults to 5
    :type n_splits: int
    :param strat_col: name of column to use for stratification in case of regression, defaults to None
    :type strat_col: str
    '''
    # necessary X,y,groups initialisations for the splitting
    df = pd.read_csv(hparams_d['data_path']).dropna(
        subset=[hparams_d['y_col']], how='any')
    X = df[hparams_d['x_d_cols']].values
    y = df[hparams_d['y_col']].values.ravel(
    ) if strat_col is None else df[strat_col].values.ravel()
    groups = df.user.values.ravel()

    # dictionary to store missingness results
    miss_results = []

    # define cross-validator
    cv = StratifiedGroupKFold(n_splits=n_splits)
    for cv_i, (train_idxs, test_idxs) in enumerate(cv.split(X, y, groups)):

        # get patient ids for train-valid-test sets
        train_idxs, val_idxs = train_idxs[:int(
            0.9*len(train_idxs))], train_idxs[int(0.9*len(train_idxs)):]
        hparams_d['train_test_split'] = (
            groups[train_idxs], groups[val_idxs], groups[test_idxs])

        # create dataloaders
        data_module_tl = MobilityDataModule(hparams_d)

        # get datasets
        n_feat = len(hparams_d['x_t_cols'])
        X_train = np.concatenate(
            data_module_tl.X_train[0], axis=0).reshape(-1, n_feat)
        X_val = np.concatenate(
            data_module_tl.X_val[0], axis=0).reshape(-1, n_feat)
        X_test = np.concatenate(
            data_module_tl.X_test[0], axis=0).reshape(-1, n_feat)

        # compute missingness per feature
        df_split = pd.DataFrame({
            'Split': [cv_i] * 3 * n_feat,
            'Dataset': [ds for ds in ['Training', 'Validation', 'Test'] for _ in range(n_feat)],
            'Feature': [ft for _ in range(3) for ft in hparams_d['x_t_cols'].keys()],
            'Missing [%]': np.asarray([np.sum(np.isnan(x), axis=0) / x.size for x in [X_train, X_val, X_test]]).reshape(-1)
        })

        # store results
        miss_results.append(df_split)

    return pd.concat(miss_results, axis=0, ignore_index=True)
