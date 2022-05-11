# @author semese

import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device(
    'cpu') if not torch.cuda.is_available() else torch.device('cuda:0')


class Trainer:
    ''' Wrapper class for model training and evaluation. 
    '''

    def __init__(self, model, loss_fn, optimiser, clip=0.5):
        ''' Class initialiser.

        :param model: pytorch model instance
        :type model: torch.nn.Model
        :param loss_fn: pytorch loss function - it can be either for regression 
            or classification
        :type loss_fn: torch.nn loss function
        :param optimiser: pytorch optimiser 
        :type optimiser: torch.optim algorithm
        :param clip: gradient clipping, defaults to 0.5
        :type clip: float, optional
        :param task: string to indicate whether it's a classification or 
            regression task, defaults to 'classification'
        :type task: str, optional
        '''
        self.model = model
        self.loss_fn = loss_fn
        self.optimiser = optimiser
        self.scheduler = ReduceLROnPlateau(
            optimiser, 'min', patience=10, verbose=True)
        self.clip = clip
        self.train_losses = []
        self.val_losses = []
        self.min_val_loss = np.inf

    def train_step(self, x, y):
        ''' One step of the training method.

        :param x: input feature tensor
        :type x: torch.Tensor
        :param y: expected output tensor
        :type y: torch.Tensor
        :return: the loss
        :rtype: float
        '''

        # Makes predictions
        y_pred, _, _ = self.model.forward(x)

        # Computes loss
        loss = self.loss_fn(y_pred, y)

        # Computes gradients
        loss.backward()

        # Clip the gradients
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        # Updates parameters and zeroes gradients
        self.optimiser.step()
        self.optimiser.zero_grad()

        # Returns the loss
        return loss.item()

    def train(self, train_loader, val_loader, n_epochs=50, start_ep=1, checkpoint_path=None):
        ''' Method to train the model.

        :param train_loader: dataloader for the training set
        :type train_loader: torch.utils.data.DataLoader
        :param val_loader: dataloader for the validation set
        :type val_loader: torch.utils.data.DataLoader
        :param n_epochs: number of epochs to train for, defaults to 50
        :type n_epochs: int, optional
        :param model_path: full path to where the model should be saved 
            at the end of the training, defaults to None
        :type model_path: str, optional
        '''
        for epoch in range(start_ep, n_epochs + 1):
            # initialise list to track training losses
            batch_losses = []
            # train the model
            self.model.train()
            for x_batch, y_batch in train_loader:
                # move to GPU if available
                x_batch, y_batch = self.to_device(x_batch, y_batch)
                # find the loss and update the model parameters
                loss = self.train_step(x_batch, y_batch)
                # store the batch loss
                batch_losses.append(loss)

            # calculate average training loss
            train_loss = np.mean(batch_losses)
            self.train_losses.append(train_loss)

            # validate the model
            self.model.eval()
            with torch.no_grad():
                # initialise list to track validation losses
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    # move to GPU if available
                    x_val, y_val = self.to_device(x_val, y_val)
                    # forward pass: compute predicted outputs by passing inputs to the model
                    y_pred, _, _ = self.model(x_val)
                    # calculate the batch loss
                    val_loss = self.loss_fn(y_pred, y_val).item()
                    # store the batch loss
                    batch_val_losses.append(val_loss)
                # calculate average validation loss
                val_loss = np.mean(batch_val_losses)
                self.scheduler.step(val_loss)
                self.val_losses.append(val_loss)

            # print training/validation statistics
            if (epoch < 10) | (epoch % 10 == 0):
                print('[Epoch: {}/{}] \tTraining Loss: {:.4f} \tValidation Loss: {:.4f}'.format(
                    epoch, n_epochs, train_loss, val_loss
                ))

            if checkpoint_path is not None:
                # create checkpoint variable and add important data
                checkpoint = {
                    'epoch': epoch,
                    'losses': (self.train_losses, self.val_losses),
                    'state_dict': self.model.state_dict(),
                    'optimiser': self.optimiser.state_dict(),
                }

                # save checkpoint
                self.save_ckp(checkpoint, False, checkpoint_path)

                # save the model separately if validation loss has decreased
                if val_loss <= self.min_val_loss:
                    print('Validation loss decreased ({:.4f} --> {:.4f}).  Saving model ...'.format(
                        self.min_val_loss, val_loss))

                    # save checkpoint as best model
                    self.save_ckp(checkpoint, True, checkpoint_path)
                    self.min_val_loss = val_loss

    def get_attentions(self, data_loader):
        ''' Method to return the attention weights from the temporal encoders.

        :param data_loader: dataloader for the data set
        :type data_loader: torch.utils.data.DataLoader
        :return: arrays of the daily and monthly attention values
        :rtype: tuple
        '''
        daily_atts, monthly_atts = [], []
        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(data_loader):
                x, y = self.to_device(x, y)

                _, energy_1, energy_2 = self.model(x)

                daily_atts.append(energy_1.detach().numpy())
                monthly_atts.append(energy_2.detach().numpy())

                if i == 0:
                    seq_len = x[0].size(1) if not isinstance(
                        x, list) else x.size(1)

        return np.vstack(daily_atts), np.vstack(monthly_atts).reshape(-1, seq_len)

    def to_device(self, x_batch, y_batch):
        ''' Method to convert the input and expected output batch to the GPU, if available.

        :param x_batch: tensor of a batch of input data
        :type x_batch: torch.Tensor
        :param y_batch: tensor of a batch of output data
        :type y_batch: torch.Tensor
        :return: the input and output tensors with the specified device
        :rtype: torch.Tensor
        '''
        if not isinstance(x_batch, list):
            x_batch = x_batch.to(device)
        else:
            x_batch = (x_batch[0].to(device), x_batch[1].to(device))

        y_batch = y_batch.to(device)

        return x_batch, y_batch

    def plot_losses(self):
        ''' Method to visualise the evolution of the training and validation losses.
        '''
        _ = plt.figure(figsize=(10, 7))
        plt.plot(self.train_losses, label='Training loss')
        plt.plot(self.val_losses, label='Validation loss')
        plt.legend()
        plt.title('Losses')
        plt.show()

    def save_ckp(self, state, is_best, checkpoint_path):
        ''' Method to save a checkpoint during training.

        :param state: checkpoint we want to save
        :type state: dict
        :param is_best: is this the best checkpoint (min validation loss)
        :type is_best: bool
        :param checkpoint_path: path to save checkpoint
        :type checkpoint_path: string
        '''
        f_path = checkpoint_path + 'ckp_ep_' + str(state['epoch']) + '.pt'
        # save checkpoint data to the path given, checkpoint_path
        torch.save(state, f_path)

        # if it is a best model, min validation loss
        if is_best:
            best_fpath = checkpoint_path + 'min_val_model.pt'
            # copy that checkpoint file to best path given
            shutil.copyfile(f_path, best_fpath)

    def load_ckp(self, checkpoint_fpath):
        ''' Method to load a previously saved checkpoint. 

        :param checkpoint_path: path to the saved checkpoint
        :type checkpoint_path: string
        '''
        # load check point
        checkpoint = torch.load(checkpoint_fpath)
        # initialize state_dict from checkpoint to model
        self.model.load_state_dict(checkpoint['state_dict'])
        # initialize optimizer from checkpoint to optimizer
        self.optimiser.load_state_dict(checkpoint['optimiser'])
        # initialize valid_loss_min from checkpoint to valid_loss_min
        self.train_losses, self.val_losses = checkpoint['losses']
        self.min_val_loss = np.amin(self.val_losses)


class ClassifierTrainer(Trainer):
    '''Extension of the wrapper class for classifier training and evaluation. 
    '''

    def __init__(self, model, loss_fn, optimiser, clip, n_classes):
        '''Class initialiser.
        '''
        super().__init__(
            model, loss_fn, optimiser, clip
        )
        self.n_classes = n_classes

    def evaluate(self, test_loader):
        ''' Evaluation method.

        :param test_loader: dataloader for the training set
        :type test_loader: torch.utils.data.DataLoader
        :return: array of the predictions and the true values
        :rtype: tuple
        '''
        predictions = []
        values = []

        self.model.eval()
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test, y_test = self.to_device(x_test, y_test)
                values.append(y_test.detach().numpy())

                y_pred, _, _ = self.model(x_test)

                if self.n_classes == 2:
                    predictions.append(torch.sigmoid(
                        y_pred).detach().numpy())
                else:
                    predictions.append(torch.log_softmax(
                        y_pred, dim=1).detach().numpy())

        if self.n_classes == 2:
            predictions = np.concatenate([p.squeeze().tolist()
                                          for p in predictions]).ravel()
        else:
            predictions = np.vstack(predictions)
        values = np.concatenate([v.squeeze().tolist()
                                for v in values]).ravel()

        return predictions, values

    def to_device(self, x_batch, y_batch):
        ''' Method to convert the input and expected output batch to the GPU, if available.
        '''
        x_batch, _ = super().to_device(x_batch, y_batch)

        if len(torch.unique(y_batch)) == 2:
            y_batch = y_batch.unsqueeze(1).to(device)
        else:
            y_batch = y_batch.long().to(device)

        return x_batch, y_batch


class RegressorTrainer(Trainer):
    ''' Extension of the wrapper class for regression model training and evaluation. 
    '''

    def __init__(self, model, loss_fn, optimiser, clip=0.5):
        ''' Class initialiser.
        '''
        super().__init__(
            model, loss_fn, optimiser, clip
        )

    def evaluate(self, test_loader):
        ''' Evaluation method.

        :param test_loader: dataloader for the training set
        :type test_loader: torch.utils.data.DataLoader
        :return: array of the predictions and the true values
        :rtype: tuple
        '''
        predictions = []
        values = []

        self.model.eval()
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test, y_test = self.to_device(x_test, y_test)
                values.append(y_test.detach().numpy())

                y_pred, _, _ = self.model(x_test)

                predictions.append(y_pred.detach().numpy())

        predictions, values = np.vstack(predictions), np.vstack(values)

        return predictions, values

    def to_device(self, x_batch, y_batch):
        ''' Method to convert the input and expected output batch to the GPU, if available.
        '''
        x_batch, _ = super().to_device(x_batch, y_batch)

        if len(y_batch.size()) == 1:
            y_batch = y_batch.view(-1, 1)
        y_batch = y_batch.float().to(device)

        return x_batch, y_batch
