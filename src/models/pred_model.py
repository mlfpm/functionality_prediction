import torch
import torch.nn as nn

from src.models.temp_embedding import MobilityEncoderNet


class WHODASPredictor(MobilityEncoderNet):
    """
    Recurrent neural network regression with self-attention.
    Model description:
        * T2V for half-hourly data embedding
        * LSTM or GRU encoder for the embedded daily 48-slot input sequence
        * Scaled dot-product self-attention with the encoder outputs as keys and values and the hidden state as the query
        * LSTM or GRU encoder for the embedded 30-day input sequence
        * Scaled dot-product self-attention with the encoder outputs as keys and values and the hidden state as the query
        * Demographic data concatenated with the attention outputs 
        * Final dense layer for predictions
    """

    def __init__(self, hparams):
        """
        Class initialiser.

        :param hparams: dictionary of model parameters
        :type hparams: dict
        """
        super(WHODASPredictor, self).__init__(hparams)

        attention_dim_2 = self.hparams.hidden_dim_2 if not self.hparams.bidirectional else 2 * \
            self.hparams.hidden_dim_2

        self.fc = nn.Linear(
            attention_dim_2 + self.hparams.demogr_input_dim, self.hparams.output_dim
        )

    def forward(self, x):
        """
        Forward step.

        :param x: input tensors of shape [batch_size, seq_len=30, day_len=48, temp_input_dim] and [batch_size, demogr_input_dim]
        :type x: tuple
        :return: predicted outputs of shape [batch_size, output_dim]
        :rtype: torch.Tensor
        """
        if type(x) is tuple:
            x_temp, x_demogr = x
            # compute temporal embeddings
            temp_out, energy_1, energy_2 = super(
                WHODASPredictor, self).forward(x_temp)

            # concatenate with the demographic data and compute predictions
            logits = self.fc(torch.relu(
                torch.cat([temp_out, x_demogr], -1)))
        else:
            # compute temporal embeddings
            temp_out, energy_1, energy_2 = super(
                WHODASPredictor, self).forward(x)

            # and compute predictions
            logits = self.fc(torch.relu(temp_out))

        return logits, energy_1, energy_2
