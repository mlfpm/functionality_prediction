import torch
import torch.nn as nn

from src.models.time2vec import T2V
from src.models.rnn_encoder import Encoder, Attention
from src.utils.data_loader import ObjectDict


class MobilityEncoderNet(nn.Module):
    """
    Recurrent neural network regression with self-attention.
    Model description:
        * T2V for half-hourly data embedding (optional)
        * LSTM or GRU encoder for the embedded daily 48-slot input sequence
        * Scaled dot-product self-attention with the encoder outputs as keys and values and the hidden state as the query
        * LSTM or GRU encoder for the embedded 30-day input sequence
        * Scaled dot-product self-attention with the encoder outputs as keys and values and the hidden state as the query
        * Dense layer on top of attention outputs 
    """

    def __init__(self, hparams):
        super(MobilityEncoderNet, self).__init__()

        self.hparams = ObjectDict()
        self.hparams.update(
            hparams.__dict__ if hasattr(hparams, "__dict__") else hparams
        )

        if self.hparams.use_t2v:
            self.embedding = T2V(self.hparams.input_dim,
                                 self.hparams.embedding_dim, self.hparams.activation)

            self.encoder_1 = Encoder(
                self.hparams.input_dim * self.hparams.embedding_dim, self.hparams.hidden_dim_1,
                n_layers=self.hparams.n_layers, dropout=self.hparams.dropout, bidirectional=self.hparams.bidirectional,
                rnn_type=self.hparams.rnn_type
            )
        else:
            self.encoder_1 = Encoder(
                self.hparams.input_dim, self.hparams.hidden_dim_1,
                n_layers=self.hparams.n_layers, dropout=self.hparams.dropout, bidirectional=self.hparams.bidirectional,
                rnn_type=self.hparams.rnn_type
            )
        attention_dim_1 = self.hparams.hidden_dim_1 if not self.hparams.bidirectional else 2 * \
            self.hparams.hidden_dim_1
            
        self.attention_1 = Attention(
            attention_dim_1, attention_dim_1
        )

        self.encoder_2 = Encoder(
            attention_dim_1, self.hparams.hidden_dim_2, n_layers=self.hparams.n_layers,
            dropout=self.hparams.dropout, bidirectional=self.hparams.bidirectional,
            rnn_type=self.hparams.rnn_type
        )

        attention_dim_2 = self.hparams.hidden_dim_2 if not self.hparams.bidirectional else 2 * \
            self.hparams.hidden_dim_2
            
        self.attention_2 = Attention(
            attention_dim_2, attention_dim_2
        )

    def forward(self, x):
        """
        Forward step.

        :param x: input tensor of shape [batch_size, seq_len=30, day_len=48, input_dim]
        :type x: torch.Tensor
        :return: predicted outputs of shape [batch_size, output_dim]
        :rtype: torch.Tensor
        """
        # original input size
        bs, seq_len, day_len, n_feat = x.size()

        if self.hparams.use_t2v:
            # temporal feature encoder - needs input shape of [bs', 1, input_dim], where bs' = bs * seq_len * day_len
            t2v_emb = self.embedding(x.view(-1, 1, n_feat))

            # half-hour slots encoder - needs input shape of [bs'', day_len, input_dim * emb_dim], where bs'' = bs * seq_len
            rnn_out_1, hidden_1 = self.encoder_1(
                t2v_emb.view(bs*seq_len, day_len, -1))
        else:
            # half-hour slots encoder - needs input shape of [bs'', day_len, input_dim], where bs'' = bs * seq_len
            rnn_out_1, hidden_1 = self.encoder_1(
                x.view(bs*seq_len, day_len, n_feat))

        # if LSTM used take the cell state
        if isinstance(hidden_1, tuple):
            hidden_1 = hidden_1[1]

        # if bidirectional, need to concat the last 2 hidden layers
        if self.encoder_1.bidirectional:
            hidden_1 = torch.cat([hidden_1[-1], hidden_1[-2]], dim=1)
        else:
            hidden_1 = hidden_1[-1]

        # first self-attention
        energy_1, linear_combination_1 = self.attention_1(
            hidden_1, rnn_out_1, rnn_out_1)

        # 30-days encoding
        rnn_out_2, hidden_2 = self.encoder_2(
            linear_combination_1.view(bs, seq_len, -1))

        # if LSTM used take the cell state
        if isinstance(hidden_2, tuple):
            hidden_2 = hidden_2[1]

        # if bidirectional, need to concat the last 2 hidden layers
        if self.encoder_2.bidirectional:
            hidden_2 = torch.cat([hidden_2[-1], hidden_2[-2]], dim=1)
        else:
            hidden_2 = hidden_2[-1]

        # second self-attention
        energy_2, linear_combination_2 = self.attention_2(
            hidden_2, rnn_out_2, rnn_out_2)

        return linear_combination_2, energy_1, energy_2
