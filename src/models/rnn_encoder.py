import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RNNS = ['LSTM', 'GRU']


class Encoder(nn.Module):
    """
    The Encoder layer will encode a temporal sequence into a single feature vector. 
    Implementation inspired by:
        https://github.com/mttk/rnn-classifier
    """

    def __init__(self, in_dim, hidden_dim, n_layers=1, dropout=0.,
                 bidirectional=False, rnn_type='LSTM'):
        """
        Class initialiser.

        :param in_dim: input size
        :type in_dim: int
        :param hidden_dim: hidden size
        :type hidden_dim: int
        :param n_layers: number of hidden layers to use, defaults to 1
        :type n_layers: int, optional
        :param dropout: dropout probability between the layers, when multiple layers are used, defaults to 0.
        :type dropout: float, optional
        :param bidirectional: if True then it becomes a bidirectional RNN, defaults to False
        :type bidirectional: bool, optional
        :param rnn_type: which type of RNN cell to use, defaults to 'LSTM'
        :type rnn_type: str, optional
        """
        assert rnn_type in RNNS, 'Use one of the following: {}'.format(
            str(RNNS))

        super(Encoder, self).__init__()

        self.bidirectional = bidirectional

        # fetch constructor from torch.nn, cleaner than if
        rnn_cell = getattr(nn, rnn_type)
        self.rnn = rnn_cell(in_dim, hidden_dim, n_layers,
                            dropout=dropout, bidirectional=bidirectional, batch_first=True)

    def forward(self, x, hidden=None):
        """
        Forward step.

        :param x: input tensor of size [batch_size, seq_len, in_dim]
        :type x: torch.Tensor
        :param hidden: initialised hidden states, defaults to None
        :type hidden: torch.Tensor, optional
        :return: the output of the recurrent layer of size [batch_size, seq_len, hidden_dim]
        :rtype: torch.Tensor
        """
        return self.rnn(x, hidden)


class Attention(nn.Module):
    """
    A simple attention layer.
    References:
        Ashish Vaswani, et al. 2017. Attention is all you need. In Proceedings of the 31st 
        International Conference on Neural Information Processing Systems (NIPS'17). 
        Curran Associates Inc., Red Hook, NY, USA, 6000–6010.
    """

    def __init__(self, query_and_key_dim, value_dim):
        super(Attention, self).__init__()
        self.d_k = query_and_key_dim
        self.d_v = value_dim
        self.scale = 1. / math.sqrt(query_and_key_dim)

    def forward(self, query, keys, values):
        """
        Forward step.

        :param query: queries of dimension d_k
        :type query: torch.Tensor
        :param keys: keys of dimension d_k
        :type keys: torch.Tensor
        :param values: values of dimension d_v
        :type values: torch.Tensor
        :return: scaled-dot product attention
        :rtype: torch.Tensor
        """

        query = query.unsqueeze(1)  # [Bxd_k] -> [Bx1xd_k]
        #print('query: ', query.size())
        keys = keys.transpose(1, 2)  # [BxTxd_k] -> [Bxd_kxT]
        #print('keys: ', keys.size())
        energy = torch.bmm(query, keys)  # [Bx1xd_k]x[Bxd_kxT] -> [Bx1xT]
        energy = F.softmax(energy.mul_(self.scale), dim=2)  # scale, normalize

        linear_combination = torch.bmm(energy, values).squeeze(
            1)  # [Bx1xT]x[BxTxd_v] -> [Bxd_v]

        return energy, linear_combination
