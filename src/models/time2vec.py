import torch
import torch.nn as nn


class T2V(nn.Module):
    """
    The Time2Vec layer will embed the daily time series into a lower 
    dimensional space. 
    Implementation inspired by:
        https://github.com/evelynmitchell/Time2Vec-PyTorch
    References: 
        Kazemi, S.M., Goel, R., Eghbali, S., Ramanan, J., Sahota, J., Thakur, S., Wu, S., Smyth, C., Poupart, P., & Brubaker, M.A. (2019). Time2Vec: Learning a Vector Representation of Time. ArXiv, abs/1907.05321.
    """

    def __init__(self, in_dim, emb_dim, activation='sin'):
        """
        Class initialiser. 

        :param emb_dim: embedding dimension 
        :type emb_dim: int
        :param activation: type of activation to use - 'sin' or 'cos', defaults to 'sin'
        :type activation: str, optional
        """
        super(T2V, self).__init__()

        self.activation = activation

        self.wb = nn.parameter.Parameter(torch.randn(in_dim, 1))
        self.bb = nn.parameter.Parameter(torch.randn(in_dim, 1))

        self.wa = nn.parameter.Parameter(torch.randn(in_dim, emb_dim-1))
        self.ba = nn.parameter.Parameter(torch.randn(in_dim, emb_dim-1))

    def forward(self, x):
        """
        Forward step.

        :param x: input tensor of shape [batch_size, 1, in_dim]
        :type x: torch.Tensor
        :return: the embedded entries as a tensor of shape  [batch_size, in_dim, emb_dim]
        :rtype: torch.Tensor
        """
        bias = torch.matmul(x, self.wb) + self.bb

        if self.activation == 'sin':
            wgts = torch.sin(torch.matmul(x, self.wa) + self.ba)
        else:
            wgts = torch.cos(torch.matmul(x, self.wa) + self.ba)

        return torch.cat([bias, wgts], -1)
