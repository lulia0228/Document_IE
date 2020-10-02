import  torch
from    torch import nn
from    torch.nn import functional as F
from    utils import sparse_dropout, dot
import  numpy as np

class GraphConvolution(nn.Module):

    # def __init__(self, input_dim, output_dim, num_features_nonzero,
    def __init__(self, input_dim, output_dim,
                 dropout=0.,
                 is_sparse_inputs=False,
                 bias=False,
                 activation = F.relu,
                 featureless=False):
        super(GraphConvolution, self).__init__()


        # self.dropout = dropout
        if dropout:
            self.dropout = dropout
        else:
            self.dropout = 0.

        self.bias = bias
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        # self.num_features_nonzero = num_features_nonzero

        def glorot(shape, name=None):
            """Glorot & Bengio (AISTATS 2010) init."""
            init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
            # initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
            # return tf.Variable(initial, name=name)
            # init = torch.DoubleTensor(shape[0], shape[1]).uniform_(-init_range,init_range )
            init = torch.FloatTensor(shape[0], shape[1]).uniform_(-init_range,init_range )
            return init

        # self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.weight = nn.Parameter(glorot((input_dim, output_dim)))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))


    def forward(self, inputs):
        # print('inputs:', inputs)
        x, support = inputs

        if self.training and self.is_sparse_inputs:
            # x = sparse_dropout(x, self.dropout, self.num_features_nonzero)
            x = x
        elif self.training:
            x = F.dropout(x, self.dropout)


        # convolve
        if not self.featureless: # if it has features x
            if self.is_sparse_inputs:
                xw = torch.sparse.mm(x, self.weight)

            else:
                # print(x.dtype)
                xw = torch.mm(x, self.weight)
        else:
            xw = self.weight
        # print(support.dtype)
        # print(xw.dtype)
        out = torch.sparse.mm(support, xw)

        if self.bias is not None:
            out += self.bias

        return self.activation(out), support

