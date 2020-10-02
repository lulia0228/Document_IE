import  torch
from    torch import nn
from    torch.nn import functional as F
from    layer import GraphConvolution

from    config import args

class GCN(nn.Module):


    # def __init__(self, input_dim, output_dim, num_features_nonzero):
    def __init__(self, input_dim, output_dim):
        super(GCN, self).__init__()

        self.input_dim = input_dim # 1433
        self.output_dim = output_dim

        print('input dim:', input_dim)
        print('output dim:', output_dim)
        # print('num_features_nonzero:', num_features_nonzero)


        # self.layers = nn.Sequential(GraphConvolution(self.input_dim, args.hidden, num_features_nonzero,
        self.layers = nn.Sequential(GraphConvolution(self.input_dim, args.hidden,
                                                     activation=F.relu,
                                                     dropout=False,
                                                     is_sparse_inputs=True),

                                    # GraphConvolution(args.hidden, output_dim, num_features_nonzero,
                                    GraphConvolution(args.hidden, args.hidden1,
                                                     activation=F.relu,
                                                     dropout=False,
                                                     is_sparse_inputs=False),

                                    GraphConvolution(args.hidden1, self.output_dim,
                                                     activation=lambda x: x,
                                                     dropout=args.dropout,
                                                     is_sparse_inputs=False),
                                    )

    def forward(self, inputs):
        x, support = inputs

        x = self.layers((x, support))

        return x

    def l2_loss(self):

        layer = self.layers.children()
        layer = next(iter(layer))

        loss = None

        for p in layer.parameters():
            if loss is None:
                loss = p.pow(2).sum()
            else:
                loss += p.pow(2).sum()

        return loss


class LSTM_GCN(nn.Module):

    # def __init__(self, input_dim, output_dim, num_features_nonzero):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, output_dim):
        super(LSTM_GCN, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim # LSTM隐层神经元数目
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # self.input_dim = input_dim # 1433
        self.output_dim = output_dim

        # print('input dim:', input_dim)
        print('output dim:', output_dim)
        # print('num_features_nonzero:', num_features_nonzero)


        # self.layers = nn.Sequential(GraphConvolution(self.input_dim, args.hidden, num_features_nonzero,
        self.layers = nn.Sequential(GraphConvolution(self.hidden_dim, args.hidden,
                                                     activation=F.relu,
                                                     dropout=False,
                                                     # is_sparse_inputs=True),
                                                     is_sparse_inputs=False),

                                    # # GraphConvolution(args.hidden, output_dim, num_features_nonzero,
                                    # GraphConvolution(args.hidden, args.hidden1,
                                    #                  activation=F.relu,
                                    #                  dropout=False,
                                    #                  is_sparse_inputs=False),

                                    GraphConvolution(args.hidden, self.output_dim,
                                                     activation=lambda x: x,
                                                     dropout=args.dropout,
                                                     is_sparse_inputs=False),
                                    )

    def forward(self, inputs):
        features, support = inputs
        tensor_list = []
        for sentence in features:
            embeds = self.word_embeddings(sentence)
            lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
            x =  lstm_out.view(len(sentence), -1)
            x = lstm_out[-1] # 取序列最后一个位置的隐藏向量
            tensor_list.append(x)
            # print("***", x)
            # print("***", x.shape)
        graph_input = torch.cat(tensor_list, dim=0)
        # print(graph_input)
        # print(graph_input.size())
        # exit()

        # x = self.layers((x, support))
        x = self.layers((graph_input, support))

        return x

    def l2_loss(self):

        layer = self.layers.children()
        layer = next(iter(layer))

        loss = None

        for p in layer.parameters():
            if loss is None:
                loss = p.pow(2).sum()
            else:
                loss += p.pow(2).sum()

        return loss