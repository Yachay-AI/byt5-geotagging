import torch
import torch.nn.functional as F
from torch import nn

class CharLSTMModel(nn.Module):
    def __init__(self, args):
        super(CharLSTMModel, self).__init__()
        self._token_embed = nn.Embedding(256, 250, 255)
        self._lstm = nn.LSTM(250, 250, 2, bidirectional=True, batch_first=True)
        self._ffn = nn.Linear(500, 2)

    def forward(self, byte_tokens, word_tokens, features_only=False):
        input_ids = byte_tokens.input_ids
        embed = self._token_embed(input_ids)
        context_embeds = self._lstm(embed)[0]
        pool = torch.mean(context_embeds, dim=1)
        if features_only:
            return pool
        else:
            return self._ffn(pool)


class CharCNNModel(nn.Module):
    def __init__(self, args):
        super(CharCNNModel, self).__init__()
        self._token_embed = nn.Embedding(256, 150, 255)

        self._conv1 = nn.Sequential(
            nn.Conv1d(150, 128, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1)
        )
        self._conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=7, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1)
        )
        self._conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self._conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self._conv5 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self._conv6 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=1)
        )

        if hasattr(args, 'dropout'):
            self.dropout = args.dropout
        else:
            self.dropout = 0.0

        self._fc1 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p=self.dropout))
        self._fc2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p=self.dropout))
        self._fc3 = nn.Linear(64, 2)

    def forward(self, byte_tokens, word_tokens, features_only=False):
        input_ids = byte_tokens.input_ids
        x = self._token_embed(input_ids)
        x = x.permute(0, 2, 1)
        # conv layers
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._conv3(x)
        x = self._conv4(x)
        x = self._conv5(x)
        x = self._conv6(x).squeeze()
        # linear layers
        x = self._fc1(x)
        x = self._fc2(x)
        if features_only:
            return x

        # final linear layer
        x = self._fc3(x)
        return x


class CompositeModel(nn.Module):
    T5_HIDDEN_SIZE = 1472
    CNN_HIDDEN_SIZE = 64

    def __init__(self, args):
        super(CompositeModel, self).__init__()
        self.dropout = args.dropout
        self.arch = args.arch
        self.reduce_layer = args.reduce_layer
        self.conf_estim = False

        if args.arch == 'char_lstm':
            self._encoder = CharLSTMModel(args)
            concat_dim = self._encoder._lstm.hidden_size * 2
        elif args.arch == 'char_cnn':
            self._encoder = CharCNNModel(args)
            concat_dim = self._encoder._fc2[0].out_features
        else:
            print('Invalid architecture choice')
            return

        reduce_dim = 100 if args.reduce_layer else concat_dim
        self._reduce = nn.Linear(concat_dim, 100)

        if args.conf_estim:
            self._predict_head = nn.Linear(reduce_dim, 2)
            self._conf_head = nn.Linear(reduce_dim, 1)
            self.conf_estim = True
        else:
            self._head = nn.Linear(reduce_dim, 2)

    def forward(self, byte_tokens, word_tokens):
        if self.arch == 'bert' or self.arch == 'byt5':
            text_encoding = F.dropout(self._encoder(byte_tokens, word_tokens),
                                      p=self.dropout, training=self.training)
        else:
            text_encoding = F.dropout(self._encoder(byte_tokens, word_tokens, features_only=True),
                                      p=self.dropout, training=self.training)

        if self.reduce_layer:
            text_encoding = self._reduce(text_encoding)

        if self.conf_estim:
            return self._predict_head(text_encoding), self._conf_head(text_encoding)
        else:
            return self._head(text_encoding)