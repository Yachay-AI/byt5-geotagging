import copy

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import sys
import csv

from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys
import csv
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import T5EncoderModel, T5Config, T5Model
from transformers import AutoTokenizer
from scipy.sparse import coo_matrix, lil_matrix

MIN_DISTANCE = 500


class CharacterLevelCNN(nn.Module):
    def __init__(self, input_length=1014, input_dim=68,
                 n_conv_filters=256,
                 n_fc_neurons=1024):
        super(CharacterLevelCNN, self).__init__()
        embedding_dim = 256
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.conv1 = nn.Sequential(nn.Conv1d(embedding_dim, n_conv_filters, kernel_size=7, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3))
        self.conv2 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=7, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3))
        self.conv3 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3))

        dimension = n_conv_filters
        # self.fc1 = nn.Sequential(nn.Linear(dimension, n_fc_neurons), nn.Dropout(0.5))
        # self.fc2 = nn.Sequential(nn.Linear(n_fc_neurons, n_fc_neurons), nn.Dropout(0.5))
        self.fc1 = nn.Sequential(nn.Linear(dimension, n_fc_neurons), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(n_fc_neurons, n_fc_neurons), nn.ReLU())
        self.fc3 = nn.Linear(n_fc_neurons, 2)

        if n_conv_filters == 256 and n_fc_neurons == 1024:
            self._create_weights(mean=0.0, std=0.05)
        elif n_conv_filters == 1024 and n_fc_neurons == 2048:
            self._create_weights(mean=0.0, std=0.02)

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def forward(self, input):
        # input = input.transpose(1, 2)
        input = self.embedding(input)  # [2048, 140, 64]    [batch, input_length, embedding_dim]
        input = input.transpose(1, 2)
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)

        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output


class CharacterLevelCNN_classifier(nn.Module):
    def __init__(self, input_length=1014, input_dim=68,
                 n_conv_filters=256,
                 n_fc_neurons=1024, n_clusters_=100, language_count=None):
        super(CharacterLevelCNN_classifier, self).__init__()
        self.language_count = language_count
        embedding_dim = 256
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.conv1 = nn.Sequential(nn.Conv1d(embedding_dim, n_conv_filters, kernel_size=7, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3))
        self.conv2 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=7, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3))
        self.conv3 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3))

        dimension = n_conv_filters

        if language_count is not None:
            language_embedding_dim = language_count // 4
            self.language_embedding = nn.Embedding(language_count, language_embedding_dim)
            dimension += language_embedding_dim

        self.fc1 = nn.Sequential(nn.Linear(dimension, n_fc_neurons), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(n_fc_neurons, n_fc_neurons), nn.Dropout(0.5))
        #self.fc1 = nn.Sequential(nn.Linear(dimension, n_fc_neurons), nn.ReLU())
       # self.fc2 = nn.Sequential(nn.Linear(n_fc_neurons, n_fc_neurons), nn.ReLU())
        self.fc3 = nn.Linear(n_fc_neurons, n_clusters_)

        if n_conv_filters == 256 and n_fc_neurons == 1024:
            self._create_weights(mean=0.0, std=0.05)
        elif n_conv_filters == 1024 and n_fc_neurons == 2048:
            self._create_weights(mean=0.0, std=0.02)

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def forward(self, input, input_language=None):
        # input = input.transpose(1, 2)
        input = self.embedding(input)  # [2048, 140, 64]    [batch, input_length, embedding_dim]
        input = input.transpose(1, 2)
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)
        output = output.view(output.size(0), -1)

        if self.language_count is not None:
            language_emb = self.language_embedding(input_language)
            output = torch.cat([output, language_emb], dim=1)

        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output


def deleteEncodingLayers(model, num_layers_to_keep):  # must pass in the full bert model
    oldModuleList = model.encoder.block
    newModuleList = torch.nn.ModuleList()

    # Now iterate over all layers, only keepign only the relevant layers.
    for i in range(0, num_layers_to_keep):
        newModuleList.append(oldModuleList[i])

    # create a copy of the model, modify it with the new list, and return
    copyOfModel = copy.deepcopy(model)
    copyOfModel.encoder.block = newModuleList
    
    return copyOfModel

class ByT5_classifier(nn.Module):
    def __init__(self, n_clusters, model_name, language_count=None, keep_layer_count=None):
        super(ByT5_classifier, self).__init__()
        self.n_clusters_ = n_clusters
        self.language_count = language_count
        self.byt5 = T5EncoderModel.from_pretrained(model_name)
        if keep_layer_count is not None:
            self.byt5 = deleteEncodingLayers(self.byt5, keep_layer_count)
        '''
        self.byt5 = T5Model(T5Config({
          "vocab_size": 384,
          "d_model": 64,
          "d_kv": 8,
          "num_heads": 8,
          "d_ff": 128,
          "num_layers": 6
        })).encoder
        '''
        hidden_size = self.byt5.config.d_model

        if language_count is not None:
            language_embedding_dim = language_count // 4
            self.language_embedding = nn.Embedding(language_count, language_embedding_dim)
            hidden_size += language_embedding_dim

        self.fc3 = nn.Linear(hidden_size, n_clusters)

    def forward(self, input, input_language=None):
        input = self.byt5(input[:, 0, :].squeeze(1))['last_hidden_state']
        input = input[:, 0, :].squeeze(1)
        # if self.language_count is not None:
        #    language_emb = self.language_embedding(input_language)
        #    input = torch.cat([input, language_emb], dim=1)
        return self.fc3(input)



class ByT5ClusteredClassifierDataset(Dataset):
    def __init__(self, dataframe, tokenizer_name, tree, max_length=1014, language_df=None, smooth_labels=None):
        self.dataframe = dataframe
        self.max_length = max_length
        self.length = len(self.dataframe)
        self.tree = tree
        self.smooth_labels = smooth_labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.language_list = language_df['lang'].values.tolist() if language_df is not None else None
        self.warning = False

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raw_text = self.dataframe['text'].iloc[index]
        data = self.tokenizer(raw_text, truncation=True, padding="max_length", max_length=self.max_length,
                              return_tensors='pt')['input_ids']
        # values = self.scaler.transform(self.dataframe.iloc[index:index+1][['lat','lng']].values)
        # label = dbscan_predict(db, np.radians(self.dataframe.iloc[index:index+1][['lat','lng']].values))[0]
        # label = self.tree.query(self.dataframe.iloc[index][['lat', 'lng']].values.tolist())[1]
        if self.smooth_labels is not None:
            coords = [[np.deg2rad(x) for x in self.dataframe.iloc[index][['lat', 'lon']].values.tolist()]]
            int_label = self.tree.query(coords, k=1)[1][0][0]           
            coord_authors, author_weights = self.smooth_labels
            coord_author_index = self.dataframe.iloc[index]['coord_author_index']
            #coord_author = self.dataframe.iloc[index]['coordinates'] + "_" + self.dataframe['author_id'].astype(str).iloc[index]
            #if coord_author in coord_authors:
            label = author_weights[coord_author_index].toarray()[0]
            #else:
            #    label = np.array(author_weights.shape[1])
            #    label[int_label] = 1.0
            #    if not self.warning:
            #        self.warning = True
            #        print('Warning! no smooth label found for ', index, 'coord_author', coord_author, 'int_label', int_label)
        elif 'label' in self.dataframe.columns:
            label = self.dataframe.iloc[index]['label']
            int_label = label
        else:
            coords = [[np.deg2rad(x) for x in self.dataframe.iloc[index][['lat', 'lon']].values.tolist()]]
            label = self.tree.query(coords, k=1)[1][0][0]
            int_label = label

        language_id = 0
        if self.language_list is not None:
            try:
                language_id = self.language_list.index(self.dataframe['lang'].iloc[index])
            except:
                language_id = len(self.language_list) - 1

        return data, label, raw_text, self.dataframe.iloc[index]['lat'], \
            self.dataframe.iloc[index]['lon'], language_id, int_label
