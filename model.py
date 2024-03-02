import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import lil_matrix
from sklearn.neighbors import BallTree
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import T5EncoderModel
from transformers import ByT5Tokenizer

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

        # self.fc1 = nn.Sequential(nn.Linear(dimension, n_fc_neurons), nn.Dropout(0.5))
        # self.fc2 = nn.Sequential(nn.Linear(n_fc_neurons, n_fc_neurons), nn.Dropout(0.5))
        self.fc1 = nn.Sequential(nn.Linear(dimension, n_fc_neurons), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(n_fc_neurons, n_fc_neurons), nn.ReLU())
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


def deleteEncodingLayersDeberta(model, num_layers_to_keep):
    oldModuleList = model.deberta.encoder.layer
    newModuleList = torch.nn.ModuleList()

    # Keep only the relevant layers
    for i in range(0, num_layers_to_keep):
        newModuleList.append(oldModuleList[i])

    # Create a copy of the model and modify it with the new list
    copyOfModel = copy.deepcopy(model)
    copyOfModel.deberta.encoder.layer = newModuleList

    return copyOfModel
    

class ByT5_classifierNew(nn.Module):
    def __init__(self, n_clusters, model_name, language_count=None, keep_layer_count=None,
                 intermediate_cluster_df=None):
        super(ByT5_classifierNew, self).__init__()
        self.n_clusters_ = n_clusters
        self.language_count = language_count
        self.byt5 = T5EncoderModel.from_pretrained(model_name)
        self.intermediate_cluster_df = intermediate_cluster_df
        if keep_layer_count is not None:
            self.byt5 = deleteEncodingLayers(self.byt5, keep_layer_count)
        hidden_size = self.byt5.config.d_model

        if language_count is not None:
            language_embedding_dim = language_count // 4
            self.language_embedding = nn.Embedding(language_count, language_embedding_dim)
            hidden_size += language_embedding_dim

        self.fc3 = nn.Linear(hidden_size, n_clusters)
        if self.intermediate_cluster_df is not None:
            self.fc_intermediate = nn.Linear(hidden_size, len(self.intermediate_cluster_df))

    def forward(self, input, input_language=None):
        input = self.byt5(input[:, 0, :].squeeze(1))['last_hidden_state']
        # store hidden state from byt5's 4-th layer
        intermediate_input = input[:, 4, :].squeeze(1)
        input = input[:, 0, :].squeeze(1)
        # if self.language_count is not None:
        #    language_emb = self.language_embedding(input_language)
        #    input = torch.cat([input, language_emb], dim=1)
        if self.intermediate_cluster_df is not None:
            # take hidden state from byt5's 4-th layer and pass it through fc_intermediate
            return self.fc3(input), self.fc_intermediate(intermediate_input)
        else:
            return self.fc3(input)


class ByT5_regressorNew(nn.Module):
    def __init__(self, model_name, language_count=None, keep_layer_count=None,
                 intermediate_cluster_df=None):
        super(ByT5_regressorNew, self).__init__()
        self.language_count = language_count
        self.byt5 = T5EncoderModel.from_pretrained(model_name)
        self.intermediate_cluster_df = intermediate_cluster_df
        if keep_layer_count is not None:
            self.byt5 = deleteEncodingLayers(self.byt5, keep_layer_count)
        hidden_size = self.byt5.config.d_model

        if language_count is not None:
            language_embedding_dim = language_count // 4
            self.language_embedding = nn.Embedding(language_count, language_embedding_dim)
            hidden_size += language_embedding_dim

        self.fc3 = nn.Linear(hidden_size, 2)
        if self.intermediate_cluster_df is not None:
            self.fc_intermediate = nn.Linear(hidden_size, len(self.intermediate_cluster_df))

    def forward(self, input, input_language=None):
        input = self.byt5(input[:, 0, :].squeeze(1))['last_hidden_state']
        # store hidden state from byt5's 4-th layer
        intermediate_input = input[:, 4, :].squeeze(1)
        input = input[:, 0, :].squeeze(1)
        # if self.language_count is not None:
        #    language_emb = self.language_embedding(input_language)
        #    input = torch.cat([input, language_emb], dim=1)
        if self.intermediate_cluster_df is not None:
            # take hidden state from byt5's 4-th layer and pass it through fc_intermediate
            return self.fc3(input), self.fc_intermediate(intermediate_input)
        else:
            return self.fc3(input)



class ModifiedCharCNN(nn.Module):
    def __init__(self, input_length=1014, input_dim=68,
                 n_conv_filters=256, n_fc_neurons=1024, n_clusters_=100, model_name="google/byt5-small"):
        super(ModifiedCharCNN, self).__init__()

        # Tokenizer initialization
        self.tokenizer = ByT5Tokenizer.from_pretrained(model_name)
        self.embedding_dim = self.tokenizer.vocab_size
        self.embedding = nn.Embedding(self.embedding_dim, self.embedding_dim, padding_idx=self.tokenizer.pad_token_id)

        # Convolution layers
        self.conv1 = nn.Sequential(nn.Conv1d(self.embedding_dim, n_conv_filters, kernel_size=7, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3))
        self.conv2 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=7, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3))
        self.conv3 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3))

        # Fully connected layers
        self.fc1 = nn.Sequential(nn.Linear(n_conv_filters, n_fc_neurons), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(n_fc_neurons, n_fc_neurons), nn.ReLU())
        self.fc3 = nn.Linear(n_fc_neurons, n_clusters_)

    def forward(self, input, input_language=None):
        # Tokenize and convert to embeddings
        input = self.embedding(input[:, 0, :].squeeze(1))
        input = input.transpose(1, 2)

        # Convolution operations
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)
        output = output.view(output.size(0), -1)

        # Fully connected operations
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output


class MultiLayerCharLSTM(nn.Module):
    def __init__(self, input_length=1014, n_lstm_units=256, n_layers=3,
                 n_fc_neurons=1024, n_clusters_=100, model_name="google/byt5-small"):
        super(MultiLayerCharLSTM, self).__init__()

        # Tokenizer initialization
        self.tokenizer = ByT5Tokenizer.from_pretrained(model_name)
        self.embedding_dim = self.tokenizer.vocab_size
        self.embedding = nn.Embedding(self.embedding_dim, self.embedding_dim, padding_idx=self.tokenizer.pad_token_id)

        # Multi-layer Bidirectional LSTM
        self.lstm = nn.LSTM(self.embedding_dim, n_lstm_units, num_layers=n_layers,
                            batch_first=True, bidirectional=True)

        # Due to bidirectionality, the LSTM output will be of size 2*n_lstm_units
        # Fully connected layers
        self.fc1 = nn.Sequential(nn.Linear(2 * n_lstm_units, n_fc_neurons), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(n_fc_neurons, n_fc_neurons), nn.ReLU())
        self.fc3 = nn.Linear(n_fc_neurons, n_clusters_)

    def forward(self, input, input_language=None):
        # Tokenize and convert to embeddings
        input = self.embedding(input[:, 0, :].squeeze(1))

        # LSTM operations
        output, (h_n, c_n) = self.lstm(input)
        output = output[:, -1, :]  # Take the output from the last LSTM step

        # Fully connected operations
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output


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
        # store hidden state from byt5's 4-th layer
        intermediate_input = input[:, 4, :].squeeze(1)
        input = input[:, 0, :].squeeze(1)
        # if self.language_count is not None:
        #    language_emb = self.language_embedding(input_language)
        #    input = torch.cat([input, language_emb], dim=1)

        return self.fc3(input)


class ByT5_regressor(nn.Module):
    def __init__(self, model_name, language_count=None, keep_layer_count=None):
        super(ByT5_regressor, self).__init__()
        self.language_count = language_count
        self.byt5 = T5EncoderModel.from_pretrained(model_name)
        if keep_layer_count is not None:
            self.byt5 = deleteEncodingLayers(self.byt5, keep_layer_count)
        hidden_size = self.byt5.config.d_model

        if language_count is not None:
            language_embedding_dim = language_count // 4
            self.language_embedding = nn.Embedding(language_count, language_embedding_dim)
            hidden_size += language_embedding_dim

        self.fc3 = nn.Linear(hidden_size, 2)

    def forward(self, input, input_language=None):
        input = self.byt5(input[:, 0, :].squeeze(1))['last_hidden_state']
        # store hidden state from byt5's 4-th layer
        intermediate_input = input[:, 4, :].squeeze(1)
        input = input[:, 0, :].squeeze(1)
        # if self.language_count is not None:
        #    language_emb = self.language_embedding(input_language)
        #    input = torch.cat([input, language_emb], dim=1)

        return self.fc3(input)

class TweetDataset(Dataset):
    def __init__(self, dataframe, scaler, vocabulary, max_length=1014):
        self.dataframe = dataframe
        self.vocabulary = vocabulary
        self.identity_mat = np.identity(len(self.vocabulary))
        self.max_length = max_length
        self.length = len(self.dataframe)
        self.scaler = scaler

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raw_text = self.dataframe['text'].iloc[index]
        data = [self.vocabulary.index(i) for i in list(raw_text) if i in self.vocabulary]
        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif 0 < len(data) < self.max_length:
            data = np.concatenate(
                (data, np.zeros((self.max_length - len(data)), dtype=np.float32)))
        elif len(data) == 0:
            data = np.zeros((self.max_length), dtype=np.float32)
        values = self.scaler.transform(self.dataframe.iloc[index:index + 1][['lat', 'lon']].values)
        return torch.tensor(data, dtype=int), values[0][0], values[0][1], raw_text


class ErrorPredictDataset(Dataset):
    def __init__(self, dataframe, scaler, vocabulary, max_length=1014, min_distance=500):
        self.dataframe = dataframe
        self.vocabulary = vocabulary
        self.identity_mat = np.identity(len(self.vocabulary))
        self.max_length = max_length
        self.length = len(self.dataframe)
        self.scaler = scaler
        self.min_distance = min_distance

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raw_text = self.dataframe['text'].iloc[index]
        data = [self.vocabulary.index(i) for i in list(raw_text) if i in self.vocabulary]
        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif 0 < len(data) < self.max_length:
            data = np.concatenate(
                (data, np.zeros((self.max_length - len(data)), dtype=np.float32)))
        elif len(data) == 0:
            data = np.zeros((self.max_length), dtype=np.float32)

        raw_distance = self.dataframe.iloc[index]['distance']
        label = 1 if raw_distance < self.min_distance else 0

        return torch.tensor(data, dtype=int), label, raw_text, raw_distance


class ClusteredClassifierDataset(Dataset):
    def __init__(self, dataframe, scaler, vocabulary, tree, max_length=1014, merges=None, language_df=None):
        self.dataframe = dataframe
        self.vocabulary = vocabulary
        self.identity_mat = np.identity(len(self.vocabulary))
        self.max_length = max_length
        self.length = len(self.dataframe)
        self.tree = tree
        self.scaler = scaler
        self.merges = merges
        self.language_list = language_df['lang'].values.tolist() if language_df is not None else None

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raw_text = self.dataframe['text'].iloc[index]
        data = [self.vocabulary.index(i) for i in list(raw_text) if i in self.vocabulary]
        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif 0 < len(data) < self.max_length:
            data = np.concatenate(
                (data, np.zeros((self.max_length - len(data)), dtype=np.float32)))
        elif len(data) == 0:
            data = np.zeros((self.max_length), dtype=np.float32)
        # values = self.scaler.transform(self.dataframe.iloc[index:index+1][['lat','lon']].values)
        # label = dbscan_predict(db, np.radians(self.dataframe.iloc[index:index+1][['lat','lng']].values))[0]
        # label = self.tree.query(self.dataframe.iloc[index][['lat', 'lng']].values.tolist())[1]
        if 'label' in self.dataframe.columns:
            label = self.dataframe.iloc[index]['label']
        else:
            coords = [[np.deg2rad(x) for x in self.dataframe.iloc[index][['lat', 'lon']].values.tolist()]]
            label = self.tree.query(coords, k=1)[1][0][0]
        language_id = 0
        if self.language_list is not None:
            try:
                language_id = self.language_list.index(self.dataframe['lang'].iloc[index])
            except:
                language_id = len(self.language_list) - 1

        return torch.tensor(data, dtype=int), label, raw_text, self.dataframe.iloc[index]['lat'], \
            self.dataframe.iloc[index]['lon'], language_id


class ByT5ClusteredClassifierDataset(Dataset):
    def __init__(self, dataframe, tokenizer_name, tree, max_length=1014, language_df=None, smooth_labels=None,
                 intermediate_cluster_df=None, model_type='byt5'):
        self.dataframe = dataframe
        self.max_length = max_length
        self.length = len(self.dataframe)
        self.tree = tree
        self.model_type = model_type
        self.smooth_labels = smooth_labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.language_list = language_df['lang'].values.tolist() if language_df is not None else None
        if intermediate_cluster_df is not None:
            self.intermediate_cluster_df = intermediate_cluster_df
            print(intermediate_cluster_df)
            self.intermediate_tree = BallTree(np.deg2rad(intermediate_cluster_df[['lat', 'lng']].values),
                                              metric='haversine')
        else:
            self.intermediate_cluster_df = None
        self.warning = False

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raw_text = self.dataframe['text'].iloc[index]
        data = self.tokenizer(raw_text, truncation=True, padding="max_length", max_length=self.max_length,
                              return_tensors='pt')
        # values = self.scaler.transform(self.dataframe.iloc[index:index+1][['lat','lng']].values)
        # label = dbscan_predict(db, np.radians(self.dataframe.iloc[index:index+1][['lat','lng']].values))[0]
        # label = self.tree.query(self.dataframe.iloc[index][['lat', 'lng']].values.tolist())[1]
        coords = [[np.deg2rad(x) for x in self.dataframe.iloc[index][['lat', 'lon']].values.tolist()]]
        if self.smooth_labels is not None:
            int_label = self.tree.query(coords, k=1)[1][0][0]
            coord_authors, author_weights = self.smooth_labels
            coord_author_index = self.dataframe.iloc[index]['coord_author_index']
            # coord_author = self.dataframe.iloc[index]['coordinates'] + "_" + self.dataframe['author_id'].astype(str).iloc[index]
            # if coord_author in coord_authors:
            label = author_weights[coord_author_index].toarray()[0]
            # else:
            #    label = np.array(author_weights.shape[1])
            #    label[int_label] = 1.0
            #    if not self.warning:
            #        self.warning = True
            #        print('Warning! no smooth label found for ', index, 'coord_author', coord_author, 'int_label', int_label)
        elif 'label' in self.dataframe.columns:
            label = self.dataframe.iloc[index]['label']
            int_label = label
        else:
            label = self.tree.query(coords, k=1)[1][0][0]
            int_label = label

        language_id = 0
        if self.language_list is not None:
            try:
                language_id = self.language_list.index(self.dataframe['lang'].iloc[index])
            except:
                language_id = len(self.language_list) - 1

        confidence_weight = 1.0
        if 'confidence_weight' in self.dataframe.columns:
            confidence_weight = self.dataframe.iloc[index]['confidence_weight']

        if self.intermediate_cluster_df is not None:
            # lookup in tree to find nearest intermediate cluster
            intermediate_cluster = self.intermediate_tree.query(coords, k=1)[1][0][0]
        else:
            intermediate_cluster = 0

        if self.model_type == 'bert':
            data['input_ids'] = data['input_ids'].squeeze(0)
            data['attention_mask'] = data['attention_mask'].squeeze(0)

        return data['input_ids'], label, raw_text, self.dataframe.iloc[index]['lat'], \
            self.dataframe.iloc[index]['lon'], language_id, int_label, confidence_weight, intermediate_cluster, data['attention_mask']


def calculate_smooth_labels(df, cluster_df, tree, nearest_count, nearest_weight, author_weight, nearest_distance,
                            top100_wrong, wrong_weight):
    df['coord_author'] = df['coordinates'] + "_" + df['author_id'].astype(int).astype(str)
    df['author_id'] = df['author_id'].astype(int)

    coords = df['coordinates'].unique()

    coords_cache = {}
    for coord in tqdm(coords):
        if coord in coords_cache:
            nearest_distances, nearest_clusters, true_label = coords_cache[coord]
        else:
            lon, lat = [float(x) for x in coord.split("_")]
            coords = [[np.deg2rad(lat), np.deg2rad(lon)]]
            nearest_distances, nearest_clusters = tree.query(coords, k=nearest_count)
            true_label = nearest_clusters[0][0]
            coords_cache[coord] = nearest_distances, nearest_clusters, true_label

    df_new = df.groupby('author_id').first()
    coord_authors = df['coord_author'].unique()

    smoothed_labels_all = lil_matrix((len(coord_authors), len(cluster_df)))
    k = 0
    for author_index, coord_author in tqdm(enumerate(coord_authors), total=len(coord_authors)):
        lon, lat, author_id = coord_author.split("_")
        coordinates = lon + "_" + lat
        row = df_new.loc[int(author_id)]

        if coordinates in coords_cache:
            nearest_distances, nearest_clusters, true_label = coords_cache[coordinates]
        else:
            coords = [[np.deg2rad(row['lat']), np.deg2rad(row['lon'])]]
            nearest_distances, nearest_clusters = tree.query(coords, k=nearest_count)
            true_label = nearest_clusters[0][0]
            coords_cache[coordinates] = nearest_distances, nearest_clusters, true_label
        #author_clusters = row['rel_clusters']
        smoothed_labels_all[k, true_label] = 1.0 - nearest_weight - author_weight - wrong_weight
        author_weight_sum = sum(row['rel_weights'])
        #for j in range(len(author_clusters)):
        #    smoothed_labels_all[k, author_clusters[j]] += row['rel_weights'][j] / author_weight_sum * author_weight
        for j in range(len(top100_wrong[true_label])):
            smoothed_labels_all[k, top100_wrong[true_label][j]] += wrong_weight / len(top100_wrong[true_label])
        if nearest_distance is not None:
            earth_radius = 6371
            nearest_ids = (nearest_distances * earth_radius < nearest_distance) & (nearest_distances > 0)
            filtered_nearest_clusters = nearest_clusters[nearest_ids]
            filtered_nearest_distances = nearest_distances[nearest_ids]
            nearest_weights = 1.0 / filtered_nearest_distances
            for j in range(len(filtered_nearest_clusters)):
                smoothed_labels_all[k, filtered_nearest_clusters[j]] += nearest_weight * nearest_weights[j] / sum(
                    nearest_weights)
        else:
            for j in range(nearest_count):
                smoothed_labels_all[k, nearest_clusters[0][j]] += 1.0 / nearest_count * nearest_weight
        k += 1

    author_indices = {author_id: idx for idx, author_id in enumerate(coord_authors)}
    df['coord_author_index'] = df['coord_author'].map(author_indices)

    return coord_authors, smoothed_labels_all.tocsr()


class DistanceBasedLoss(CrossEntropyLoss):
    def __init__(self, weight=None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0,
                 distance_between_clusters=None) -> None:
        super(DistanceBasedLoss, self).__init__(weight, size_average, ignore_index, reduce, reduction)
        self.label_smoothing = label_smoothing
        self.distance_between_clusters = distance_between_clusters

    def forward(self, input, target_classes):
        target_proba = torch.zeros_like(input)
        for i in range(input.shape[1]):  # samples in batch
            t = target_classes[i]
            inv_distances = self.distance_between_clusters[t, :].pow_(-1)
            inv_distances[t] = 0.0
            target_proba = torch.nn.Softmax(dim=0)(inv_distances) * self.label_smoothing
            target_proba[i, t] = 1.0 - self.label_smoothing

        return F.cross_entropy(input, target_proba, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)


def cost_sensitive_cross_entropy_loss(output, target, cost_matrix, alpha=1.0):
    num_classes = cost_matrix.shape[0]
    cost_matrix = alpha * cost_matrix

    # Convert the target to one-hot encoding
    target_onehot = torch.zeros_like(output)
    target_onehot.scatter_(1, target.unsqueeze(1), 1)

    # Calculate the weighted cross-entropy loss
    loss = (- target_onehot * torch.log(output + 1e-6) * cost_matrix).sum(dim=1)
    return loss.mean()


def lat_fix_tensor(val):
    return torch.maximum(torch.minimum(val, torch.ones_like(val) * 90), torch.ones_like(val) * -90)

def haversine_loss_with_penalty(true_lat, true_lon, pred_lat, pred_lon, device='cuda'):
    # Convert latitude and longitude from degrees to radians
    p = math.pi / 180

    # Scaling from [-1, 1]
    pred_lat = pred_lat * 90
    pred_lon = pred_lon * 180

    # Apply transformations to latitude and longitude, if necessary
    # true_lat, true_lon = your_transform(true_lat, true_lon)
    # pred_lat, pred_lon = your_transform(pred_lat, pred_lon)

    # Ensure lat values are in [-90, 90]
    pred_lat = lat_fix_tensor(pred_lat)

    t1 = (pred_lat - true_lat) * p
    t2 = true_lat * p
    t3 = pred_lat * p
    t4 = (pred_lon - true_lon) * p

    a = 0.5 - torch.cos(t1) / 2 + torch.cos(t2) * torch.cos(t3) * (1 - torch.cos(t4)) / 2
    haversine_loss = (12742 * torch.asin(torch.sqrt(a))).mean()

    # Add penalty for lat values out of range [-90, 90]
    transformed_lat = pred_lat  # Apply your inverse transformation here if needed
    lat_penalty = (torch.clamp(transformed_lat, min=-90, max=90) - transformed_lat).abs().mean()


    total_loss = haversine_loss / 1000.0 + lat_penalty

    return total_loss
