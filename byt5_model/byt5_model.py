import copy
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from torch.utils.data import Dataset
from transformers import T5EncoderModel, T5Config, T5Model
from transformers import AutoTokenizer

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
        hidden_size = self.byt5.config.d_model

        if language_count is not None:
            language_embedding_dim = language_count // 4
            self.language_embedding = nn.Embedding(language_count, language_embedding_dim)
            hidden_size += language_embedding_dim

        self.fc3 = nn.Linear(hidden_size, n_clusters)

    def forward(self, input, input_language=None):
        input = self.byt5(input[:, 0, :].squeeze(1))['last_hidden_state']
        input = input[:, 0, :].squeeze(1)
        return self.fc3(input)



class ByT5ClusteredClassifierDataset(Dataset):
    def __init__(self, dataframe, scaler, tokenizer_name, tree, max_length=1014, merges=None, language_df=None):
        self.dataframe = dataframe
        self.max_length = max_length
        self.length = len(self.dataframe)
        self.tree = tree
        self.scaler = scaler
        self.merges = merges
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.language_list = language_df['lang'].values.tolist() if language_df is not None else None

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raw_text = self.dataframe['text'].iloc[index]
        data = self.tokenizer(raw_text, truncation=True, padding="max_length", max_length=self.max_length,
                              return_tensors='pt')['input_ids']
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

        return data, label, raw_text, self.dataframe.iloc[index]['lat'], \
            self.dataframe.iloc[index]['lon'], language_id

