import os
import math
import json
from tqdm.auto import tqdm
import random
import torch
from torch.utils.data import Dataset


def pad_chars(instance, tokenizers, max_length=-1):
    '''
    Padding texts to the same size

        arguments:
            instance (list, list): tuple of [list of tokens, list of [lon, lat]]
            tokenizers (BertTokenizer, ByT5Tokenizer): tokenizers for char and word tokenization

        return:
            (byte_tokens, word_tokens), coords: tokenized by bytes and words tokens the same size and coords
    '''
    try:
        tokens, coords = zip(*instance)
        encoded_coords = torch.stack(coords)
    except ValueError:
        tokens = instance
        encoded_coords = None

    byte_tokenizer, word_tokenizer = tokenizers
    word_tokens = word_tokenizer(tokens, padding=True, return_tensors='pt', truncation=True)

    def tokenize_maybe_pad(tokenizer, tokens, length=7):
        tokenized = tokenizer(tokens, padding=True, return_tensors='pt')
        if tokenized.input_ids.size(1) < length:
            tokenized = tokenizer(tokens, padding='max_length', max_length=length, return_tensors='pt')
        return tokenized

    if max_length == -1:
        byte_tokens = tokenize_maybe_pad(byte_tokenizer, tokens)
    else:
        byte_tokens = byte_tokenizer(tokens, truncation=True, padding='max_length', max_length=max_length,
                                     return_tensors='pt')

    encoded_tokens = (byte_tokens, word_tokens)
    return encoded_tokens, encoded_coords


def truncate_dataset(dataset, ratio):
    ''' Truncate the dataset, leaving only `ration` of the data '''
    count_to_keep = int(math.ceil(len(dataset) * float(ratio)))
    dataset_list = random.sample(range(len(dataset)), count_to_keep)
    return torch.utils.data.Subset(dataset, dataset_list)


class AllTweets2021Dataset(Dataset):
    ''' Loading the data from `all_tweets_2021` dataset '''

    def __init__(self: Dataset, data_dir, subsample=9e9):
        '''
        Prepare the data
        
            arguments:
                data_dir (str): path to folder with data
                subsample (float): max count of tweets per distinct location
        '''
        self.tweet_tokens = []
        self.uids = []
        self.coords = []

        for fname in tqdm(os.listdir(f"{data_dir}")):
            counter = 0
            with open(f"{data_dir}/{fname}") as f:
                for line in f:
                    if counter > subsample:
                        break
                    try:
                        d = json.loads(line)
                    except Exception as error:
                        print('error loading JSON: ' + repr(error))
                        print('file name: ' + str(fname))

                    self.tweet_tokens.append(str(d['text']))
                    self.uids.append(int(d['author_id']))
                    self.coords.append(tuple(map(float, fname.rstrip('.json').split('_'))))
                    counter += 1

    def __len__(self: Dataset) -> int:
        assert len(self.tweet_tokens) == len(self.coords)
        return len(self.tweet_tokens)

    def __getitem__(self: Dataset, idx: int):
        ''' Return: tokens(str), labels([float, float]) as [lon, lat] '''
        tokens = self.tweet_tokens[idx]
        labels = torch.FloatTensor(self.coords[idx])
        return tokens, labels