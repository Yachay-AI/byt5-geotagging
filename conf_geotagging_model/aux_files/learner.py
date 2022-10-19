import numpy as np
import sys
import json
import torch
from tqdm.auto import tqdm
import math
from types import SimpleNamespace
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from torch.utils.data import Subset, DataLoader
from transformers import BertTokenizer, ByT5Tokenizer, get_scheduler

import aux_files.args_parser as args_parser
from aux_files.data_loading import truncate_dataset, pad_chars
from aux_files.models import CompositeModel


class Learner():
    '''
    Wrapper for current solution of geotagging with confidential scores
        
        methods:
            __init__(args): prepare dataloaders and the model
            
            get_model_by_arch(model_arch):
                select the model by `model_arch` name 
                (if you need to change the model after initialisation of Learner)

            train(): train the model (all parameters sutup in `self.args`)

            train_step(batch, optimizer, scheduler, criterion, current_epoch, metric):
                train process one batch with selected parameters

            val_step(self, batch, criterion, metric):
                validation process one batch with selected loss criterion and metric

            conf_estim_eval_handler(self, distances, confidences):
                Get metrics by confidences from `self.args.confidence_bands`
    '''
    def __init__(self, args=None):
        '''
        Initial arguments, prepare datasets, dataloaders and model.
        You can look at all arguments in `args_parser.default_args`
        '''
        self.args = args_parser.default_args
        if args is not None:
            self.args = SimpleNamespace(** {**self.args, **args})
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # DATA
        # dataset loading
        TweetDataset = args_parser.get_dataset(self.args.dataset_name)
        tweet_dataset = TweetDataset(data_dir=self.args.data_dir, subsample=self.args.subsample)
        
        # splitting
        if self.args.split_uids:
            gss = GroupShuffleSplit(n_splits=1, train_size=0.9)
            train_indices, val_indices = next(gss.split(np.arange(len(tweet_dataset)), groups=tweet_dataset.uids))
        else:
            ss = ShuffleSplit(n_splits=1, train_size=0.9)
            train_indices, val_indices = next(ss.split(np.arange(len(tweet_dataset))))

        self.train_dataset = Subset(tweet_dataset, train_indices)
        self.val_dataset = Subset(tweet_dataset, val_indices)

        # truncating
        if self.args.truncate_ratio:
            self.train_dataset = truncate_dataset(self.train_dataset, ratio=self.args.truncate_ratio)
            self.val_dataset = truncate_dataset(self.val_dataset, ratio=self.args.truncate_ratio)
        
        # tokenizing
        byte_tokenizer = ByT5Tokenizer.from_pretrained('google/byt5-small')
        word_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', model_max_length=512)
        tokenizers = (byte_tokenizer, word_tokenizer)

        # combining together tweets with different sizes
        collate_fn = lambda instance: pad_chars(instance, tokenizers, self.args.max_seq_len)

        # dataloaders
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=int(self.args.batch_size), collate_fn=collate_fn, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=int(self.args.batch_size), collate_fn=collate_fn, shuffle=True)

        # select the model architecture
        self.set_model_by_arch(self.args.arch)

        self.min_conf, self.max_conf = 1, 1


    def set_model_by_arch(self, model_arch):
        ''' 
        Select the model by architecture name 
        (if you need to change model before training)
        '''
        self.args.arch = model_arch
        self.model = CompositeModel(self.args)
        self.model.to(self.device)

    def train(self):
        ''' Train the model. Setup parameters during initialisation or in field `self.args` '''
        # The band by which we will calculate the performance to save the model
        if self.args.model_save_band not in self.args.confidence_bands:
            self.args.confidence_bands.append(self.args.model_save_band)
        
        
        

        num_training_steps = self.args.num_epochs * len(self.train_dataloader)
        criterion = args_parser.get_criterion(self.args.loss)
        optimizer = args_parser.get_optimizer(self.args.optimizer)(self.model.parameters(), lr=self.args.lr)
        scheduler = get_scheduler(self.args.scheduler, optimizer, 
                              num_warmup_steps=num_training_steps * self.args.warmup_ratio,
                              num_training_steps=num_training_steps)

        metric = args_parser.get_metric(self.args.metric)
        best_mean = 999999
        
        for epoch in tqdm(range(self.args.num_epochs)):
            self.model.train()
            train_iter = tqdm(self.train_dataloader)
            for batch in enumerate(train_iter):
                train_loss = self.train_step(batch, optimizer, scheduler, criterion, epoch, metric)
                train_iter.set_description(f"train loss: {train_loss.item()}")

            with torch.no_grad():
                self.model.eval()
                distances = []
                confidences = []
                val_iter = tqdm(self.val_dataloader)
                for batch in val_iter:
                    eval_stats = self.val_step(batch, criterion, metric)
                    val_loss, val_distance = eval_stats
                    if self.args.conf_estim:
                        val_distance, val_confidence = val_distance
                        confidences.extend(val_confidence.squeeze().tolist())

                    distances.extend(val_distance.tolist())
                    val_iter.set_description(f"val loss: {val_loss.item()}")

                self.min_conf = np.min(confidences)
                self.max_conf = np.max(confidences)
                
                # log
                val_mean = np.nan_to_num(np.mean(distances))
                val_median = np.nan_to_num(np.median(distances))
                print(f"val_mean_distance: {val_mean}")
                print(f"val_median_distance: {val_median}")

                # calculate distances by bands
                if self.args.conf_estim:  # estimate by confidence
                    scores = self.conf_estim_eval_handler(distances, confidences)
                    if self.args.confidence_validation_criterion: # model criterion by `model_save_band`
                        val_mean, val_median = scores[self.args.model_save_band]
                        print(f"\nusing top {self.args.model_save_band}% - val mean: {val_mean}")
                        print(f"using top {self.args.model_save_band}% - val median: {val_median}")

                is_best = val_mean < best_mean
                best_mean = min(val_mean, best_mean)

                if is_best:
                    print(f"\nsaving {self.args.save_prefix}.pt @ epoch {epoch}; mean distance: {val_mean:.6f}; median distance: {val_median:.6f}")
                    self.save_model()
                    
    def save_model(self):
        ''' Save model by '{self.args.prefix}.pt' path '''
        save_args = {
            'state_dict': self.model.state_dict(),
            'training_args': self.args,
            'model_args': {
                'arch': self.args.arch,
                'dropout': self.args.dropout,
                'reduce_layer': self.args.reduce_layer,
                'conf_estim': self.args.conf_estim
            }
        }
        if self.args.conf_estim:
            save_args['model_args']['min_conf'] = self.min_conf
            save_args['model_args']['max_conf'] = self.max_conf
        torch.save(save_args, self.args.save_prefix + '.pt')

    def train_step(self, batch, optimizer, scheduler, criterion, current_epoch, metric):
        ''' Train step by one batch '''
        step, batch = batch
        encoded_tokens, coords = batch
        encoded_tokens = [i.to(self.device) for i in encoded_tokens]
        coords = coords.to(self.device)

        byte_tokens, word_tokens = encoded_tokens

        if self.args.conf_estim:
            coord_pred, conf_pred = self.model(byte_tokens, word_tokens)
            indices = torch.argsort(metric(coord_pred, coords))
            conf_pred = torch.gather(conf_pred.squeeze(), 0, indices).unsqueeze(dim=-1)
            conf_target = torch.cat([torch.ones(math.floor(conf_pred.size(0) * self.args.conf_p)),
                                    torch.zeros(math.ceil(conf_pred.size(0) * (1 - self.args.conf_p)))]).unsqueeze(dim=-1).to(self.device)
            if math.ceil(self.args.no_conf_epochs * self.args.num_epochs) >= current_epoch:
                loss = criterion(coord_pred, coords) + (self.args.conf_lambda * criterion(conf_pred, conf_target))
            else:
                loss = criterion(coord_pred, coords)

        else:
            pred = self.model(byte_tokens, word_tokens)
            loss = criterion(pred, coords)

        loss.backward()
        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        return loss

    def val_step(self, batch, criterion, metric):
        ''' Validation step by one batch '''
        encoded_tokens, coords = batch

        encoded_tokens = [i.to(self.device) for i in encoded_tokens]
        coords = coords.to(self.device)

        byte_tokens, word_tokens = encoded_tokens

        # check if batch dim squeezed out during pred, fix
        if self.args.conf_estim:
            pred, confidence = self.model(byte_tokens, word_tokens)
            pred = pred.unsqueeze(0) if len(pred.size()) == 1 else pred
            distance = (metric(coords, pred), confidence)
        else:
            pred = self.model(byte_tokens, word_tokens)
            pred = pred.unsqueeze(0) if len(pred.size()) == 1 else pred
            distance = metric(coords, pred)

        loss = criterion(pred, coords)

        return loss, distance

    def conf_estim_eval_handler(self, distances, confidences):
        ''' Get scores by confidences '''
        scores = {}
        distances = np.array(distances)
        confidences = np.array(confidences)
        indices = np.argsort(confidences)[::-1]
        assert confidences.max() == confidences[indices[0]]
        for i in self.args.confidence_bands:
            numel = (indices.size * i) // 100
            band = np.take_along_axis(distances, indices[:numel], axis=0)
            val_mean = np.nan_to_num(np.mean(band))
            val_median = np.nan_to_num(np.median(band))
            scores[i] = (val_mean, val_median)
            print(f"top {i}% - test mean: {val_mean}")
            print(f"top {i}% - test median: {val_median}")
        
        return scores
