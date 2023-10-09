import argparse
import logging.handlers
import os
import math
import os.path
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.neighbors import BallTree
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

import wandb
from model import ByT5ClusteredClassifierDataset, ByT5_classifier
from utils import read_csv_data, \
    read_train_test_data, \
    true_distance_from_pred, true_distance_from_pred_cluster
    
def validate_rel(epoch, model, valid_generator, logger, show_progress=False, no_save=False, smooth_labels=False):
    """
    Validates a PyTorch neural network model on a validation dataset.

    Parameters:
    -----------
    epoch : int
        The current epoch number of the training process.
    model : torch.nn.Module
        The neural network model to be validated.
    valid_generator : torch.utils.data.DataLoader
        The data loader for the validation dataset.
    logger : logging.Logger
        The logger object to record the validation metrics.
    show_progress : bool, optional
        Whether to display the validation progress bar. Default is False.
    no_save : bool, optional
        Whether to skip saving the validation results. Default is False.
    smooth_labels : bool, optional
        Are ground truth labels smoothed or not. Default is False.

    Returns:
    --------
    true_lats : list of float
        The true latitudes of the validation dataset.
    true_lngs : list of float
        The true longitudes of the validation dataset.
    true_clusters : list of int
        The true cluster IDs of the validation dataset.
    pred_clusters : list of int
        The predicted cluster IDs of the validation dataset.
    texts : list of str
        The input texts of the validation dataset.
    """

    model.eval()
    loss_ls = []
    texts = []
    true_distance_ls = []
    true_clusters = []
    pred_clusters = []
    true_lats = []
    true_lngs = []
    if show_progress:
        generator = tqdm(valid_generator)
    else:
        generator = valid_generator
    tr = 0
    for batch in generator:
        te_feature, te_label_true, text, te_lat, te_lng, te_langs, int_label = batch
        te_feature = te_feature.to(device)
        te_label_true = te_label_true.to(device)
        te_langs = te_langs.to(device)
        with torch.no_grad():
            te_predictions = model(te_feature, te_langs)
        if smooth_labels and args.loss in ["kl", "nll"]:
            te_loss = custom_loss(F.log_softmax(te_predictions), te_label_true)
        else:
            te_loss = custom_loss(te_predictions, te_label_true)
        texts.append(text)
        loss_ls.append(te_loss.item())
        true_lats.append(te_lat.detach().cpu())
        true_lngs.append(te_lng.detach().cpu())
        if smooth_labels:
            te_label_true_index = torch.argmax(te_label_true, dim=1)
        else:
            te_label_true_index = te_label_true
        true_clusters.append(te_label_true_index.detach().cpu())
        pred_clusters.append(te_predictions.detach().cpu())
        true_distance_ls.append(
            true_distance_from_pred(te_predictions.detach().cpu(), te_lat.detach().cpu(), te_lng.detach().cpu(),
                                    cluster_df))
        loss_ls.append(te_loss.item())
        if len(loss_ls) % 500 == 0:
            tr += 1
            test_results_filename = f'test_results_partial-{tr}.pkl'
            with open(test_results_filename, 'wb') as fout:
                pickle.dump(torch.cat(pred_clusters, 0).numpy(), fout)
            try:
                os.remove(f'test_results_partial-{tr-1}.pkl')
            except:
                pass
    te_loss = sum(loss_ls) / len(loss_ls)
    true_distance_ls = torch.cat(true_distance_ls, 0)
    true_distance_ls = pd.Series(true_distance_ls.numpy())
    true_lats = torch.cat(true_lats, 0).numpy()
    true_lngs = torch.cat(true_lngs, 0).numpy()
    true_clusters = torch.cat(true_clusters, 0).numpy()
    pred_clusters = torch.cat(pred_clusters, 0).numpy()
    texts = [item for sublist in texts for item in sublist]
    pred_cluster_ids = [pred_clusters[i].argmax() for i in range(len(pred_clusters))]
    acc = len(true_clusters[true_clusters == pred_cluster_ids]) / len(true_clusters)
    logger.info(
        f'Epoch {epoch} eval loss {te_loss}  accuracy {acc} ' +
        f'true distance avg {true_distance_ls.mean()} true distance median {true_distance_ls.median()}')

    if not no_save:
        model_name = "byt5"
        filename = f"models/{model_name}-class-%d" % epoch
        torch.save(model, filename)
        logger.info(f"saved to {filename}")
    if args.wandb_project:
        wandb.log({'eval_loss': te_loss, 'eval_accuracy': acc, 'distance_median': true_distance_ls.median(),
                   'distance_mean': true_distance_ls.mean(),
                   'distinct_true_clusters': len(pd.Series(true_clusters).unique()),
                   'distinct_pred_clusters': len(pd.Series(pred_cluster_ids).unique())})
    return true_lats, true_lngs, true_clusters, pred_clusters, texts


def train_epoch_rel(epoch, model, training_generator, valid_generator, optimizer, args, logger, smooth_labels=False):
    """
    Trains the model for one epoch on a training dataset, and runs validation after the epoch.

    Parameters:
    -----------
    epoch : int
        The current epoch number of the training process.
    model : torch.nn.Module
        The neural network model to be trained.
    training_generator : torch.utils.data.DataLoader
        The data loader for the training dataset.
    valid_generator : torch.utils.data.DataLoader
        The data loader for the validation dataset.
    optimizer : torch.optim.Optimizer
        The optimization algorithm used to update the model parameters.
    args : argparse.Namespace
        The command-line arguments used to configure the training process.
    logger : logging.Logger
        The logger object to record the training metrics.
    smooth_labels : bool, optional
        Are ground truth labels smoothed or not. Default is False.

   
    """

    model.train()
    losses = []
    for iter, batch in tqdm(enumerate(training_generator), total=len(training_generator)):
        model.train()
        feature, label_true, _, lat, lng, langs, int_label = batch
        feature = feature.to(device)
        label_true = label_true.to(device)
        langs = langs.to(device)
        optimizer.zero_grad()
        predictions = model(feature, langs)
        if smooth_labels and args.loss in ["kl", "nll"]:
            loss = custom_loss(F.log_softmax(predictions), label_true)
        else:
            loss = custom_loss(predictions, label_true)
        losses.append(loss.item())
        loss.backward()
        if args.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
        optimizer.step()
        if iter % args.log_steps == 0:
            logger.info(f'Epoch {epoch} training loss {sum(losses) / len(losses)}')
            if args.wandb_project:
                wandb.log({'training_loss': sum(losses) / len(losses)})
            losses = []
        if iter % args.eval_steps == 0 or iter == len(training_generator) - 1:
            validate_rel(epoch=epoch, model=model, valid_generator=valid_generator, logger=logger,
                         smooth_labels=smooth_labels)

    # training loss display
    logger.info(f'Epoch {epoch} training loss {sum(losses) / len(losses)}')


def show_metrics(true_lats, true_lngs, true_clusters, pred_clusters, min_distance, logger,
                 all_thresholds=False, smooth_labels=False):
    """
    Computes and logs distance based evaluation metrics for the model.

    Parameters:
    -----------
    true_lats : list of float
        The true latitudes of the validation dataset.
    true_lngs : list of float
        The true longitudes of the validation dataset.
    true_clusters : list of int
        The true cluster IDs of the validation dataset.
    pred_clusters : list of int
        The predicted cluster IDs of the validation dataset.
    min_distance : float
        The minimum distance threshold to use for computing evaluation metrics.
    logger : logging.Logger
        The logger object to record the evaluation metrics.
    all_thresholds : bool, optional
        Whether to compute evaluation metrics for all distance thresholds. Default is False (compute only for 0 and 0.75).
    smooth_labels : bool, optional
        Are ground truth labels smoothed or not. Default is False.

    Returns:
    --------
    None
        The function logs the evaluation metrics using the provided logger object.

    """
    MAEs = []
    Medians = []
    F1s = []
    percentages = []
    thresholds = [j / 20.0 for j in range(0, 20)] if all_thresholds else [0, 0.75]
    for threshold in thresholds:
        part_true_distance_ls = []
        acc = []
        vals = []
        tp = 0
        fp = 0
        fn = 0
        for i in range(true_clusters.shape[0]):
            pred_cluster_proba = torch.nn.Softmax(dim=0)(torch.tensor(pred_clusters[i])).numpy()
            pred = pred_clusters[i].argmax()
            pred_val = pred_cluster_proba[pred]
            dist = true_distance_from_pred_cluster(pred, true_lats[i], true_lngs[i], cluster_df)
            if dist < min_distance and pred_val >= threshold:
                tp += 1
            elif dist >= min_distance and pred_val >= threshold:
                fp += 1
            elif dist < min_distance and pred_val < threshold:
                fn += 1
            if pred_val < threshold:
                continue
            vals.append(pred_val)
            if smooth_labels:
                acc.append(1 if pred == true_clusters[i].argmax() else 0)
            else:
                acc.append(1 if pred == true_clusters[i] else 0)
            part_true_distance_ls.append(dist)
        part_true_distance_ls = pd.Series(part_true_distance_ls)
        MAEs.append(part_true_distance_ls.mean())
        Medians.append(part_true_distance_ls.median())
        percentages.append(len(part_true_distance_ls) / true_clusters.shape[0])
        if tp > 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = tp / (tp + 0.5 * (fp + fn))
        else:
            precision = 0
            recall = 0
            f1 = 0
        F1s.append(f1)
        logger.info(f'threshold {threshold} MAE {part_true_distance_ls.mean()} ' +
                    f'Median {part_true_distance_ls.median()} ' +
                    f'percentage {len(part_true_distance_ls) / true_clusters.shape[0]} ' +
                    f'acc {pd.Series(acc).mean()} ' +
                    f'precision {precision} recall {recall} ' +
                    f'f1@{min_distance} {f1}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_test_input_file', type=str, help='Source csv file')
    parser.add_argument('--train_input_file', type=str, help='Source csv file')
    parser.add_argument('--test_input_file', type=str, help='Source csv file for test')
    parser.add_argument('--max_test', type=int, help='Limit number of testing samples')
    parser.add_argument('--do_train', type=bool)
    parser.add_argument('--do_test', type=bool)
    parser.add_argument('--load_model_dir', type=str, help='Load model from dir and continue training')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--max_length', type=int, default=140)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--load_clustering', type=str, help='Load cluster centers from directory')
    parser.add_argument('--max_train', type=int, help='Limit number of training samples')
    parser.add_argument('--train_skiprows', type=int, help='Skip first N training samples')
    parser.add_argument('--test_skiprows', type=int, help='Skip first N test samples')
    parser.add_argument('--random_state', type=int, default=300)
    parser.add_argument('--eval_batches', type=int, default=32, help='Number of batches for evaluation')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_steps', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--min_distance', type=int, default=500)
    parser.add_argument('--byt5_model_name', type=str, default='google/byt5-small')
    parser.add_argument('--wandb_project', type=str)
    parser.add_argument('--use-language', type=str)
    parser.add_argument('--all_thresholds', type=bool, default=False)
    parser.add_argument('--label_smoothing', type=float, default=0)
    parser.add_argument('--country_by_coordinates', type=str)
    parser.add_argument('--lang_home_country', type=str)
    parser.add_argument('--external_factor', type=float, default=0.0)
    parser.add_argument('--rare_language_factor', type=float, default=0.0)
    parser.add_argument('--rare_cluster_factor', type=float, default=0.0)
    parser.add_argument('--weight_max', type=float, default=16.0)
    parser.add_argument('--weight_stats', type=str)
    parser.add_argument('--distance_based_smoothing', type=float, default=0.0)
    parser.add_argument('--keep_layer_count', type=int)
    parser.add_argument('--lr_epoch', type=str)
    parser.add_argument('--smooth_labels', type=bool, default=False)
    parser.add_argument('--loss', type=str)
    parser.add_argument('--nearest_count', type=int)
    parser.add_argument('--nearest_weight', type=float, default=0.0)
    parser.add_argument('--author_weight', type=float, default=0.0)
    parser.add_argument('--nearest_distance', type=float)
    parser.add_argument('--nearest_weight_epoch_increase', type=float, default=0.0)
    parser.add_argument('--gradient_clipping', type=float)

    smooth_labels = False
    smooth_labels_train = None
    smooth_labels_test = None

    args = parser.parse_args()

    Path('logs').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)

    logger = logging.getLogger("")
    logging.basicConfig(level=logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler("logs/debug.log"))

    logger.info("start")

    if args.wandb_project:
        wandb.init(project=args.wandb_project)

    df_train = None

    distance_between_clusters = None
    if args.load_clustering is not None:
        with open(args.load_clustering + 'clustering.pkl', 'rb') as fin:
            cluster_df, merges = pickle.load(fin)
        if os.path.exists(args.load_clustering + 'distance_between_clusters.pkl'):
            with open(args.load_clustering + 'distance_between_clusters.pkl', 'rb') as fin:
                distance_between_clusters = pickle.load(fin)
                logger.info('loaded distance_between_clusters')
        tree = BallTree(np.deg2rad(cluster_df[['lat', 'lng']].values), metric='haversine')

    if args.train_test_input_file is not None:
        df_train, df_test = read_train_test_data(args.train_test_input_file)
        logger.info("finish reading train, test file")

    if args.test_input_file is not None:
        df_test = read_csv_data(args.test_input_file, nrows=args.max_test, skiprows=(
            lambda x: x > 0 and x < args.test_skiprows) if args.test_skiprows is not None else None)
        logger.info("finish reading test file")
        if args.smooth_labels:
            smooth_labels_test = calculate_smooth_labels(df_test, cluster_df, tree, args.nearest_count,
                                                         args.nearest_weight, args.author_weight,
                                                          args.nearest_distance)
            print('calculated smooth_labels_test')
            smooth_labels = True
            with open('smooth_labels_test.pkl', 'wb') as fout:
                pickle.dump(smooth_labels_test, fout)

    if args.train_input_file is not None:
        df_train = read_csv_data(args.train_input_file, nrows=args.max_train, skiprows=(
            lambda x: x > 0 and x < args.train_skiprows) if args.train_skiprows is not None else None)
        logger.info("finish reading train file")
        logger.info(df_train.columns)
        if args.smooth_labels:
            smooth_labels_train = calculate_smooth_labels(df_train, cluster_df, tree, args.nearest_count,
                                                          args.nearest_weight, args.author_weight,
                                                          args.nearest_distance)
            print('calculated smooth_labels_train')
            smooth_labels = True
            with open('smooth_labels_train.pkl', 'wb') as fout:
                pickle.dump(smooth_labels_train, fout)

    language_df = None
    if args.use_language is not None:
        language_df = pd.read_csv(args.use_language)
        logger.info("language_df", len(language_df))

    max_length = args.max_length
    device = args.device
    n_clusters_ = len(cluster_df)

    # calculate weights for sampling
    if args.country_by_coordinates is not None and args.lang_home_country is not None:
        logger.info("calculating weights")
        with open(args.country_by_coordinates, 'rb') as fin:
            country_by_coordinates = pickle.load(fin)
        with open(args.lang_home_country, 'rb') as fin:
            lang_home_country = pickle.load(fin)

        if not smooth_labels and not 'label' in df_train.columns:
            labels = []
            for i, row in tqdm(df_train.iterrows(), total=len(df_train)):
                coords = [[np.deg2rad(row['lat']), np.deg2rad(row['lon'])]]
                labels.append(tree.query(coords, k=1)[1][0][0])
            logger.info("labels done")
            df_train['label'] = labels

        if 'weight' in df_train.columns:
            sampler = WeightedRandomSampler(torch.tensor(df_train['weight'].round(decimals=1).values), len(df_train))
            logger.info('initialize sampler')
        else:
            if not 'country' in df_train.columns:
                df_train['country'] = [country_by_coordinates[c] if c in country_by_coordinates else '' for c in
                                       df_train['coordinates']]
                logger.info("countries done")
            if not 'is_external' in df_train.columns:
                df_train['is_external'] = [
                    0 if (df_train.iloc[i]['country'] in (lang_home_country[df_train.iloc[i]['lang']])) else 1 for i in
                    range(len(df_train))]
                logger.info("is_external done")

            lang_weights = {}
            top_language_count = df_train['lang'].value_counts().max()
            for lang in lang_home_country.keys():
                if len(df_train[df_train['lang'] == lang]) > 0:
                    lang_weights[lang] = 1.0 + math.log(
                        top_language_count / len(df_train[df_train['lang'] == lang])) * args.rare_language_factor
            logger.info("lang_weights", lang_weights)
            cluster_weights = {}
            top_cluster_count = df_train['label'].value_counts().max()
            for cluster_id, count in df_train['label'].value_counts().iteritems():
                cluster_weights[cluster_id] = 1.0 + math.log(top_cluster_count / count) * args.rare_cluster_factor
            logger.info("cluster_weights", cluster_weights)
            weights = []
            for i in range(len(df_train)):
                row = df_train.iloc[i]
                w = 1.0 \
                    * (1 + (args.external_factor if row['is_external'] else 0)) * lang_weights[row['lang']] * \
                    cluster_weights[row['label']]
                if w > args.weight_max:
                    w = args.weight_max
                weights.append(w)
            sampler = WeightedRandomSampler(torch.tensor(weights), len(weights))
            df_train['weight'] = weights
            logger.info("number of weights", len(weights), "max", max(weights), "min", min(weights))
        # df_train.to_csv('df_train_weights.csv')
    else:
        sampler = None

    training_params = {"batch_size": args.batch_size,
                       "shuffle": True if sampler is None else False,
                       "num_workers": 0, "sampler": sampler}
    test_params = {"batch_size": args.batch_size,
                   "shuffle": False,
                   "num_workers": 0}
    if args.do_train:
        training_set = ByT5ClusteredClassifierDataset(df_train, args.byt5_model_name, tree, max_length, None,
                                                      smooth_labels_train)
       

    full_test_set = ByT5ClusteredClassifierDataset(df_test, args.byt5_model_name, tree, max_length, None,
                                                   smooth_labels_test)

    test_set = torch.utils.data.Subset(full_test_set,
                                       random.choices(range(0, len(df_test)),
                                                      k=args.eval_batches * test_params['batch_size']))
    random.seed(args.random_state)
    valid_set = torch.utils.data.Subset(full_test_set,
                                        random.choices(range(0, len(df_test)), k=test_params['batch_size']))

    if args.load_model_dir is not None:
        model = torch.load(args.load_model_dir, map_location=torch.device(device))
        logger.info("model loaded")
    else:  # train from pre-trained
        model = ByT5_classifier(n_clusters=len(cluster_df), model_name=args.byt5_model_name,
                                keep_layer_count=args.keep_layer_count)
        logger.info(model)

    if torch.cuda.is_available():
        model.to(device)

    if args.wandb_project:
        wandb.watch(model, log_freq=100)

    test_generator = DataLoader(test_set, **test_params)
    valid_generator = DataLoader(valid_set, **test_params)
    full_test_generator = DataLoader(full_test_set, **test_params)

    if smooth_labels:
        if args.loss == "bce":
            custom_loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
        elif args.loss == "nll":
            custom_loss = torch.nn.NLLLoss(reduction='mean')
        elif args.loss == "ce":
            custom_loss = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction='mean')
        elif args.loss == "kl":
            custom_loss = torch.nn.KLDivLoss(reduction="batchmean")
        else:
            raise NotImplementedError(f"Unknown loss: {args.loss}")
    elif args.distance_based_smoothing == 0.0:
        custom_loss = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, reduction='mean')
    else:
        custom_loss = DistanceBasedLoss(label_smoothing=args.distance_based_smoothing,
                                        distance_between_clusters=distance_between_clusters)

    optimizer = torch.optim.Adam(
        [{'params': model.fc3.parameters(), 'lr': 1e-3}, {'params': model.byt5.parameters(), 'lr': 1e-4}])

    if args.do_train:
        training_generator = DataLoader(training_set, **training_params)
        model.train()

        lr_epoch = [1e-3, 1e-4, 1e-5]
        if smooth_labels:
            lr_epoch = [1e-3, 1e-3, 1e-4]
        if args.lr_epoch is not None:
            lr_epoch = [float(x) for x in args.lr_epoch.split(",")]
            logger.info(f"using lr_epoch {lr_epoch}")

        num_epochs = args.num_epochs
        for epoch in range(args.start_epoch, num_epochs):
            if args.nearest_weight_epoch_increase > 0.0:
                smooth_labels_train = calculate_smooth_labels(df_train, cluster_df, tree, args.nearest_count,
                                                              args.nearest_weight + epoch * args.nearest_weight_epoch_increase,
                                                              args.author_weight,
                                                              args.nearest_distance)
                logger.info(
                    f'recalculated smooth_labels_train with nearest_weight {args.nearest_weight + epoch * args.nearest_weight_epoch_increase}')

            if epoch < len(lr_epoch):
                logger.info(f"setting learning rate to {lr_epoch[epoch]}")
                optimizer.param_groups[0]['lr'] = lr_epoch[epoch]
                for g in optimizer.param_groups:
                    g['lr'] = lr_epoch[epoch]
                    # g['lr'] = min(g['lr'], lr_epoch[epoch])
            train_epoch_rel(epoch=epoch, model=model, training_generator=training_generator,
                            valid_generator=test_generator, optimizer=optimizer, args=args, logger=logger,
                            smooth_labels=smooth_labels)

    if args.do_test:
        true_lats, true_lngs, true_clusters, pred_clusters, texts = validate_rel(epoch=args.num_epochs, model=model,
                                                                                 valid_generator=full_test_generator,
                                                                                 logger=logger, show_progress=True,
                                                                                 no_save=True,
                                                                                 smooth_labels=smooth_labels)
        test_results_filename = 'test_results.pkl'
        with open(test_results_filename, 'wb') as fout:
            pickle.dump((true_lats, true_lngs, true_clusters, pred_clusters, texts), fout)
        show_metrics(true_lats, true_lngs, true_clusters, pred_clusters, args.min_distance, logger,
                     all_thresholds=args.all_thresholds, smooth_labels=smooth_labels)
