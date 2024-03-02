import argparse
import logging.handlers
import math
import os.path
import pickle
import random
from pathlib import Path
import wandb

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.neighbors import BallTree
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from model import ByT5ClusteredClassifierDataset, ByT5_classifierNew, DistanceBasedLoss, calculate_smooth_labels, \
    ModifiedCharCNN, MultiLayerCharLSTM, deleteEncodingLayers, deleteEncodingLayersDeberta, ByT5_regressorNew, haversine_loss_with_penalty
from transformers import AutoModelForSequenceClassification

# Util imports
from utils import read_csv_data, read_train_test_data, true_distance_from_pred, true_distance_from_pred_cluster, true_distance_from_coords

def validate_rel(epoch, model, valid_generator, logger, show_progress=False, no_save=False, smooth_labels=False,
                 steps=0, smooth_labels_test=False):
    """
    Validates the model on a validation dataset and computes evaluation metrics.
    
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
    steps : int, optional
        Current step number in the training process.
    smooth_labels_test : bool, optional
        Whether the validation data has smooth labels. Default is False.

    Returns:
    --------
    tuple
        A tuple containing true latitudes, longitudes, clusters, predicted clusters, input texts, and confidence scores.
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
    for batch in generator:
        te_feature, te_label_true, text, te_lat, te_lng, te_langs, int_label, confidence_weight, te_intermediate_cluster, te_mask = batch
        te_feature = te_feature.to(device)
        te_label_true = te_label_true.to(device)
        te_intermediate_cluster = te_intermediate_cluster.to(device)
        te_langs = te_langs.to(device)
        te_mask = te_mask.to(device)
        with torch.no_grad():
            te_predictions = model(te_feature, te_mask)
        if args.model_type == 'bert':
            te_predictions = te_predictions.logits
        if args.regression:
            pred_lats = te_predictions[:, 0]
            pred_lons = te_predictions[:, 1]
            te_loss = haversine_loss_with_penalty(te_lat.to(device), te_lng.to(device), pred_lats, pred_lons, device)
        elif args.intermediate_clustering is not None:
            te_loss = custom_loss(te_predictions[0], te_label_true) + custom_loss(te_predictions[1], te_intermediate_cluster)
            te_predictions = te_predictions[0]
        else:
            te_loss = custom_loss(te_predictions, te_label_true)
        texts.append(text)
        loss_ls.append(te_loss.item())
        true_lats.append(te_lat.detach().cpu())
        true_lngs.append(te_lng.detach().cpu())
        if smooth_labels and smooth_labels_test:
            te_label_true_index = torch.argmax(te_label_true, dim=1)
        else:
            te_label_true_index = te_label_true
        true_clusters.append(te_label_true_index.detach().cpu())
        if not args.regression:
            pred_clusters.append(te_predictions.detach().cpu())
            true_distance_ls.append(
                true_distance_from_pred(te_predictions.detach().cpu(), te_lat.detach().cpu(), te_lng.detach().cpu(),
                                        cluster_df))
        else:
            true_distance_ls.append(true_distance_from_coords(pred_lats, pred_lons, te_lat, te_lng))
        loss_ls.append(te_loss.item())
    te_loss = sum(loss_ls) / len(loss_ls)
    true_distance_ls = torch.cat(true_distance_ls, 0)
    true_distance_ls = pd.Series(true_distance_ls.numpy())
    true_lats = torch.cat(true_lats, 0).numpy()
    true_lngs = torch.cat(true_lngs, 0).numpy()
    true_clusters = torch.cat(true_clusters, 0).numpy()
    texts = [item for sublist in texts for item in sublist]
    if not args.regression:
        pred_clusters = torch.cat(pred_clusters, 0).numpy()
        pred_cluster_ids = [pred_clusters[i].argmax() for i in range(len(pred_clusters))]
        acc = len(true_clusters[true_clusters == pred_cluster_ids]) / len(true_clusters)
    else:
        acc = len(true_distance_ls[true_distance_ls < 161]) / len(true_distance_ls)
        pred_cluster_ids = 0
    try:
        confidence = [pred_clusters[i].max() for i in range(len(pred_clusters))]
        threshold = -1 * np.percentile(-1 * np.array(confidence), 10)
        top_confidence_distances = true_distance_ls[confidence >= threshold]
    except:
        confidence = []
        top_confidence_distances = pd.Series([])
    logger.info(
        f'Epoch {epoch} eval loss {te_loss}  accuracy {acc} ' +
        f'true distance avg {true_distance_ls.mean()} true distance median {true_distance_ls.median()}' +
        f'top 10 avg {top_confidence_distances.mean()}, top 10 median {top_confidence_distances.median()}' +
        f'top 10 count {len(top_confidence_distances)}, eval set count {len(true_distance_ls)}')
    # , 'Metrics', test_metrics
    if not no_save and steps % args.save_steps == 0:
        model_name = "byt5"
        filename = f"models/{model_name}-class-%d-%d" % (epoch, steps)
        torch.save(model, filename)
        logger.info(f"saved to {filename}")
    else:
        if not no_save:
            model_name = "byt5"
            filename = f"models/{model_name}-checkpoint"
            torch.save(model, filename)
            logger.info(f"saved to {filename}")
    if args.wandb_project:
        wandb.log({'eval_loss': te_loss, 'eval_accuracy': acc, 'distance_median': true_distance_ls.median(),
                   'distance_mean': true_distance_ls.mean(),
                   'distinct_true_clusters': len(pd.Series(true_clusters).unique()),
                   'distinct_pred_clusters': len(pd.Series(pred_cluster_ids).unique()),
                   '10_median': top_confidence_distances.median(),
                   '10_mean': top_confidence_distances.mean()})
    return true_lats, true_lngs, true_clusters, pred_clusters, texts, np.array(confidence)

def train_epoch_rel(epoch, model, training_generator, valid_generator, optimizer, args, logger, smooth_labels=False):
    """
    Trains the model for one epoch on a training dataset and runs validation after the epoch.

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

    Returns:
    --------
    None
    """
    model.train()
    losses = []
    main_losses = []
    intermediate_losses = []
    max_steps = len(training_generator)
    logger.info(f'training max_steps={max_steps}')
    for iter, batch in tqdm(enumerate(training_generator), total=max_steps):
        model.train()
        feature, label_true, _, lat, lng, langs, int_label, confidence_weight, intermediate_cluster, mask = batch
        if smooth_labels:
            label_true = torch.argmax(label_true, dim=1)
        if confidence_weight.max() > 1.0:
            label_true = make_smooth_labels_for_confidence_weight(label_true, confidence_weight,
                                                                  label_smoothing=args.label_smoothing,
                                                                  num_classes=len(cluster_df))
        feature = feature.to(device)
        label_true = label_true.to(device)
        intermediate_cluster = intermediate_cluster.to(device)
        langs = langs.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        predictions = model(feature, mask)
        if args.model_type == 'bert':
            predictions = predictions.logits
        if args.regression:
            pred_lats = predictions[:, 0]
            pred_lons = predictions[:, 1]
            loss = haversine_loss_with_penalty(lat.to(device), lng.to(device), pred_lats, pred_lons, device)
        elif args.intermediate_clustering is not None:
            main_loss = custom_loss(predictions[0], label_true)
            intermediate_loss = custom_loss(predictions[1], intermediate_cluster)
            loss = main_loss + intermediate_loss
            main_losses.append(main_loss.item())
            intermediate_losses.append(intermediate_loss.item())
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
                if len(main_losses) == 0:
                    main_losses = [0]
                    intermediate_losses = [0]
                wandb.log({'training_loss': sum(losses) / len(losses), 'main_loss': sum(main_losses) / len(main_losses),
                           'intermediate_loss': sum(intermediate_losses) / len(intermediate_losses)})
            losses = []
            main_losses = []
            intermediate_losses = []

        if iter % args.eval_steps == 0 or iter == max_steps - 1:
            validate_rel(epoch=epoch, model=model, valid_generator=valid_generator, logger=logger,
                         smooth_labels=smooth_labels, steps=iter, smooth_labels_test=smooth_labels_test)
        if iter == max_steps - 1:
            break

    # training loss display
    logger.info(f'Epoch {epoch} training loss {sum(losses) / len(losses)}')

def make_smooth_labels_for_confidence_weight(label_true, confidence_weight, label_smoothing, num_classes):
    """
    Modifies the true labels based on the confidence weight and label smoothing parameters.

    This function adjusts the true labels using label smoothing and confidence weights.
    It increases the label smoothing effect for samples with high confidence weights.

    Parameters:
    -----------
    label_true : torch.Tensor
        A tensor of true labels.
    confidence_weight : torch.Tensor
        A tensor representing the confidence weight for each sample.
    label_smoothing : float
        The label smoothing parameter.
    num_classes : int
        The number of classes in the classification task.

    Returns:
    --------
    torch.Tensor
        A tensor of modified true labels after applying label smoothing and confidence weight adjustments.
    """
    label_smoothing_increased = 1.0
    device = label_true.device
    label_true = label_true.to(device)
    label_true_one_hot = torch.zeros(label_true.shape[0], num_classes).to(device)
    label_true_one_hot.scatter_(1, label_true.view(-1, 1), 1)
    # Apply label smoothing to all labels
    label_true_one_hot = label_true_one_hot * (1 - label_smoothing) + label_smoothing / num_classes
    # For samples with confidence_weight > 1.0, apply special label smoothing
    high_confidence_indices = (confidence_weight > 1.0).nonzero(as_tuple=True)[0]
    for i in high_confidence_indices:
        true_class = label_true[i].item()
        label_true_one_hot[i] = label_smoothing_increased / num_classes
        label_true_one_hot[i, true_class] = 1 - label_smoothing_increased
    return label_true_one_hot
    
def show_metrics(true_lats, true_lngs, true_clusters, pred_clusters, min_distance, logger,
                 all_thresholds=False, smooth_labels=False):
    """
    Computes and logs various evaluation metrics for model performance.

    This function calculates metrics like mean absolute error, median, accuracy, precision,
    recall, F1-score, and others for different threshold values. It is used for evaluating 
    the model's performance on a validation or test dataset.

    Parameters:
    -----------
    true_lats : list or numpy.ndarray
        The true latitude values of the dataset.
    true_lngs : list or numpy.ndarray
        The true longitude values of the dataset.
    true_clusters : list or numpy.ndarray
        The true cluster IDs of the dataset.
    pred_clusters : list or numpy.ndarray
        The predicted cluster IDs of the dataset.
    min_distance : float
        The minimum distance threshold used for computing some evaluation metrics.
    logger : logging.Logger
        The logger object for logging the computed metrics.
    all_thresholds : bool, optional
        Whether to compute evaluation metrics for a range of thresholds. Default is False.
    smooth_labels : bool, optional
        Indicates if smooth labels were used. Affects the accuracy calculation. Default is False.

    Returns:
    --------
    None
        The function logs the evaluation metrics using the provided logger object.
    """
    # displays metrics, for different thresholds
    MAEs = []
    Medians = []
    F1s = []
    percentages = []
    Acc5 = []
    Acc20 = []
    Acc100 = []
    thresholds = [j / 20.0 for j in range(0, 20)] if all_thresholds else [0, 0.75]
    for threshold in thresholds:
        part_true_distance_ls = []
        acc = []
        acc5 = []
        acc20 = []
        acc100 = []
        vals = []
        tp = 0
        fp = 0
        fn = 0
        for i in range(true_clusters.shape[0]):
            pred_cluster_proba = torch.nn.Softmax(dim=0)(torch.tensor(pred_clusters[i])).numpy()
            pred = pred_clusters[i].argmax()
            # sort pred by value, and store indexes to pred_sorted, so that pred_sorted[0]==pred, and pred_sorted[1] is the next
            pred_sorted = np.argsort(pred_cluster_proba)[::-1]
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
            acc5.append(1 if true_clusters[i] in pred_sorted[:5] else 0)
            acc20.append(1 if true_clusters[i] in pred_sorted[:20] else 0)
            acc100.append(1 if true_clusters[i] in pred_sorted[:100] else 0)
            part_true_distance_ls.append(dist)
        part_true_distance_ls = pd.Series(part_true_distance_ls)
        MAEs.append(part_true_distance_ls.mean())
        Medians.append(part_true_distance_ls.median())
        percentages.append(len(part_true_distance_ls) / true_clusters.shape[0])
        Acc5.append(pd.Series(acc5).mean())
        Acc20.append(pd.Series(acc20).mean())
        Acc100.append(pd.Series(acc100).mean())
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
                    f'Accuracy@5 {pd.Series(acc5).mean()} ' +
                    f'Accuracy@20 {pd.Series(acc20).mean()} ' +
                    f'Accuracy@100 {pd.Series(acc100).mean()} ' +
                    f'percentage {len(part_true_distance_ls) / true_clusters.shape[0]} ' +
                    f'acc {pd.Series(acc).mean()} ' +
                    f'precision {precision} recall {recall} ' +
                    f'f1@{min_distance} {f1}')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_sampling_weights(df_train, args, tree, country_by_coordinates, lang_home_country, cluster_df):
    """
    Calculates sampling weights for training data based on language, country, and cluster distribution.

    Parameters:
    -----------
    df_train : pandas.DataFrame
        The training dataframe.
    args : argparse.Namespace
        The command-line arguments for configuration.
    tree : sklearn.neighbors.BallTree
        BallTree for efficient distance queries.
    country_by_coordinates : dict
        A dictionary mapping coordinates to country codes.
    lang_home_country : dict
        A dictionary mapping language codes to home countries.
    cluster_df : pandas.DataFrame
        DataFrame containing cluster information.

    Returns:
    --------
    pandas.DataFrame
        The updated dataframe with a new 'weight' column containing the calculated weights.
    """
    logger.info("calculating weights")

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
        logger.info(f"number of weights {len(weights)} max  {max(weights)} min {min(weights)}")
    # df_train.to_csv('df_train_weights.csv')
    return df_train
    
    
def handle_confident_errors(df_train, model, args, device, tokenizer_name, tree, max_length, smooth_labels_train, intermediate_cluster_df, training_params):
    """
    Upsamples confident errors in the training data based on model predictions and confidence.

    Parameters:
    -----------
    df_train : pandas.DataFrame
        The training dataframe.
    model : torch.nn.Module
        The trained model for making predictions.
    args : argparse.Namespace
        The command-line arguments for configuration.
    device : torch.device
        The device (CPU/GPU) to be used for the model.
    tokenizer_name : str
        The name of the tokenizer used in the dataset.
    tree : sklearn.neighbors.BallTree
        BallTree for efficient distance queries.
    max_length : int
        Maximum sequence length for the model inputs.
    smooth_labels_train : torch.Tensor or None
        Smoothed labels for the training data, if applicable.
    intermediate_cluster_df : pandas.DataFrame or None
        DataFrame containing intermediate cluster information, if applicable.
    model_type : str
        The type of model used.

    Returns:
    --------
    pandas.DataFrame
        The updated training dataframe with adjusted weights for confident errors.
    """
    if args.upsample_confident_errors > 0.0:
        df_train['confidence_weight'] = 1.0

    if args.max_train_steps_epoch is not None:
        df_train_part = df_train.sample(n=args.max_train_steps_epoch * args.batch_size).reset_index(drop=True)
        training_set = ByT5ClusteredClassifierDataset(df_train_part, tokenizer_name, tree, max_length,
                                                      None,
                                                      smooth_labels_train, intermediate_cluster_df, args.model_type)
        training_generator = DataLoader(training_set, **training_params)
        training_generator_not_sampled = DataLoader(training_set, **test_params)
        if args.upsample_confident_errors > 0.0 and epoch > 0:
            logger.info(f"upsample_confident_errors={args.upsample_confident_errors}")
            logger.info(f"old max weight={df_train_part['weight'].max()}")
            # do predictions for training set
            true_lats, true_lngs, true_clusters, pred_clusters, texts, confidences = validate_rel(
                epoch=args.num_epochs, model=model,
                valid_generator=training_generator_not_sampled,
                logger=logger, show_progress=True,
                no_save=True,
                smooth_labels=smooth_labels, smooth_labels_test=smooth_labels)

            percentage_upsample_confidence_weight, percentage_upsample_weight = [int(x) for x in
                                                                                 args.upsample_confident_thresholds.split(
                                                                                     ",")]
            pred_cluster_ids = [pred_clusters[i].argmax() for i in range(len(pred_clusters))]

            if percentage_upsample_confidence_weight > 0:
                # increase sampling weight if prediction is confident and wrong
                # for top 5% of confident errors increase confidence_weight:
                threshold = -1 * np.percentile(-1 * np.array(confidences),
                                               percentage_upsample_confidence_weight)
                logger.info(f"threshold for top {percentage_upsample_confidence_weight}%={threshold}")
                indexes_to_increase = (np.array(confidences) > threshold) & (
                        np.array(true_clusters) != np.array(pred_cluster_ids))
                logger.info(
                    f"top {percentage_upsample_confidence_weight} % confidence upsample count={indexes_to_increase.sum()}")

                df_train_part.iloc[indexes_to_increase].to_csv('indexes_to_increase.csv')
                if indexes_to_increase.sum() >= 10:
                    df_train_part.iloc[indexes_to_increase, df_train_part.columns.get_loc(
                        'confidence_weight')] *= 1.0 + args.upsample_confident_errors

                    # display 10 random samples from those with increased weight:
                    indexes_to_display = np.random.choice(np.where(indexes_to_increase)[0], size=10)
                    for i in indexes_to_display:
                        logger.info(
                            f"index={indexes_to_increase[i]} true={true_clusters[i]}, pred={pred_cluster_ids[i]}, "
                            f"confidence={confidences[i]}, text={texts[i]}")
            else:
                threshold = max(confidences)

            if args.upsample_nonconfident_true > 0:
                threshold20 = -1 * np.percentile(-1 * np.array(confidences), 30)
                threshold10 = -1 * np.percentile(-1 * np.array(confidences), 10)
                indexes_to_increase = (np.array(confidences) > threshold20) & (
                        np.array(confidences) < threshold10) & np.array(true_clusters) == np.array(
                    pred_cluster_ids)
                logger.info("upsample_nonconfident_true count={}".format(indexes_to_increase.sum()))
                df_train_part.loc[indexes_to_increase, 'weight'] *= (1 + args.upsample_nonconfident_true)

            # for top 5..10% of confident errors, increase weight
            threshold_for_weight = -1 * np.percentile(-1 * np.array(confidences), percentage_upsample_weight)
            logger.info(f"threshold={threshold} for top {percentage_upsample_weight}%")
            logger.info(f"true_clusters {np.array(true_clusters).shape}")
            logger.info(f"pred_cluster_ids {np.array(pred_cluster_ids).shape}")
            acc = (np.array(true_clusters) == np.array(pred_cluster_ids)).sum() / len(df_train_part)
            logger.info(f"accuracy on all samples={acc}")
            # now we pick by confidence between old a new threshold
            indexes_to_increase = (np.array(confidences) > threshold_for_weight) & (
                    np.array(confidences) <= threshold) & (
                                          np.array(true_clusters) != np.array(pred_cluster_ids))
            # logger.info(f"indexes_to_increase={indexes_to_increase}")
            logger.info(f"upsample count={np.count_nonzero(indexes_to_increase)} / {len(df_train_part)}")
            # acc_part = (true_clusters[indexes_to_increase] == pred_cluster_ids[indexes_to_increase]).sum() / len(indexes_to_increase)
            # logger.info(f"accuracy on part={acc_part}")

            # display 10 random samples from those with increased weight:
            if np.count_nonzero(indexes_to_increase) >= 10:
                indexes_to_display = np.random.choice(np.where(indexes_to_increase)[0], size=10)
                for i in indexes_to_display:
                    logger.info(
                        f"index={indexes_to_increase[i]} true={true_clusters[i]}, pred={pred_cluster_ids[i]}, "
                        f"confidence={confidences[i]}, text={texts[i]}")

            df_train_part.loc[indexes_to_increase, 'weight'] *= (1 + args.upsample_confident_errors)
            logger.info(f"new max weight={df_train_part['weight'].max()}")

        sampler = WeightedRandomSampler(torch.tensor(df_train_part['weight']), len(df_train_part))
        training_params = {"batch_size": args.batch_size,
                           "shuffle": True if sampler is None else False,
                           "num_workers": 0, "sampler": sampler}
        logger.info(f"selected {len(df_train_part)} samples from train df {len(df_train)}")
    else:
        df_train_part = df_train
    return df_train_part
    
    
def configure_argparse():
    """
    Configures argparse for command-line argument parsing.

    Returns:
    --------
    argparse.ArgumentParser
        The configured argparse.ArgumentParser instance.
    """
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
    parser.add_argument('--random_state', type=int, default=300)
    parser.add_argument('--eval_batches', type=int, default=32, help='Number of batches for evaluation')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_steps', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--min_distance', type=int, default=500)
    parser.add_argument('--byt5_model_name', type=str, default='google/byt5-small')
    parser.add_argument('--wandb_project', type=str)
    parser.add_argument('--wandb_resume', type=bool, default=False)
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
    parser.add_argument('--confidence_loss_weight', type=float)
    parser.add_argument('--confidence_loss_warmup_steps', type=int, default=10000)
    parser.add_argument('--gradient_clipping', type=float)
    parser.add_argument('--upsample_confident_errors', type=float, default=0.0)
    parser.add_argument('--upsample_confident_thresholds', type=str, default="5,10",
                        help="Percentage how many confident errors to upsample by confidence_weight and by weights")
    parser.add_argument('--max_train_steps_epoch', type=int, help='Limit number of training steps per epoch')
    parser.add_argument('--upsample_nonconfident_true', type=float, default=0.0)
    parser.add_argument('--intermediate_clustering', type=str, default=None)
    parser.add_argument('--top100_wrong', type=str, default=None)
    parser.add_argument('--wrong_weight', type=float, default=None)
    parser.add_argument('--model_type', type=str, default='byt5')
    parser.add_argument('--bert_model_name', type=str)
    parser.add_argument('--regression', type=bool, default=False)
    return parser
    

if __name__ == "__main__":
    parser = configure_argparse()
    args = parser.parse_args()
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
        wandb.init(project=args.wandb_project, resume=args.wandb_resume)

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

    intermediate_cluster_df = None
    if args.intermediate_clustering is not None:
        with open(args.intermediate_clustering + 'clustering.pkl', 'rb') as fin:
            intermediate_cluster_df, _ = pickle.load(fin)

    top100_wrong = None
    if args.top100_wrong is not None:
        with open(args.top100_wrong, 'rb') as fin:
            top100_wrong = pickle.load(fin)

    if args.train_test_input_file is not None:
        df_train, df_test = read_train_test_data(args.train_test_input_file)
        logger.info("finish reading train, test file")

    if args.test_input_file is not None:
        df_test = read_csv_data(args.test_input_file, nrows=args.max_test)
        logger.info("finish reading test file")
        if args.smooth_labels and 'author_id' in df_test.columns:
            smooth_labels_test = calculate_smooth_labels(df_test, cluster_df, tree, args.nearest_count,
                                                         args.nearest_weight, args.author_weight,
                                                         args.nearest_distance, top100_wrong, args.wrong_weight)
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
                                                          args.nearest_distance, top100_wrong, args.wrong_weight)
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
        with open(args.country_by_coordinates, 'rb') as fin:
            country_by_coordinates = pickle.load(fin)
        with open(args.lang_home_country, 'rb') as fin:
            lang_home_country = pickle.load(fin)
        df_train = calculate_sampling_weights(df_train, args, tree, country_by_coordinates, lang_home_country, cluster_df)
    else:
        sampler = None

    tokenizer_name = args.bert_model_name if args.bert_model_name is not None else args.byt5_model_name

    training_params = {"batch_size": args.batch_size,
                       "shuffle": True if sampler is None else False,
                       "num_workers": 0, "sampler": sampler}
    test_params = {"batch_size": args.batch_size,
                   "shuffle": False,
                   "num_workers": 0}
    if args.do_train:
        training_set = ByT5ClusteredClassifierDataset(df_train,tokenizer_name, tree, max_length, None,
                                                      smooth_labels_train, intermediate_cluster_df, args.model_type)

    full_test_set = ByT5ClusteredClassifierDataset(df_test, tokenizer_name, tree, max_length, None,
                                                   smooth_labels_test, intermediate_cluster_df, args.model_type)

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
        if args.model_type == 'byt5':
            if args.regression:
                model = ByT5_regressorNew(model_name=args.byt5_model_name,
                                    keep_layer_count=args.keep_layer_count, intermediate_cluster_df=intermediate_cluster_df)
            else:
                model = ByT5_classifierNew(n_clusters=len(cluster_df), model_name=args.byt5_model_name,
                                    keep_layer_count=args.keep_layer_count, intermediate_cluster_df=intermediate_cluster_df)
        elif args.model_type == 'charcnn':
            model = ModifiedCharCNN(n_clusters_=len(cluster_df))
        elif args.model_type == 'charlstm':
            model = MultiLayerCharLSTM(n_clusters_=len(cluster_df))
        elif args.model_type == 'bert':
            model = AutoModelForSequenceClassification.from_pretrained(args.bert_model_name, num_labels=len(cluster_df))
            if args.keep_layer_count is not None:
                model = deleteEncodingLayersDeberta(model, args.keep_layer_count)
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

    smooth_labels_test = 'author_id' in df_test.columns

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Then call this function with your model
    print(f'The model has {count_parameters(model):,} trainable parameters')

    if args.do_train:
        lr_epoch = [1e-3, 1e-4, 1e-5]
        if smooth_labels:
            lr_epoch = [1e-3, 1e-3, 1e-4]
        if args.lr_epoch is not None:
            lr_epoch = [float(x) for x in args.lr_epoch.split(",")]
            logger.info(f"using lr_epoch {lr_epoch}")

        num_epochs = args.num_epochs
        for epoch in range(args.start_epoch, num_epochs):
            df_train_part = handle_confident_errors(df_train, model, args, device, tokenizer_name, tree, max_length, smooth_labels_train, intermediate_cluster_df, training_params)
            training_set = ByT5ClusteredClassifierDataset(df_train_part, tokenizer_name, tree, max_length, None,
                                                          smooth_labels_train, model_type=args.model_type)
            training_generator = DataLoader(training_set, **training_params)
            training_generator_not_sampled = DataLoader(training_set, **test_params)
            model.train()

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
        true_lats, true_lngs, true_clusters, pred_clusters, texts, confidences = validate_rel(epoch=args.num_epochs,
                                                                                              model=model,
                                                                                              valid_generator=full_test_generator,
                                                                                              logger=logger,
                                                                                              show_progress=True,
                                                                                              no_save=True,
                                                                                              smooth_labels=smooth_labels,
                                                                                              smooth_labels_test=smooth_labels_test)
        test_results_filename = 'test_results.pkl'
        with open(test_results_filename, 'wb') as fout:
            pickle.dump((true_lats, true_lngs, true_clusters, pred_clusters, texts), fout)
        show_metrics(true_lats, true_lngs, true_clusters, pred_clusters, args.min_distance, logger,
                     all_thresholds=args.all_thresholds, smooth_labels=smooth_labels)
