from collections import Counter

import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import csv
import re
import geopy.distance
import sklearn
import sklearn.model_selection
import joblib
import pickle
from math import pi
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import random
import torch
import numpy as np


def lat_fix(val):
    if val < -90:
        val = -90
    elif val > 90:
        val = 90
    return val


def get_dist(pred_lats, pred_lngs, true_lats, true_lngs, scaler):
    pred_values = scaler.inverse_transform(np.stack([pred_lats, pred_lngs], axis=1))
    true_values = scaler.inverse_transform(np.stack([true_lats, true_lngs], axis=1))
    dist = [geopy.distance.great_circle((lat_fix(pred_values[i, 0]), pred_values[i, 1]),
                                        (lat_fix(true_values[i, 0]), true_values[i, 1])).km for i in
            list(range(len(pred_lats)))]
    return dist


def get_evaluation(pred_lats, pred_lngs, true_lats, true_lngs, scaler):
    # print(np.stack([pred_lats, pred_lngs], axis=1))
    pred_values = scaler.inverse_transform(np.stack([pred_lats, pred_lngs], axis=1))
    true_values = scaler.inverse_transform(np.stack([true_lats, true_lngs], axis=1))
    dist = [geopy.distance.great_circle((lat_fix(pred_values[i, 0]), pred_values[i, 1]),
                                        (lat_fix(true_values[i, 0]), true_values[i, 1])).km for i in
            list(range(len(pred_lats)))]
    dist100 = [1.0 if x < 100 else 0.0 for x in dist]
    dist500 = [1.0 if x < 500 else 0.0 for x in dist]
    dist1000 = [1.0 if x < 1000 else 0.0 for x in dist]

    return {'MAE': sum(dist) / len(dist), 'Median error': pd.Series(dist).median(),
            'ACC@100': pd.Series(dist100).mean(), 'ACC@500': pd.Series(dist500).mean(),
            'ACC@1000': pd.Series(dist1000).mean()}
    # return 0


def read_csv_data(filename, nrows=None, skiprows=None):
    if '.parquet' in filename:
        data = pd.read_parquet(filename, engine='pyarrow')
        print('finish read parquet')
        if nrows is not None:
            data = data.sample(n=nrows)
    else:
        data = pd.read_csv(filename, encoding='utf-8', nrows=nrows, skiprows=skiprows, lineterminator='\n')
    nna = data.isna().sum().sum()
    if nna > 0:
        print(f"Warning, dropping {nna} NaN rows")
    data.dropna(inplace=True)
    if not 'coordinates' in data.columns and not 'coords' in data.columns:
        data['coordinates'] = data['lon'].astype(str) + "_" + data['lat'].astype(str)
    else:
        if 'coords' in data.columns:
            data.rename(columns={'coords': 'coordinates'}, inplace=True)
        if not 'lat' in data.columns:
            data[['lon', 'lat']] = data['coordinates'].str.split('_', 1, expand=True)
            data['lat'] = pd.to_numeric(data['lat'])
            data['lng'] = pd.to_numeric(data['lon'])
        else:
            data['lng'] = data['lon']
    if 'label\r' in data.columns:
        data.rename(columns={'label\r': 'label'}, inplace=True)
    if 'clean_text' in data.columns:
        data.rename(columns={'clean_text': 'text'}, inplace=True)
    return data


def read_train_test_data(filename, nrows=None, ntest=None):
    data = read_csv_data(filename, nrows)
    # data['text'] = data['tweet_text']
    random.seed(1)
    df_train, df_test = sklearn.model_selection.train_test_split(data, test_size=ntest if ntest is not None else 0.1)
    return df_train, df_test


def read_scaler_and_vocab(load_dir="models"):
    scaler_filename = load_dir + "/scaler.save"
    scaler = joblib.load(scaler_filename)
    with open(load_dir + '/vocabulary', 'rb') as fin:
        vocabulary = pickle.load(fin)
    vocabulary = list(vocabulary.keys())
    return scaler, vocabulary


def lat_fix_tensor(val):
    return torch.maximum(torch.minimum(val, torch.ones_like(val) * 90), torch.ones_like(val) * -90)


def distance(lat1, lon1, lat2, lon2, scaler, device):
    transformed1 = torch.tensor(scaler.mean_).to(device) + torch.tensor(scaler.scale_).to(device) * torch.vstack(
        [lat1, lon1]).transpose(0, 1).to(device)
    lat1 = lat_fix_tensor(transformed1[:, 0])
    lon1 = transformed1[:, 1]
    transformed2 = torch.tensor(scaler.mean_).to(device) + torch.tensor(scaler.scale_).to(device) * torch.vstack(
        [lat2, lon2]).transpose(0, 1).to(device)
    lat2 = lat_fix_tensor(transformed2[:, 0])
    lon2 = transformed2[:, 1]
    p = pi / 180
    t1 = (lat2 - lat1) * p
    t2 = lat1 * p
    t3 = lat2 * p
    t4 = (lon2 - lon1) * p
    t5 = torch.cos(t4)
    t6 = torch.cos(t1)
    t7 = torch.cos(t2)
    t8 = torch.cos(t3)
    a = 0.5 - t6 / 2 + t7 * t8 * (1 - t5) / 2
    # a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return (12742 * torch.asin(torch.sqrt(a))).mean()


def distance_all(lat1, lon1, lat2, lon2):
    transformed1 = torch.tensor(scaler.mean_).to(device) + torch.tensor(scaler.scale_).to(device) * torch.vstack(
        [lat1, lon1]).transpose(0, 1).to(device)
    lat1 = lat_fix_tensor(transformed1[:, 0])
    lon1 = transformed1[:, 1]
    transformed2 = torch.tensor(scaler.mean_).to(device) + torch.tensor(scaler.scale_).to(device) * torch.vstack(
        [lat2, lon2]).transpose(0, 1).to(device)
    lat2 = lat_fix_tensor(transformed2[:, 0])
    lon2 = transformed2[:, 1]
    p = pi / 180
    t1 = (lat2 - lat1) * p
    t2 = lat1 * p
    t3 = lat2 * p
    t4 = (lon2 - lon1) * p
    t5 = torch.cos(t4)
    t6 = torch.cos(t1)
    t7 = torch.cos(t2)
    t8 = torch.cos(t3)
    a = 0.5 - t6 / 2 + t7 * t8 * (1 - t5) / 2
    # a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return (12742 * torch.asin(torch.sqrt(a)))


def get_evaluation_error_predict(preds, trues, threshold=0.5):
    preds_int = [1 if x >= threshold else 0 for x in preds]
    return {'f1': f1_score(trues, preds_int), 'roc_auc': roc_auc_score(trues, preds),
            'acc': accuracy_score(trues, preds_int), 'balance': (sum(trues) / len(trues)).clone().cpu().item()}


def add_cluster_id_column(dataset, cluster_merges, cluster_df):
    dataset['label'] = [
        cluster_df['coordinates'].index(cluster_merges[dataset.iloc[i]['coordinates']]) if dataset.iloc[i][
                                                                                               'coordinates'] in cluster_merges else -1
        for i in tqdm(list(range(len(dataset))))]


def add_subcluster_id_column(dataset, subcluster_df):
    subclusters = subcluster_df['coordinates'].values.tolist()
    dataset['label'] = [
        subclusters.index(dataset.iloc[i]['coordinates']) if dataset.iloc[i]['coordinates'] in subclusters else -1 for i
        in tqdm(list(range(len(dataset))))]


def get_distances(n_clusters_, cluster_df):
    distance_between_clusters = np.zeros((n_clusters_, n_clusters_))
    for i in range(n_clusters_):
        for j in range(n_clusters_):
            distance_between_clusters[i][j] = geopy.distance.great_circle(
                (cluster_df.iloc[i]['lat'], cluster_df.iloc[i]['lng']),
                (cluster_df.iloc[j]['lat'], cluster_df.iloc[j]['lng'])).km
    distance_between_clusters = torch.tensor(distance_between_clusters)
    return distance_between_clusters


def train_scaler_and_vocab(df):
    print("train_scaler_and_vocab", len(df))
    scaler = StandardScaler()
    scaler.fit_transform(df[['lat', 'lng']].values)
    joblib.dump(scaler, "models/scaler.save")
    vocabulary = Counter()
    for i, row in tqdm(df.iterrows()):
        for c in row['text']:
            vocabulary[c] += 1
    with open('models/vocabulary', 'wb') as fout:
        pickle.dump(vocabulary, fout)
    print("finish train_scaler_and_vocab")
    return read_scaler_and_vocab()


def true_distance_from_pred_cluster(pred, lat_true, lng_true, cluster_df):
    return geopy.distance.great_circle((cluster_df.iloc[pred]['lat'], cluster_df.iloc[pred]['lng']),
                                       (lat_true, lng_true)).km


def dbscan_predict(model, X):
    nr_samples = X.shape[0]
    y_new = np.ones(shape=nr_samples, dtype=int) * -1
    for i in range(nr_samples):
        diff = model.components_ - X[i, :]  # NumPy broadcasting
        dist = np.linalg.norm(diff, axis=1)  # Euclidean distance
        shortest_dist_idx = np.argmin(dist)
        # if dist[shortest_dist_idx] < model.eps:
        y_new[i] = model.labels_[model.core_sample_indices_[shortest_dist_idx]]

    return y_new


def distance_from_pred(cluster_pred_logits, cluster_true, distance_between_clusters):
    pred = cluster_pred_logits.argmax(dim=1)
    s = []
    for i in range(len(cluster_pred_logits)):
        s.append(distance_between_clusters[cluster_true[i], pred[i]])
    return torch.tensor(s)
    # distances = distance_between_clusters[cluster_true, :]


def true_distance_from_pred(cluster_pred_logits, lat_true, lng_true, cluster_df):
    pred = cluster_pred_logits.argmax(dim=1)
    s = []
    for i in range(len(cluster_pred_logits)):
        s.append(geopy.distance.great_circle(
            (cluster_df.iloc[pred[i].item()]['lat'], cluster_df.iloc[pred[i].item()]['lng']),
            (lat_true[i], lng_true[i])).km)
    return torch.tensor(s)


def prepare_subclusters(cluster_df, merges, cluster_id):
    cluster_merges = merges[cluster_df.iloc[cluster_id]['coordinates']]
    subcluster_df = pd.Series(cluster_merges).str.split('_', 1, expand=True)
    subcluster_df.columns = ['lng', 'lat']
    subcluster_df['lat'] = pd.to_numeric(subcluster_df['lat'])
    subcluster_df['lng'] = pd.to_numeric(subcluster_df['lng'])
    # distance_between_subclusters = get_distances(len(subcluster_df), subcluster_df)
    return subcluster_df, len(subcluster_df)
