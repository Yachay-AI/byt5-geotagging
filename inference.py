from transformers import AutoTokenizer
from model import ByT5_classifier
import torch
import pandas as pd

def byt5_preprocess(text):
    return text.lower()
    #return text
def byt5_predict(text):
    text = byt5_preprocess(text)
    te_feature = byt5_tokenizer([text], truncation=True, padding="max_length", max_length=140,
                              return_tensors='pt')['input_ids']
    te_feature = te_feature.to(device).unsqueeze(0)
    with torch.no_grad():
        te_predictions = byt5_model(te_feature)
    pred_cluster_proba = torch.nn.Softmax(dim=1)(torch.tensor(te_predictions.detach().cpu())).numpy()
    return pred_cluster_proba
def geolocate_text_byt5(text, relevance_threshold=0.25):
    ret = {}
    relevance = 0
    pred_clusters = byt5_predict(text)
    if pred_clusters.max() >= relevance_threshold:
        ret['lat'] = cluster_df.iloc[pred_clusters.argmax()]['lat']
        ret['lon'] = cluster_df.iloc[pred_clusters.argmax()]['lng']
        ret['from'] = 'byt5'
        ret['relevance'] = pred_clusters.max()
        return ret
    else:
        relevance = max(relevance, pred_clusters.max())
    ret['relevance'] = relevance
    ret['from'] = 'none'
    return ret

device = 'cpu'
byt5_model = ByT5_classifier(n_clusters=3000, model_name='google/byt5-small')
byt5_model.load_state_dict(torch.load('pretrained.pt'))
byt5_tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')
cluster_df = pd.read_csv('cluster_df.csv')
    
print(geolocate_text_byt5("im at moscow"))