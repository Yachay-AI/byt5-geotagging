import torch
from transformers import BertTokenizer, ByT5Tokenizer
from types import SimpleNamespace
from aux_files.models import CompositeModel

class GeoModelPredictor():
    '''
    Wrapper for current architecture of model.
    '''
    def __init__(self, model_path, device=None):
        '''
        Select the device, load tokenizers and model.

            arguments:
                model_path (str): path to the pretrained model
                device (torch.device): CPU / GPU device for the model
        '''
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.byte_tokenizer = ByT5Tokenizer.from_pretrained('google/byt5-small')
        self.word_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.load_model(model_path, self.device)
        

    def encode_texts(self, texts):
        '''
        Encode text/texts by ByT5 byte and Bert word tokenizers
            arguments:
                texts (list<str>): list of texts
            
            return:
                (byte_tokens, word_tokens): first for char tokenization, second for word tokenization. 
                                            Both as torch.FloatTensors
        '''
        word_tokens = self.word_tokenizer(texts, padding=True, return_tensors='pt', truncation=True)

        def tokenize_maybe_pad(tokenizer, tokens, length=7):
            '''Padding to the same length'''
            tokenized = tokenizer(tokens, padding=True, return_tensors='pt')
            if tokenized.input_ids.size(1) < length:
                tokenized = tokenizer(tokens, padding='max_length', max_length=length, return_tensors='pt')
            return tokenized

        byte_tokens = tokenize_maybe_pad(self.byte_tokenizer, texts)

        encoded_tokens = (byte_tokens, word_tokens)
        encoded_tokens = [i.to(self.device) for i in encoded_tokens]

        return encoded_tokens


    def load_model(self, model_path, device=None):
        '''
        Load the model by path

            arguments:
                model_path (str): path to the pretrained model
                device (torch.device): CPU / GPU device for the model
        '''
        state = torch.load(model_path, map_location=device)
        self.model_args = SimpleNamespace(** state['model_args'])
        if (self.model_args.conf_estim):
            self.min_conf = self.model_args.min_conf
            self.max_conf = self.model_args.max_conf
            self.conf_range = self.max_conf - self.min_conf

        self.model = CompositeModel(self.model_args)
        self.model.load_state_dict(state['state_dict'])
        self.model.to(device)
        self.model.eval();

    def unsqueze_conf(self, confidence):
        ''' Fit confidence to [0,1] bounds '''
        confidence = (confidence - self.min_conf) / self.conf_range
        return torch.clamp(confidence, min=0, max=1)
    
    def predict(self, texts):
        '''
        Predict coordinates with confidence from text or list of texts.

            arguments:
                texts (str | list<str> | np.array<str>): text or list of texts

            return: 
                (if conf_estim) list of [[longitude, latitude], confidence]
                (else) list of [longitude, latitude]
        '''
        if isinstance(texts, str):
            texts = [texts]
        preds = []
        confidences = []
        for i in range(0, len(texts), 200):  # 200 - optimal batch size
            byte_tokens, word_tokens = self.encode_texts(texts[i:i+200])

            with torch.no_grad():
                pred = self.model(byte_tokens, word_tokens)
                if (self.model_args.conf_estim):
                    pred, confidence = pred

                    confidence = confidence.T[0].T if confidence.size()[0] > 1 else confidence[0]
                    confidence = self.unsqueze_conf(confidence)

                    confidence = confidence.cpu().detach().numpy().tolist()
                    confidences.extend(confidence)
            
            if len(pred.size()) == 1:
                pred = pred.unsqueeze(0)
                
            pred = pred.cpu().detach().numpy().tolist()
            preds.extend(pred)
        if (self.model_args.conf_estim):
            if len(preds) == 1:
                return [preds[0], confidences[0]]
            return list(zip(preds, confidences))
        else:
            return preds[0] if len(preds) == 1 else preds