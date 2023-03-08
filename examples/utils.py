import numpy as np
import _pickle as pickle
import random
import torch.nn as nn
import torch
import os
import sklearn
import sklearn.metrics as skm
import warnings


def nonMASK(tokens, token2idx):
    output_label = []
    output_token = []
    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                output_token.append(token2idx["MASK"])

            # 10% randomly change token to random token
            elif prob < 0.9:
                output_token.append(random.choice(list(token2idx.values())))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later
            output_label.append(token2idx.get(token, token2idx['UNK']))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
            output_token.append(token2idx.get(token, token2idx['UNK']))

    return tokens, output_token, output_label


# static var masking
def covarUnsupMaker(covar, covarprobb=0.4):
    inputcovar = []
    labelcovar = []
    for i,x in enumerate(covar):
        prob = random.random()
        if x != 0:
            if prob <covarprobb:
                inputcovar.append(0)
                if covar[i]==0:

                    labelcovar.append(-1)
                else:
                    labelcovar.append(covar[i])

            else:
                inputcovar.append(covar[i])
                labelcovar.append(-1)
        else:
            inputcovar.append(covar[i])
            labelcovar.append(-1)

    return np.array(inputcovar), np.array(labelcovar)


def randommaskreal(tokens, token2idx):
    output_label = []
    output_token = []
    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                output_token.append(token2idx["MASK"])
                output_label.append(token2idx.get(token, token2idx['UNK']))

            # 10% randomly change token to random token
            elif prob < 0.9:
                output_token.append(random.choice(list(token2idx.values())))
                output_label.append(token2idx.get(token, token2idx['UNK']))

            # -> rest 10% randomly keep current token
            else:
                output_label.append(-1)

            # append current token to output (we will predict these later
                output_token.append(token2idx.get(token, token2idx['UNK']))



        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
            output_token.append(token2idx.get(token, token2idx['UNK']))

    return tokens, output_token, output_label



def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def code2index(tokens, token2idx):
    output_tokens = []
    for i, token in enumerate(tokens):
        output_tokens.append(token2idx.get(token, token2idx['UNK']))
    return tokens, output_tokens




def index_seg(tokens, symbol='SEP'):
    flag = 0
    seg = []

    for token in tokens:
        if token == symbol:
            seg.append(flag)
            if flag == 0:
                flag = 1
            else:
                flag = 0
        else:
            seg.append(flag)
    return seg


def position_idx(tokens, symbol='SEP'):
    pos = []
    flag = 0

    for token in tokens:
        if token == symbol:
            pos.append(flag)
            flag += 1
        else:
            pos.append(flag)
    return pos


def age_vocab(max_age, year=False, symbol=None):
    age2idx = {}
    idx2age = {}
    if symbol is None:
        symbol = ['PAD', 'UNK']

    for i in range(len(symbol)):
        age2idx[str(symbol[i])] = i
        idx2age[i] = str(symbol[i])

    if year:
        for i in range(max_age):
            age2idx[str(i)] = len(symbol) + i
            idx2age[len(symbol) + i] = str(i)
    else:
        for i in range(max_age * 12):
            age2idx[str(i)] = len(symbol) + i
            idx2age[len(symbol) + i] = str(i)

    return age2idx, idx2age


def seq_padding(tokens, max_len, token2idx=None, symbol=None):
    if symbol is None:
        symbol = 'PAD'

    seq = []
    token_len = len(tokens)
    for i in range(max_len):
        if token2idx is None:
            if i < token_len:
                seq.append(tokens[i])
            else:
                seq.append(symbol)
        else:
            if i < token_len:
                # 1 indicate UNK
                seq.append(token2idx.get(tokens[i], token2idx['UNK']))
            else:
                seq.append(token2idx.get(symbol))
    return seq


def seq_padding_reverse(tokens, max_len, token2idx=None, symbol=None):
    if symbol is None:
        symbol = 'PAD'

    seq = []
    token_len = len(tokens)
    tokens = tokens[::-1]
    for i in range(max_len):
        if token2idx is None:
            if i < token_len:
                seq.append(tokens[i])
            else:
                seq.append(symbol)
        else:
            if i < token_len:
                # 1 indicate UNK
                seq.append(token2idx.get(tokens[i], token2idx['UNK']))
            else:
                seq.append(token2idx.get(symbol))
    return seq[::-1]


def age_seq_padding(tokens, max_len, token2idx=None, symbol=None):
    if symbol is None:
        symbol = 'PAD'

    seq = []
    token_len = len(tokens)
    for i in range(max_len):
        if token2idx is None:
            if i < token_len:
                seq.append(tokens[i])
            else:
                seq.append(symbol)
        else:
            if i < token_len:
                # 1 indicate UNK
                seq.append(token2idx[tokens[i]])
            else:
                seq.append(token2idx[symbol])
    return seq



def cal_acc(label, pred, logS=True):
    logs = nn.LogSoftmax()
    label = label.cpu().numpy()
    ind = np.where(label != -1)[0]
    truepred = pred.detach().cpu().numpy()
    truepred = truepred[ind]
    truelabel = label[ind]
    if logS == True:
        truepred = logs(torch.tensor(truepred))
    else:
        truepred = torch.tensor(truepred)
    outs = [np.argmax(pred_x) for pred_x in truepred.numpy()]
    precision = skm.precision_score(truelabel, outs, average='micro')

    return precision

def cal_acc(label, pred, logS=True):
    logs = nn.LogSoftmax()
    label = label.cpu().numpy()
    ind = np.where(label != -1)[0]
    truepred = pred.detach().cpu().numpy()
    truepred = truepred[ind]
    truelabel = label[ind]
    if logS ==True:
        truepred = logs(torch.tensor(truepred))
    else:
        truepred = torch.tensor(truepred)
    outs = [np.argmax(pred_x) for pred_x in truepred.numpy()]
    precision = skm.precision_score(truelabel, outs, average='micro')

    return precision

def partition(values, indices):
    idx = 0
    for index in indices:
        sublist = []
        idxfill = []
        while idx < len(values) and values[idx] <= index:
            # sublist.append(values[idx])
            idxfill.append(idx)

            idx += 1
        if idxfill:
            yield idxfill


def toLoad(model, filepath, custom=None):
    pre_bert = filepath

    pretrained_dict = torch.load(pre_bert, map_location='cpu')
    modeld = model.state_dict()
    # 1. filter out unnecessary keys
    if custom == None:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in modeld}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in modeld and k not in custom}

    modeld.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(modeld)
    return model




def OutcomePrecision(logits, label, sig=True):
    sig = nn.Sigmoid()
    if sig == True:
        output = sig(logits)
    else:
        output = logits
    label, output = label.cpu(), output.detach().cpu()
    tempprc = sklearn.metrics.average_precision_score(label.numpy(), output.numpy())
    return tempprc, output, label


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def precision_test(logits, label, sig=True):
    sigm = nn.Sigmoid()
    if sig == True:
        output = sigm(logits)
    else:
        output = logits
    label, output = label.cpu(), output.detach().cpu()

    tempprc = sklearn.metrics.average_precision_score(label.numpy(), output.numpy())
    return tempprc, output, label


def roc_auc(logits, label, sig=True):
    sigm = nn.Sigmoid()
    if sig == True:
        output = sigm(logits)
    else:
        output = logits
    label, output = label.cpu(), output.detach().cpu()

    tempprc = sklearn.metrics.roc_auc_score(label.numpy(), output.numpy())
    return tempprc, output, label


# golobal function
def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)

