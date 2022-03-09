#!/usr/bin/env python
# _*_coding:utf-8_*_

import math
import re
import pdb
import torch

from transformers import T5EncoderModel, T5Tokenizer
from transformers import BertModel, BertTokenizer
from transformers import AutoModel, AlbertTokenizer
import gc
import os
import pandas as pd
import requests
from tqdm.auto import tqdm
import numpy as np

def PortTrans_loader(model_name, device = torch.device('cpu')):
    if model_name =="T5":
      model_name = "Rostlab/prot_t5_xl_uniref50" 
      tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False )
      model = T5EncoderModel.from_pretrained(model_name)

    elif model_name == "AlBert":
      model_name = "Rostlab/prot_albert" 
      tokenizer = AlbertTokenizer.from_pretrained(model_name, do_lower_case=False )
      model = AutoModel.from_pretrained(model_name)

    elif model_name == "Bert":
      model_name = "Rostlab/prot_bert" 
      tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
      model = BertModel.from_pretrained(model_name)

    model.to(device)
    model = model.eval()
    if not device==torch.device('cpu'):
      model = model.half()

    return tokenizer, model
      

def embed_dataset(dataset_seqs, tokenizer, model, device, shift_left = 0, shift_right = -1):
  inputs_embedding = []

  #   for sample in tqdm(dataset_seqs):
  for sample in dataset_seqs:
    with torch.no_grad():
        ids = tokenizer.batch_encode_plus([sample], add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")
        embedding = model(input_ids=ids['input_ids'].to(device))[0]
        inputs_embedding.append(embedding[0].detach().cpu().numpy()[shift_left:shift_right])

  return inputs_embedding


def PortTrans_fast(sequences, tokenizer, model, device, **kw):
    ''' PortTrans feature extraction function. 
      The default model is T5, from https://github.com/agemagician/ProtTrans
    Parameters
    ----------
    sequences : list, list of str
      a list of str represent the amino acids
    tokenizer : transformers.tokenizer
      a predefined tokenizer
    model : transformer.model
      a feature extraction model
    device : torch.device()
      which device will be used to extract features
    Returns
    ----------
    list:
      the extracted feature, shape = [num, feature_dim]
    '''
    if isinstance(sequences,str):
        sequences = [sequences]

    encodings = []
    for seq in sequences:
      split_sequence = [s for s in seq]
      
      seq_embd = embed_dataset([split_sequence], tokenizer, model, device)
      # import pdb; pdb.set_trace()

      mean_embed = seq_embd[0].mean(axis = 0).tolist()
      encodings.append(mean_embed)
    # import pdb; pdb.set_trace()
    return np.array(encodings)



def PortTrans_array(sequences, model_name, device, **kw):
    ''' PortTrans feature extraction function. 
      The default model is T5, from https://github.com/agemagician/ProtTrans
    Parameters
    ----------
    sequences : list, list of str
      a list of str represent the amino acids
    model_name : str
      use which pretrained model to extract features
    device : torch.device()
      which device will be used to extract features
    Returns
    ----------
    numpy.Array:
      the extracted feature, shape = [num, feature_dim]
    '''

    tokenizer, model = PortTrans_loader(model_name, device)

    encodings = []
    for seq in tqdm(sequences):
      split_sequence = [s for s in seq]
      seq_embd = embed_dataset([split_sequence], tokenizer, model, device)
      mean_embed = seq_embd[0].mean(axis = 0).tolist()
        
      encodings.append(mean_embed)
    # import pdb; pdb.set_trace()
    print(np.array(encodings).shape)
    return np.array(encodings)





def PortTrans(fastas, **kw):

    # model_name = "Rostlab/prot_t5_xl_uniref50" 
    model_name = "Rostlab/prot_bert" 
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False )
    model = T5EncoderModel.from_pretrained(model_name)
    # device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = model.to(device)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.half()

    encodings = []
    header = ['#']
    for i in range(1024):
        header.append(str(i))
    encodings.append(header)

    for i in tqdm(fastas):
        name, sequence = i[0], re.sub('-', '', i[1])
        # print(name)
        split_sequence = [s for s in sequence]
        seq_embd = embed_dataset([split_sequence], tokenizer, model, device)
        mean_embed = seq_embd[0].mean(axis = 0).tolist()
        code = [name]
        code = code + mean_embed
        assert len(code) == 1025, ("length error")
        encodings.append(code)

    return encodings

