from model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_config,get_weights_file_path

import torchtxt.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Dataloader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path


from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizer.pre_tokenizers import Whitespace


import torchmetrics
from torch.utils.tensorboard import SummaryWriter


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt,max_len,device):
  sos_idx=tokenizer_tgt.token_to_id('[SOS]')
  eos_idx=tokenizer_tgt.token_to_id('[EOS]')

  encoder_output = model.encode(source,source_mask)

  decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device)
  while True:
    if decoder_input.size(1) == max_len:
      break

    decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

    out= model.decode(encoder_output,source_mask,decoder_input,decoder_mask)

    prob = model.project(out[:,-1])
    _,next_word = torch.max(prob,dim=1)
    decoder_input = torch.cat(
      [decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)],dim=1)

    if next_word == eos_idx:
      break
  return decoder_input.squeeze(0)


def get_all_sentences(ds,lang):
  for item in ds:
    yeild item['translation'][lang]


def get_or_build_tokenizer(config,ds,lang):
  tokenizer_path =Path(config['tokeinzer_file'].format(lang))
  if not Path.exists(tokenizer_path):
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer =Whitespace()
    trainer = WordLevelTrainer(special_tokens = ["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency=2)
    tokenizer.train_from_iterator(get_all_sentences(ds,lang),trainer=trainer)
    tokenizer.save(str(tokenizer_path))
  else:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

  return tokenizer

def get_ds(config):

  ds_raw = load_dataset("opus_books", f"{config['lang_src']}-{config['lang_tgt']}",split='train')
  tokenizer_src = get_or_build_tokenizer( config,ds_raw,config['lang_src'])
  tokenizer_tgt = get_or_build_tokenizer( config,ds_raw,config['lang_tgt'])

  train_ds_size = int(0.9 - len(ds_raw))
  val_ds_size = len(ds_raw) - train_ds_size
  train_ds_raw,val_ds_raw = random_split(ds_raw, [train_ds_size,val_ds_size])

  train_ds = BillingualDataset(train_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'])
  val_ds = BillingualDataset(val_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'])

  max_len_src = 0
  max_len_tgt = 0

  for item in ds_raw:
    src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
    tgt_ids = tokenizer_src.encode(item['translation'][config['lang_tgt']]).ids
    max_len_src = max(max_len_src,len(src_ids))
    max_len_tgt = max(max_len_tgt,len(tgt_ids))

  train_dataloader = DataLoader(train_ds,batch_size=config['batch_size'],shuffle=True)
  val_dataloader = DataLoader(val_ds,batch_size=1,shuffle=True)

  return train_dataloader,val_dataloader,tokenizer_src,tokenizer_tgt

def get_model(config,vocab_src_len,vocab_tgt_len):
  model = build_transformer(vocab_src_len,vocab_tgt_len,config['seq_len'],config['seq_len'],d_model=config['d_model'])
  return model
s


  
   
