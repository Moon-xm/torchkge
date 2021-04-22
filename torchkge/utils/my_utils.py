import os
import pickle
import time
from datetime import timedelta
import torch
import torch.nn as nn
from torch.optim import optimizer
from typing import Tuple

_MODEL_STATE_DICT = "model_state_dict"
_OPTIMIZER_STATE_DICT = "optimizer_state_dict"
_EPOCH = "epoch"
_BEST_SCORE = "best_score"

def safe_strip(x):
    """
    safe strip '\n' ,' ', '\t' at two side
    :param x: str
    :return: str
    """
    try:
        return x.strip()
    except:
        return x

def create_dir_not_exists(path):
    """
    make dir if not exists
    :param path: file_path
    :return: None
    """
    if not os.path.exists(path):
        os.mkdir(path)

def pickle_dump(ls_or_dic, path):
    """
    save file using binary
    :param ls_or_dic: a list or a dic
    :param path: where to save
    :return: None
    """
    with open(path, 'wb') as f:
        pickle.dump(ls_or_dic, f)
        # f.write(ls_or_dic)

def pickle_load(path):
    """
    load a file in binary
    :param path: file path
    :return: file content
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

def time_since(since):
    """
    time cost
    :param since: start time
    :return: eg: 00:05:12
    """
    time_dif = time.time() - since
    time_usage = timedelta(seconds=int(round(time_dif)))
    return time_usage

def load_ckpt(checkpoint_path, model, optim=None, train=True) -> Tuple[int, float]:
    """Loads training checkpoint.
    :param checkpoint_path: path to checkpoint
    :param model: model to update state
    :param optim: optimizer to  update state
    :return tuple of starting epoch id, starting step id, best checkpoint score
    """
    if train:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint[_MODEL_STATE_DICT])
        optim.load_state_dict(checkpoint[_OPTIMIZER_STATE_DICT])
        start_epoch_id = checkpoint[_EPOCH] + 1
        best_score = checkpoint[_BEST_SCORE]
        return start_epoch_id,  best_score
    else:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint[_MODEL_STATE_DICT])


def save_ckpt(model: nn.Module, optim: optimizer.Optimizer, epoch_id: int, best_score: float, model_save_path: str):
    torch.save({
        _MODEL_STATE_DICT: model.state_dict(),
        _OPTIMIZER_STATE_DICT: optim.state_dict(),
        _EPOCH: epoch_id,
        _BEST_SCORE: best_score
    }, model_save_path)


def dic_save(ls_or_dic, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write("%d\n"%(len(ls_or_dic)))
        for (idx, word) in enumerate(ls_or_dic):
            f.write(word+'\t'+str(idx)+'\n')


def dic_load(path):
    dic = {}
    with open(path, 'r', encoding='utf-8') as f:
        tot = int(f.readline())
        for i in range(tot):
            content = f.readline()
            word, idx = content.strip().split('\t')
            dic[word] = int(idx)
    return dic