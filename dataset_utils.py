import os, random

import yaml
from easydict import EasyDict as edict

import numpy as np
import torch
from rdkit import Chem


def read_config(path):
    with open(path, 'r') as stream:
        config = yaml.safe_load(stream)
    cfg = edict(config)
    return cfg


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def randomize_smiles(smiles, canonical=False, isomericSmiles=True):
    """Perform a randomization of a SMILES string
    must be RDKit sanitizable"""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    ans = list(range(m.GetNumAtoms()))
    np.random.shuffle(ans)
    nm = Chem.RenumberAtoms(m,ans)
    return Chem.MolToSmiles(nm, canonical=canonical, isomericSmiles=isomericSmiles)


def time_to_str(time):
    hours, minutes, seconds = 0, 0, 0
    if time > 3600:
        hours = int(time / 3600)
        r1 = int(time) % 3600
        if r1 > 60:
            minutes = int(r1 / 60)
            r2 = int(r1) % 60
            seconds = r2
        else:
            seconds = r1
    else:
        if time > 60:
            minutes = int(time / 60)
            seconds = int(time) % 60
        else:
            seconds = int(time)
    time_str = ""
    if hours != 0:
        time_str += str(hours) + "h "
    if minutes != 0:
        time_str += str(minutes) + "m "
    if seconds != 0:
        time_str += str(seconds) + "s "
    return time_str[:-1]


CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                "U": 19, "T": 20, "W": 21,
                "V": 22, "Y": 23, "X": 24,
                "Z": 25 }

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARCANSMISET = { "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6,
             ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12,
             "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18,
             "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
             "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30,
             "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36,
             "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42,
             "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48,
             "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54,
             "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
             "t": 61, "y": 62}
