import os
from check import PHONEMES, LABELS
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


DATA_DIR = r'/media/konan/TempDrive/homework'
DATA_DIR = os.path.join(DATA_DIR, 'hw3p2')
TRAIN = 'train-clean-360'
VALID = 'dev-clean'
TEST = 'test-clean'


def MFCC(folder):
    return os.path.join(DATA_DIR, folder, 'mfcc',)


def TRANSCRIPT(folder):
    return os.path.join(DATA_DIR, folder, 'transcript', 'raw')


stoi = {phoneme: i for i, phoneme in enumerate(PHONEMES)}
itos = {i: phoneme for i, phoneme in enumerate(LABELS)}

config = {
    'labels': LABELS,
    'stoi': stoi,
    'itos': itos,
    'mfcc': MFCC,
    'transcript': TRANSCRIPT,
    'train_dir': TRAIN,
    'valid_dir': VALID,
    'test_dir': TEST,
    'context': 0,
    '<pad>': stoi[''],
    'blank': stoi[''],
    'batch_size': 350,
    'num_workers': 18,
    'device': device,
    'max_epochs':900,
    'lr': 1e-3,
}


print()
