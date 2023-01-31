import os
from check import VOCAB_MAP
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


DATA_DIR = r'/media/konan/ActiveDrive/homework'
DATA_DIR = os.path.join(DATA_DIR, 'hw3p2')
TRAIN = 'train-clean-360'
VALID = 'dev-clean'
TEST = 'test-clean'


def MFCC(folder):
    return os.path.join(DATA_DIR, folder, 'mfcc',)


def TRANSCRIPT(folder):
    return os.path.join(DATA_DIR, folder, 'transcript', 'raw')


stoi = VOCAB_MAP
itos = {value: key for key,value in VOCAB_MAP.items()}

config = {
    'stoi': stoi,
    'itos': itos,
    'mfcc': MFCC,
    'transcript': TRANSCRIPT,
    'train_dir': TRAIN,
    'valid_dir': VALID,
    'test_dir': TEST,
    'context': 0,
    # '<sos>': stoi['<sos>'],
    # '<eos>': stoi['<eos>'],
    # '<pad>': stoi['<pad>'],
    '<sos>': stoi['[SOS]'],
    '<eos>': stoi['[EOS]'],
    '<pad>': stoi['[PAD]'],
    'batch_size': 240,
    'num_workers': 32,
    'device': device,
    'max_epochs':700,
    'lr': 7e-4,
}


print()
