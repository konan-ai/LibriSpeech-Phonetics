from multiprocessing import Pool
from dataloader import get_test_dataloader
from dataset import AudioDataset
import re

import torch
from torch.utils.data import DataLoader
from ctcdecode import CTCBeamDecoder
from config import config

from model import Network
from tqdm import tqdm
from lev_distance import indices_to_chars


checkpoint_dir = 'checkpoints/first_try/epoch=137-lev_distance=2.37.ckpt'


model = Network.load_from_checkpoint(checkpoint_dir,config=config).cuda()
model.eval()
preds = []
stoi = config['stoi']
itos = config['itos']

from check import ARPAbet
itos = ARPAbet


test_dataloader = get_test_dataloader()

for i, (mfcc, mfcc_len) in enumerate(tqdm(test_dataloader)):
    mfcc_len = mfcc_len.cpu()
    with torch.no_grad():
        probs,_ = model(mfcc.cuda(), mfcc_len)
        
        raw_preds = probs.argmax(dim=2).cpu().numpy()
        
        # probs = probs.transpose(1,2)
        
        for i in range(len(raw_preds)):
            h_sliced = list(raw_preds[i])
            h_sliced = indices_to_chars(h_sliced)

            h_string = ''.join([itos[c.item()] for c in h_sliced])
            
            preds.append(h_string)
            
# save as submission csv
import pandas as pd
from config import DATA_DIR,TEST
data_dir = DATA_DIR + '/' + TEST + "/transcript/random_submission.csv"
df = pd.read_csv(data_dir)
df.label = preds

# df = df.rename(columns={'index':'id'})
df.to_csv('submission.csv', index = False)

checkpoint_dir = ".".join(checkpoint_dir.split(".")[:-1]) + "_submission.csv"
df.to_csv(checkpoint_dir, index = False)


    
    
    