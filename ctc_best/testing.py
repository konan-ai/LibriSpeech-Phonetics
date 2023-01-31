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


checkpoint_dir = 'checkpoints/v8/epoch=695-lev_distance=2.68.ckpt'


model = Network.load_from_checkpoint(checkpoint_dir,config=config).cuda()
model.eval()

decoder_test = CTCBeamDecoder(config['labels'], beam_width=35, log_probs_input=True,num_processes=10)


preds = []
stoi = config['stoi']
itos = config['labels']



test_dataloader = get_test_dataloader()

for i, (mfcc, mfcc_len) in enumerate(tqdm(test_dataloader)):
    
    with torch.no_grad():
        probs = model(mfcc.cuda(), mfcc_len)
    
        beam_results, beam_scores, timesteps, out_seq_len = decoder_test.decode(probs,seq_lens=mfcc_len)
        
        for i in range(len(beam_results)):
            h_sliced = list(beam_results[i,0][:out_seq_len[i,0]].cpu().numpy())

            h_string = ''.join([itos[c.item()] for c in h_sliced])
            
            preds.append(h_string)

# save as submission csv
import pandas as pd
from config import DATA_DIR,TEST
data_dir = DATA_DIR + '/' + TEST + "/transcript/random_submission.csv"
df = pd.read_csv(data_dir)
df.label = preds
df.to_csv('submission.csv', index = False)

checkpoint_dir = ".".join(checkpoint_dir.split(".")[:-1]) + "_submission.csv"
df.to_csv(checkpoint_dir, index = False)


    
    
    