import torch
from torch.utils.data import Dataset

from config import config
import os
import numpy as np
from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torchaudio import transforms
from torch import nn


def f_mask(data):
    mask = transforms.FrequencyMasking(freq_mask_param=5)
    return mask(data)

def t_mask(data):
    mask = transforms.TimeMasking(time_mask_param=70)
    return mask(data)    

def transforms_data(data):
    data = f_mask(data)
    data = t_mask(data)
    return data
# valid_audio_transforms = transforms.MelSpectrogram()



class AudioDataset(Dataset):

    def __init__(self, mode, check = False): 
        '''
        Initializes the dataset.

        INPUTS: What inputs do you need here?
        '''
        
        self.mode = mode

        # Load the directory and all files in them
        folders = [config['train_dir'], config['valid_dir'], config['test_dir']]
        
        max_samples = 1000 if check else int(1e8)
        
        self.mfcc_dir = config['mfcc'](folders[mode])
        self.mfcc_files = sorted(os.listdir(self.mfcc_dir))[:max_samples]
        
        # if not check and mode == 0: 
        #     self.mfcc_dir2 = self.mfcc_dir[-3:] + "360"
        #     self.mfcc_files = sorted(os.listdir(self.mfcc_dir2) + self.mfcc_files)

        
        self.stoi = config['stoi']
        self.itos = config['itos']
        self.context = config['context']

        #TODO
        # WHAT SHOULD THE LENGTH OF THE DATASET BE?
        self.length = len(self.mfcc_files)
        
        mfcc_names = self.mfcc_files
        self.mfcc = []
        
        if mode in [0,1]:
            self.transcript_dir = config['transcript'](folders[mode])
            self.transcript_files = sorted(os.listdir(self.transcript_dir))[:max_samples]
            transcript_names = self.transcript_files
            self.transcripts = []
            max_transcript = 0
        
        max_mfcc = 0
        
        for i in tqdm(range(0, len(mfcc_names))):

            mfcc = np.load(os.path.join(self.mfcc_dir, mfcc_names[i]))

            # cepstral mean normalization
            mfcc = (mfcc - mfcc.mean())/mfcc.std()
            mfcc = torch.tensor(mfcc)
            self.mfcc.append(mfcc)

            if len(mfcc) > max_mfcc:
                max_mfcc = len(mfcc)
            
            if mode in [0,1]:
                transcript = np.load(os.path.join(
                    self.transcript_dir, transcript_names[i]))
                # convert to phoneme indices
                transcript = torch.Tensor([self.stoi[phoneme] for phoneme in transcript]).to(torch.long)
                # transcript = torch.Tensor([self.stoi["[SOS]"]] + [self.stoi[phoneme] for phoneme in transcript] + [self.stoi["[EOS]"]])
                self.transcripts.append(transcript)
                
                if len(transcript) > max_transcript:
                    max_transcript = len(transcript)
            
            
            # TODO: check if padding required
            # mfcc = np.pad(mfcc, ((self.context, self.context), (0, 0)), 'constant', constant_values=0)
            # for i in range(self.context, len(mfcc)-self.context):
            #     self.mfccs.append(mfcc[np.newaxis,i-self.context:i+self.context+1])
            #     self.transcripts.append(transcript[i]) 
        
        # TODO: sort the mfccs and transcripts based on length
        
        print("Max mfcc length: ", max_mfcc)
        if mode in [0,1]:
            print("Max transcript: ", max_transcript)
            # self.sort_dataset()
        

        '''
        You may decide to do this in __getitem__ if you wish.
        However, doing this here will make the __init__ function take the load of
        loading the data, and shift it away from training.
        '''
    
    def sort_dataset(self):
        # sort the dataset by mfcc length
        self.mfcc, self.transcripts = zip(*sorted(zip(self.mfcc, self.transcripts), key=lambda x: len(x[0])))

    def __len__(self):
        
        '''
        TODO: What do we return here?
        '''
        return self.length
    
    def __getitem__(self, ind):
        '''
        TODO: RETURN THE MFCC COEFFICIENTS AND ITS CORRESPONDING LABELS

        If you didn't do the loading and processing of the data in __init__,
        do that here.

        Once done, return a tuple of features and labels.
        '''
        if self.mode == 0:
            return transforms_data(self.mfcc[ind].T.unsqueeze(0)).squeeze(0).T, self.transcripts[ind]
        if self.mode == 1:
            return self.mfcc[ind], self.transcripts[ind]
        else:
            return self.mfcc[ind]

    def collate_fn(self,batch):
        '''
        TODO:
        1.  Extract the features and labels from 'batch'
        2.  We will additionally need to pad both features and labels,
            look at pytorch's docs for pad_sequence
        3.  This is a good place to perform transforms, if you so wish. 
            Performing them on batches will speed the process up a bit.
        4.  Return batch of features, labels, lenghts of features, 
            and lengths of labels.
        '''
        # batch of input mfcc coefficients
        if self.mode in [0,1]:
            batch_mfcc = [item[0] for item in batch]
        else:
            batch_mfcc = batch
            
        # HINT: CHECK OUT -> pad_sequence (imported above)
        # Also be sure to check the input format (batch_first)
        lengths_mfcc =  torch.tensor([len(item) for item in batch_mfcc])
        batch_mfcc_pad = pad_sequence(batch_mfcc, batch_first=True, padding_value=config["<pad>"])

        # batch of output phonemes
        if self.mode in [0,1]:
            batch_transcript = [item[1] for item in batch]
            batch_transcript_pad = pad_sequence(batch_transcript, batch_first=True, padding_value=config['<pad>'])
            lengths_transcript = torch.tensor([len(item) for item in batch_transcript])
            
            # random shuffle the batch
            index = torch.randperm(len(batch_mfcc))
            batch_mfcc_pad = batch_mfcc_pad[index]
            lengths_mfcc = lengths_mfcc[index]
            batch_transcript_pad = batch_transcript_pad[index]
            lengths_transcript = lengths_transcript[index]

            # TODO: transforms
            # You may apply some transformation, Time and Frequency masking, here in the collate function;
            # Food for thought -> Why are we applying the transformation here and not in the __getitem__?
            #                  -> Would we apply transformation on the validation set as well?
            #                  -> Is the order of axes / dimensions as expected for the transform functions?
            
            # Return the following values: padded features, padded labels, actual length of features, actual length of the labels
            return batch_mfcc_pad, batch_transcript_pad, lengths_mfcc, lengths_transcript

        else:
            return batch_mfcc_pad, torch.tensor(lengths_mfcc)


# Dataset class for the Toy dataset
class ToyDataset(torch.utils.data.Dataset):

    def __init__(self, partition):
        
        X_train = np.load("f0176_mfccs_train.npy")
        Y_train = np.load("f0176_hw3p2_train.npy")
        X_valid = np.load("f0176_mfccs_dev.npy")
        Y_valid = np.load("f0176_hw3p2_dev.npy")
        
        # VOCAB_MAP = config["stoi"]
        VOCAB_MAP = dict(zip(np.unique(Y_valid), range(len(np.unique(Y_valid))))) 
        VOCAB_MAP["[PAD]"]  = len(VOCAB_MAP)
        
        
        config['stoi'] = VOCAB_MAP
        config['itos'] = {v:k for k,v in VOCAB_MAP.items()}
        config['<pad>'] = VOCAB_MAP["[PAD]"]
        config['<sos>'] = VOCAB_MAP["[SOS]"]
        config['<eos>'] = VOCAB_MAP["[EOS]"]
        

        if partition == "train":
            Y_train = [np.array([VOCAB_MAP[p] for p in seq]) for seq in Y_train]
            self.mfccs = X_train[:, :, :15]
            self.transcripts = Y_train

        elif partition == "valid":
            Y_valid = [np.array([VOCAB_MAP[p] for p in seq]) for seq in Y_valid]
            self.mfccs = X_valid[:, :, :15]
            self.transcripts = Y_valid

        assert len(self.mfccs) == len(self.transcripts)

        self.length = len(self.mfccs)

    def __len__(self):

        return self.length

    def __getitem__(self, i):

        x = torch.tensor(self.mfccs[i])
        y = torch.tensor(self.transcripts[i])

        return x, y

    def collate_fn(self, batch):

        x_batch, y_batch = list(zip(*batch))

        x_lens      = [x.shape[0] for x in x_batch] 
        y_lens      = [y.shape[0] for y in y_batch] 

        x_batch_pad = torch.nn.utils.rnn.pad_sequence(x_batch, batch_first=True, padding_value=config["<pad>"])
        y_batch_pad = torch.nn.utils.rnn.pad_sequence(y_batch, batch_first=True, padding_value=config["<pad>"]) 
        
        return x_batch_pad, y_batch_pad, torch.tensor(x_lens), torch.tensor(y_lens)


     
if __name__ == "__main__":
    dataset = AudioDataset(0,check=True)
    # dataset = ToyDataset("train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=dataset.collate_fn)
    for i, (mfcc, transcript, lengths_mfcc, lengths_transcript) in enumerate(dataloader):
        print(mfcc.shape, transcript.shape, lengths_mfcc)
        break