import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from config import config
from torchnlp.nn import LockedDropout

class pBLSTM(nn.Module):

    '''
    Pyramidal BiLSTM
    Read the write up/paper and understand the concepts and then write your implementation here.

    At each step,
    1. Pad your input if it is packed (Unpack it)
    2. Reduce the input length dimension by concatenating feature dimension
        (Tip: Write down the shapes and understand)
        (i) How should  you deal with odd/even length input? 
        (ii) How should you deal with input length array (x_lens) after truncating the input?
    3. Pack your input
    4. Pass it into LSTM layer

    To make our implementation modular, we pass 1 layer at a time.
    '''
    
    def __init__(self, input_size, hidden_size):
        self.dropout = 0.3
        pad_token = config['<pad>']
        super(pBLSTM, self).__init__()
        self.downsample_factor = 2
        self.blstm = nn.LSTM(input_size*self.downsample_factor, hidden_size, bidirectional=True, 
                             num_layers=2,
                             batch_first=True,dropout=self.dropout)
        self.lock_drop = LockedDropout(self.dropout)
        
    def forward(self, x_packed): # x_packed is a PackedSequence

        # TODO: Pad Packed Sequence
        x, lx = pad_packed_sequence(x_packed, batch_first=True)
        x, lx = self.trunc_reshape(x, lx)
        x = self.lock_drop(x)
        x = pack_padded_sequence(x, lx, batch_first=True, enforce_sorted=False)
        
        x, _ = self.blstm(x)

        return x, lx

    def trunc_reshape(self, x, x_lens): 
        # TODO: If you have odd number of timesteps, how can you handle it? (Hint: You can exclude them)
        # TODO: Reshape x. When reshaping x, you have to reduce number of timesteps by a downsampling factor while increasing number of features by the same factor
        # TODO: Reduce lengths by the same downsampling factor
        b, t, f = x.shape
        # add padd to make divisible by downsampling factor
        
        if t % self.downsample_factor != 0:
            pad = torch.zeros(b, self.downsample_factor - t % self.downsample_factor, f).to(x.device)
            x = torch.cat([x, pad], dim=1)

        
        # x = torch.cat([x, torch.ones(b, self.downsample_factor - t % self.downsample_factor, f).to(x.device)*pad_token], dim=1)
        x = x.reshape(b, -1, f*self.downsample_factor)
        x_lens = torch.div(x_lens,self.downsample_factor,rounding_mode="floor") + (x_lens % self.downsample_factor != 0)
        
        return x, x_lens