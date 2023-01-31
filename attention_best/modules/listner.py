import torch
from torch import nn
from modules.p_lstm import pBLSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchnlp.nn import LockedDropout

class Listener(torch.nn.Module):
    '''
    The Encoder takes utterances as inputs and returns latent feature representations
    '''
    def __init__(self, input_size, encoder_hidden_size, config):
        super(Listener, self).__init__()
        
        self.input_size = input_size
        self.hs = encoder_hidden_size
        self.embd = 256
        self.config = config
        
        # TODO: have 1D convolutional layer 
        self.fe = nn.Sequential(
            nn.Conv1d(self.input_size, self.embd, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(self.embd),
            nn.GELU(),
            nn.Conv1d(self.embd, self.embd, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(self.embd),
            nn.GELU(),
        )
        
        # self.lockdrop = LockedDropout(0.2)
        # # The first LSTM at the very bottom
        # self.base_lstm = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.hs, 
        #                                bidirectional=True, 
        #                                batch_first=True,
        #                                num_layers=3,dropout=0.3)
        self.drop = nn.Dropout(0.2)

        self.pBLSTMs = torch.nn.ModuleList( # How many pBLSTMs are required?
            # TODO: Fill this up with pBLSTMs - What should the input_size be? 
            # Hint: You are downsampling timesteps by a factor of 2, upsampling features by a factor of 2 and the LSTM is bidirectional)
            # Optional: Dropout/Locked Dropout after each pBLSTM (Not needed for early submission)
            [pBLSTM(self.embd, self.hs),
            pBLSTM(2*self.hs, self.hs),
            pBLSTM(2*self.hs, self.hs),]
        )
         
    def forward(self, x, lx):
        # Where are x and x_lens coming from? The dataloader
        X = x
        lx = lx.cpu()
        # lx = lx//4
        
        X = X.transpose(1,2)
        X = self.fe(X)
        X = X.transpose(1,2)
        
        X = pack_padded_sequence(X, lx, batch_first=True, enforce_sorted=False)
        for i in range(len(self.pBLSTMs)):
            X, lx = self.pBLSTMs[i](X)
        X, _ = pad_packed_sequence(X, batch_first=True)
        X = self.drop(X)
        # TODO: check this dropout

        return X, lx

if __name__ == "__main__":
    encoder = Listener(40, 128)
    print(encoder)
    from torchsummaryX import summary
    
    summary(encoder, torch.zeros(32, 100, 40), torch.zeros(32).int()+100)