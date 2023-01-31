import pytorch_lightning as pl
import torch.nn as nn
import torch

from lev_distance import calculate_levenshtein
from ctcdecode import CTCBeamDecoder
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class Network(pl.LightningModule):

    def __init__(self, config):
        super().__init__()

        self.config = config
        feature_size = 256
        hidden_size = 256
        out_size = len(config["stoi"].keys())

        self.criterion = nn.CTCLoss(blank=config['blank'])
        self.decoder = CTCBeamDecoder(
            config['labels'], beam_width=2, log_probs_input=True)

        # Adding some sort of embedding layer or feature extractor might help performance.
        # self.embedding = ?
        self.fe = nn.Sequential(
            nn.Conv1d(15, feature_size, 5, 2, 2),
            nn.BatchNorm1d(feature_size),
            nn.GELU(),
            nn.Conv1d(feature_size, feature_size, 5, 2, 2),
            nn.BatchNorm1d(feature_size),
            nn.GELU()
        )

        # TODO : look up the documentation. You might need to pass some additional parameters.
        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=hidden_size, num_layers=4,
                            batch_first=True, dropout=0.3, bidirectional=True)
        
        self.dropout_lstm = nn.Dropout(0.3)

        self.classification = nn.Sequential(
            # TODO: Linear layer with in_features from the lstm module above and out_features = OUT_SIZE
            nn.Linear(2*hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, x, lx):
        # TODO
        # The forward function takes 2 parameter inputs here. Why?
        # Refer to the handout for hints
        # lx is the length of the input sequence. You might need to use this.
        X = x
        X = X.transpose(1, 2)
        X = self.fe(X)
        X = X.transpose(1, 2)

        X = pack_padded_sequence(
            X, lx.cpu(), enforce_sorted=False, batch_first=True)
        X, h = self.lstm(X)
        X, _ = pad_packed_sequence(X, batch_first=True)
        X = self.dropout_lstm(X)

        X = self.classification(X)
        X = nn.LogSoftmax(2)(X)
        return X

    def training_step(self, batch, batch_idx):

        mfcc, transcript, mfcc_len,  transcript_len = batch
        probs = self(mfcc, mfcc_len)
        probs = probs.transpose(0, 1)
        loss = self.criterion(probs, transcript, mfcc_len, transcript_len)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        mfcc, transcript, mfcc_len,  transcript_len = batch
        probs = self(mfcc, mfcc_len)
        loss = self.criterion(probs.transpose(
            0, 1), transcript, mfcc_len, transcript_len)
        distance = calculate_levenshtein(
            probs, transcript, mfcc_len, transcript_len, self.decoder, self.config['stoi'])
        self.log('lev_distance', distance)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=15, verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "lev_distance"
        }


if __name__ == "__main__":
    from config import config
    import torch
    from torchsummaryX import summary
    model = Network(config)
    print(model)
    summary(model, torch.zeros((64, 200, 15)), torch.ones((64))*200)
