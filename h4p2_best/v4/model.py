import torch
import torch.nn as nn
import pytorch_lightning as pl

from modules import Listener, Speller, Attention, plot_attention
from lev_distance import calculate_levenshtein
import wandb
import matplotlib.pyplot as plt


class LAS(torch.nn.Module):
    def __init__(self, input_size,
                 vocab_size,
                 encoder_hidden_size,
                 decoder_hidden_size,
                 config=None):

        super(LAS, self).__init__()

        self.encoder = Listener(input_size, encoder_hidden_size, config)

        decoder_params = {
            "hidden_size":decoder_hidden_size,
            "listener_hidden_size":encoder_hidden_size*2,
            "vocab_size":vocab_size,
            "config": config,
            "max_len": 550
        }
        self.decoder = Speller(**decoder_params)

    def forward(self, x, x_lens, y=None, tf_rate=1):

        encoder_outputs, encoder_lens = self.encoder(
            x, x_lens)  # from Listener

        predictions, attention_weights = self.decoder(
            encoder_outputs, encoder_lens, y, tf_rate)

        return predictions, attention_weights


class Network(pl.LightningModule):
    def __init__(self, config):

        super(Network, self).__init__()

        # input_size = 15
        # encoder_hidden_size = 256
        self.vocab_size = len(config['stoi'])
        # projection_size = 128
        # decoder_hidden_size = 512
        # # decoder_output_size = 128
        # decoder_input_embed = 512
        self.config = config
        self.lr = config['lr']

        params = {
            "input_size": 15,
            "vocab_size": len(config['stoi']),
            "encoder_hidden_size": 512,
            "decoder_hidden_size": 512,
            "config": config,
        }

        self.las = LAS(**params)
        self.flag = 1
        self.tf_rate = 0.95

        self.criterion = nn.CrossEntropyLoss(ignore_index=config['<pad>'],reduction='mean')
        # self.criterion = nn.NLLLoss(ignore_index=config['<pad>'], reduction='none')

    def forward(self, x, x_lens, y=None, tf_rate=1):
        
        x = self.las(x, x_lens, y, tf_rate)

        return x

    def training_step(self, batch, batch_idx):
        x, y, x_lens, y_lens = batch
        y_lens = y_lens.cpu()
        
        y_hat, attention_plots = self(x, x_lens, y, self.tf_rate)
        loss = self.criterion(y_hat.contiguous().view(-1,self.vocab_size), y.view(-1))
        distance = calculate_levenshtein(y_hat, y)

        perplexity = torch.exp(loss)
        self.log('train_distance', distance)
        self.log('teacher_forcing_rate', self.tf_rate)
        self.log('train_loss', loss)
        self.log('train_perplexity', perplexity)
        
        if self.flag:
            if self.current_epoch % 2 == 0 and self.current_epoch > 20:
                self.tf_rate = max(self.tf_rate*0.95,0.5)
            plot_attention(attention_plots,
                       f'logs/attention_plots_t{self.current_epoch}.png')
            self.flag = 0
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, x_lens, y_lens = batch
        y_lens = y_lens.cpu()
        y_hat, attention_plots = self(x, x_lens, y)
        max_length = torch.max(y_lens)
        distance = calculate_levenshtein(y_hat, y)
        y_hat = y_hat[:, :max_length]
        loss = self.criterion(y_hat.contiguous().view(-1,self.vocab_size), y.view(-1))
        perplexity = torch.exp(loss)

        # images = wandb.Image(attention_plots)
        # self.log('attention_plots', images)
        # save attention plots
        
        if not self.flag:
            plot_attention(attention_plots,
                       f'logs/attention_plots_v{self.current_epoch}.png')
            self.flag = 1
            
        

        self.log('val_loss', loss)
        self.log('val_perplexity', perplexity)
        self.log('lev_distance', distance)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=15, verbose=True)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }


if __name__ == '__main__':
    from config import config
    config["device"] = "cpu"
    from torchsummaryX import summary
    # import summarywritter
    from torch.utils.tensorboard import SummaryWriter

    model = Network(config)
    print(model)
    config['device'] = 'cpu'

    summary(model, torch.zeros(32, 100, 15), torch.zeros(
        32)+15, torch.zeros(32, 30).to(torch.long), 1)

    # writter = SummaryWriter()

    # writter.add_graph(model.las, [torch.zeros(32, 100, 15), torch.zeros(
    #     32)+15, torch.zeros(32, 30).to(torch.long), torch.tensor(1)])
