import os
import argparse
from config import config

import torch
from model import Network
import pytorch_lightning as pl
from torchsummaryX import summary

from dataloader import get_training_dataloaders, get_toy_dataloader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == "__main__":
    
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=str, required=True, help="name of the run")

    name = parser.parse_args().n
    
    # wandb
    wandb_logger = WandbLogger(project='H4P2', name='postsem'+name)
    
    # checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor='lev_distance',
        dirpath=f'checkpoints/{name}/',
        filename='{epoch:02d}-{lev_distance:.2f}',
        save_top_k=3,
        mode='min',
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    
    # dataloader
    train_dataloader, valid_dataloader = get_training_dataloaders()
    # train_dataloader, valid_dataloader = get_toy_dataloader()

    # model
    model = Network.load_from_checkpoint("checkpoints/first_try/epoch=134-lev_distance=2.36.ckpt",config=config)
    # model = Network(config=config)
    config['device'] = 'cpu'
    summary(model, torch.zeros(32, 100, 15), torch.zeros(32)+15, torch.zeros(32, 30).to(torch.long), 1)    
    config['device'] = 'cuda'
    
    # training
    trainer = pl.Trainer(accelerator='gpu',devices=[1], precision=16,
                         max_epochs=config["max_epochs"], 
                         logger=wandb_logger, 
                         callbacks=[lr_monitor, checkpoint_callback],
                         )
    
    trainer.fit(model, train_dataloader, valid_dataloader)