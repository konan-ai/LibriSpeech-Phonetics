import os
import argparse
from config import config

import torch
from model import Network
import pytorch_lightning as pl
from torchsummaryX import summary

from dataloader import get_training_dataloaders
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == "__main__":
    
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=str, required=True, help="name of the run")

    name = parser.parse_args().n
    
    
    # wandb
    wandb_logger = WandbLogger(project='H3P2', name="postsem - " + name)
    
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

    # model
    model = Network.load_from_checkpoint("checkpoints/v8/epoch=695-lev_distance=2.68.ckpt",config=config)
    summary(model, torch.zeros(5, 200, 15), torch.ones(5)*200//4)
    
    # training
    trainer = pl.Trainer(accelerator='gpu',devices=[1], precision=16,
                         max_epochs=config["max_epochs"], 
                         logger=wandb_logger, 
                         callbacks=[lr_monitor, checkpoint_callback],
                         )
    
    trainer.fit(model, train_dataloader, valid_dataloader)