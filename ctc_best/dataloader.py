

from dataset import AudioDataset
from config import config

from torch.utils.data import DataLoader

def get_training_dataloaders(check=False):
    pin_memory = True
    shuffle = False
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    
    train_dataset = AudioDataset(mode=0, check=check)
    valid_dataset = AudioDataset(mode=1, check=check)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn, 
                                  shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=valid_dataset.collate_fn,
                                  shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
    
    return train_dataloader, valid_dataloader

def get_test_dataloader():
    pin_memory = True
    shuffle = False
    batch_size = 128
    num_workers = config['num_workers']
    
    test_dataset = AudioDataset(mode=2)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.collate_fn,
                                 shuffle=False, pin_memory=pin_memory, num_workers=1)
    
    return test_dataloader

if __name__ == "__main__":
    # test dataloader
    test = get_test_dataloader()
    for i, (mfcc, mfcc_len) in enumerate(test):
        print(mfcc.shape)
        print(mfcc_len)
        break