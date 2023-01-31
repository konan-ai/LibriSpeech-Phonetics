from pickle import MEMOIZE
from config import config
import torch

import Levenshtein


# We have given you this utility function which takes a sequence of indices and converts them to a list of characters
def indices_to_chars(indices):
    SOS_TOKEN = config['<sos>']
    EOS_TOKEN = config['<eos>']
    tokens = []
    for i in indices: # This loops through all the indices
        if int(i)== SOS_TOKEN: # If SOS is encountered, dont add it to the final list
            continue
        elif int(i) == EOS_TOKEN: # If EOS is encountered, stop the decoding process
            break
        else:
            tokens.append(i)
    return tokens


# Use debug = True to see debug outputs
def calculate_levenshtein(y_hat,y):
        
    # TODO: look at docs for CTC.decoder and find out what is returned here

    batch_size = y.shape[0] 
    distance = 0 # Initialize the distance to be 0 initially

    for i in range(batch_size): 
        # TODO: Loop through each element in the batch
        # calculate distance between y and beam_results[i][0][1]
        y_i = indices_to_chars(y[i].tolist())
        b_i = indices_to_chars(torch.argmax(y_hat[i],axis=-1).tolist())
        distance += Levenshtein.distance(y_i,b_i) 
        
        
    distance /= batch_size # TODO: Uncomment this, but think about why we are doing this

    return distance

