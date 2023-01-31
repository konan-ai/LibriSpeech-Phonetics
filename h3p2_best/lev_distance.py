from pickle import MEMOIZE
from config import config

import Levenshtein


# Use debug = True to see debug outputs
def calculate_levenshtein(h, y, lh, ly, decoder, labels, debug = False):
        
    # TODO: look at docs for CTC.decoder and find out what is returned here
    beam_results, beam_scores, timesteps, out_seq_len = decoder.decode(h, seq_lens=lh)

    batch_size = h.shape[0] 
    distance = 0 # Initialize the distance to be 0 initially

    for i in range(batch_size): 
        # TODO: Loop through each element in the batch
        # calculate distance between y and beam_results[i][0][1]
        y_i = (y[i][:ly[i]]).to(int).tolist()
        b_i = (beam_results[i,0][:out_seq_len[i,0]]).tolist()
        distance += Levenshtein.distance(y_i,b_i) 
        
        
    distance /= batch_size # TODO: Uncomment this, but think about why we are doing this

    return distance

