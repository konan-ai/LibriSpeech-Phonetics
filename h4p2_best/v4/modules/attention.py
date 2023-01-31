import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn


def plot_attention(attention,path=None):
    plt.clf()
    sns.heatmap(attention, cmap='GnBu')
    path = path if path else 'attention.png'
    plt.savefig(path)


class Attention(torch.nn.Module):
    '''
    Attention is calculated using the key, value (from encoder hidden states) and query from decoder.
    Here are different ways to compute attention and context:

    After obtaining the raw weights, compute and return attention weights and context as follows.:

    masked_raw_weights  = mask(raw_weights) # mask out padded elements with big negative number (e.g. -1e9 or -inf in FP16)
    attention           = softmax(masked_raw_weights)
    context             = bmm(attention, value)

    At the end, you can pass context through a linear layer too.

    '''

    def __init__(self,
                 encoder_hidden_size,
                 decoder_output_size,
                 projection_size,
                 config):
        super(Attention, self).__init__()

        self.config = config
        self.hidden_size = projection_size
        self.encoding_projection = nn.Linear(
            encoder_hidden_size, projection_size)
        self.key_proj = nn.Linear(encoder_hidden_size, projection_size)
        self.query_proj = nn.Linear(decoder_output_size, projection_size)
        # Optional : Define an nn.Linear layer which projects the context vector

        self.softmax = nn.Softmax(dim=-1)

    def set_key_value_mask(self, encoder_outputs, encoder_lens):

        _, encoder_max_seq_len, _ = encoder_outputs.shape

        self.listner_feature = self.key_proj(encoder_outputs)
        self.value = self.encoding_projection(encoder_outputs)

        # encoder_max_seq_len is of shape (batch_size, ) which consists of the lengths encoder output sequences in that batch
        # The raw_weights are of shape (batch_size, timesteps)

        # self.padding_mask = (torch.arange(
        #     encoder_max_seq_len, device=encoder_lens.device)[None, :] < encoder_lens[:, None]).bool().to(self.config['device'])

    def forward(self, decoder_output_embedding, encoder_outputs, encoder_lens):
        # key   : (batch_size, timesteps, projection_size)
        # value : (batch_size, timesteps, projection_size)
        # query : (batch_size, projection_size)

        self.query = self.query_proj(decoder_output_embedding)


        raw_weights = torch.bmm(
            self.listner_feature, self.query.unsqueeze(-1)).squeeze(-1) #/(self.key.shape[-1]**0.5)
        
        # masked_raw_weights = raw_weights.masked_fill(~self.padding_mask, -1e4)
        
        # masked_attention_weights = self.softmax(masked_raw_weights)
        masked_attention_weights = self.softmax(raw_weights)

        context = torch.sum(masked_attention_weights.unsqueeze(-1)*self.value, axis=1)

        return context, masked_attention_weights  # Return the context, attention_weights
