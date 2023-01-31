import torch
import torch.nn as nn

from modules.attention import Attention, plot_attention
from torchnlp.nn import LockedDropout


class Speller(torch.nn.Module):

    def __init__(self, hidden_size, listener_hidden_size, vocab_size, config=None,max_len=40):
        super().__init__()

        self.config = config
        self.sos = config['<sos>']

        attn_params = {
            "encoder_hidden_size":listener_hidden_size,
            "decoder_output_size":hidden_size,
            "projection_size":512,
            "config":config
        }

        self.attention = Attention(**attn_params)
        self.num_layers = 2
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.embedding = nn.Embedding(
            self.vocab_size, self.hidden_size, padding_idx=config['<pad>'])

        # self.lstm_cells = nn.LSTM(input_size=self.hidden_size + self.attention.hidden_size,
        #                 hidden_size=self.hidden_size, num_layers=self.num_layers,batch_first=True)

        self.lock_drop = nn.Dropout(0.2)
        self.lstm_cells = nn.ModuleList(
            [nn.LSTMCell(input_size=self.hidden_size + self.attention.hidden_size, hidden_size=self.hidden_size),
            nn.LSTMCell(input_size=self.hidden_size, hidden_size=self.hidden_size)]
        )
        
        # mlp
        self.output_to_char = nn.Linear(self.hidden_size + self.attention.hidden_size,self.hidden_size)
        self.activation = nn.Tanh()
        self.char_prob = nn.Linear(self.hidden_size, self.vocab_size)
        self.char_prob.weight = self.embedding.weight  # Weight tying
    
    def mlp(self,x):
        x = self.char_prob(self.activation(self.output_to_char(x)))    
        return x
    

    def forward_step(self, input_word, hidden_state, listner_feature, listner_feature_lens):
        input_word = self.lock_drop(input_word)
        for i in range(len(self.lstm_cells)):
            hidden_state[i] = self.lstm_cells[i](input_word, hidden_state[i])
            input_word = hidden_state[i][0]
        
        rnn_output = hidden_state[-1][0]
        
        attn_c, attn_w = self.attention(
            rnn_output, listner_feature, listner_feature_lens)
        concat_feat = torch.cat([rnn_output, attn_c], dim=-1)
        raw_pred = self.mlp(concat_feat)

        return raw_pred, hidden_state, attn_c, attn_w

    def forward(self, encoder_outputs, encoder_lens, y=None, tf_rate=1):
        '''
        Args: 
            embedding: Attention embeddings 
            hidden_list: List of Hidden States for the LSTM Cells
        '''

        batch_size, encoder_max_seq_len, _ = encoder_outputs.shape

        if self.training and not y is None:
            timesteps = y.shape[1]
            label_embed = self.embedding(y)
        else:
            # 600 is a design choice that we recommend, however you are free to experiment.
            timesteps = self.max_len

        # INITS
        raw_pred_seq = []

        # Initialize the first character input to your decoder, SOS
        char = torch.full((batch_size,), fill_value=self.sos,
                          dtype=torch.long).to(device=self.config['device'])

        # Initialize a list to keep track of LSTM Cell Hidden and Cell Memory States, to None
        hidden_state = [None for _ in range(len(self.lstm_cells))]
        attention_plot = []
        context = torch.zeros(batch_size, self.attention.hidden_size).to(
            device=self.config['device'])

        # # Set Attention Key, Value, Padding Mask just once
        self.attention.set_key_value_mask(encoder_outputs, encoder_lens)

        for t in range(timesteps):

            char_embed = self.embedding(char)  # Embed the character input

            if self.training and t > 0:
                char_embed = label_embed[:, t-1,
                                         :] if torch.rand(1) < tf_rate else char_embed

            # Concatenate the character embedding and context from attention
            rnn_input = torch.cat((char_embed, context), dim=-1)

            raw_pred, hidden_state, context, attention_weights = self.forward_step(
                rnn_input, hidden_state, encoder_outputs, encoder_lens)

            attention_plot.append(attention_weights[0].detach().cpu())
            raw_pred_seq.append(raw_pred)
            char = raw_pred.argmax(dim=-1)

        attention_plot = torch.stack(attention_plot, dim=1)
        raw_pred_seq = torch.stack(raw_pred_seq, dim=1)

        return raw_pred_seq, attention_plot
