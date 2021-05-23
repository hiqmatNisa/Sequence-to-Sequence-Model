from torch import nn
import torch
from .lstm import ResLSTMLayer

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        self.dropout=0.5
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn=ResLSTMLayer((enc_hid_dim * 2) + emb_dim + (enc_hid_dim * 2), dec_hid_dim)

        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        
        #self.ln=nn.LayerNorm(dec_hid_dim)
 
        self.dropout_LSTM=nn.Dropout(self.dropout)

        
    def forward(self, input, hidden, cell, encoder_outputs, train, prev_c):
             
        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        input=torch.topk(input,1)[1] #[8,1] 
        
        embedded = self.embedding(input)
        
        embedded = embedded.permute(1, 0, 2)

        #embedded = [1, batch size, emb dim]
        
        a = self.attention(hidden, encoder_outputs)
                
        #a = [batch size, src len]
        
        a = a.unsqueeze(1)
        
        #a = [batch size, 1, src len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        
        c = torch.bmm(a, encoder_outputs)

        #c = [batch size, 1, enc hid dim * 2]
        
        c = c.permute(1, 0, 2)
        
        #c = [1, batch size, enc hid dim * 2]
        prev_c =prev_c.unsqueeze(0) #1, b, enchid*2 # intially the prev_c will be zeros 
        # this prev_att_weights is mention as input-feeding approach in this paper
        
        rnn_input = torch.cat((embedded, c, prev_c), dim = 2)
        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        #unidirectional lstm in decoder
        output, hidden_out = self.rnn(rnn_input, (hidden.unsqueeze(0),cell.unsqueeze(0)))
        hx= self.dropout_LSTM( hidden_out[0])
        cx=self.dropout_LSTM(hidden_out[1])
        output, hidden_out = self.rnn(rnn_input, (hx.unsqueeze(0),cx.unsqueeze(0)))
        hidden = hidden_out[0]
        cell = hidden_out[1]
        

        
        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden

        #assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        c = c.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, c, embedded), dim = 1)) # this is the last linear layer  wich will produces 83 probablities at each time step
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell, c