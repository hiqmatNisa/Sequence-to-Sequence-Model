import torch
from torch import nn
from torch.autograd import Variable
import random
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, output_max_len, vocab_size):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.vocab_size = vocab_size
        self.output_max_len=output_max_len
        
    def forward(self, src, trg, train_in_len, teacher_rate, train=True):
        #train_in, train_out, train_in_len, teacher_rate=0.50, train=True
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        
        batch_size = src.shape[0]
        trg = trg.permute(1, 0)
        trg_len = trg.shape[0]
        #outputs = Variable(torch.zeros(self.output_max_len-1, batch_size, self.vocab_size), requires_grad=True)
        #tensor to store decoder outputs
        outputs = torch.zeros(self.output_max_len-1, batch_size, self.vocab_size).cuda()#.to(torch.float64)
        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden, cell = self.encoder(src, train_in_len, train)

        #first input to the decoder is the <sos> tokens
        input = Variable(self.one_hot(trg[0].data)).to(torch.int64)

        prev_c = Variable(torch.zeros(encoder_outputs.shape[1], encoder_outputs.shape[2]), requires_grad=True).cuda() #b,f
        
        for t in range(0, self.output_max_len-1):
            
            #insert input token embedding, previous hidden state and all encoder hidden states
            #receive output tensor (predictions) and new hidden state
            output, hidden, cell, prev_att_weights= self.decoder(input, hidden, cell, encoder_outputs, train, prev_c)
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_rate
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = Variable(self.one_hot(trg[t+1].data)).to(torch.int64) if train and teacher_force else output.data
        return outputs
    def one_hot(self, src): # src: torch.cuda.LongTensor
        ones = torch.eye(self.vocab_size).cuda()
        return ones.index_select(0, src)