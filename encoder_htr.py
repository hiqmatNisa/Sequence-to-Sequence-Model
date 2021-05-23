from torch import nn
from torch.autograd import Variable
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .lstm import ReverseLSTMLayer, ResLSTMLayer

class Encoder(nn.Module):
    def __init__(self,height, width, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.height= height
        self.enc_hid_dim=enc_hid_dim
        self.width=width
        self.num_layers=2
        self.dropout=dropout
        
        self.layer0 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(3,3),stride =(1,1), padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(16),
                nn.MaxPool2d(2,2),
                )
        self.layer1 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=(3,3),stride =(1,1), padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(2,2),
                )
        self.layer2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=(3,3),stride =(1,1), padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(64),
                
                )
        self.layer3 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=(3,3),stride =(1,1), padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(64),
                )
        self.layer4 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=(3,3),stride =(1,1), padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(2,1),
                )
        self.layer5 = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=(3,3),stride =(1,1), padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(2,1),
                )
        self.layer6 = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=(2,2),stride =(1,1), padding=0),
                nn.LeakyReLU(),
                nn.BatchNorm2d(128),
                )
        self.forwardlstm=ResLSTMLayer(5*128, self.enc_hid_dim)
        self.backwardlstm=ReverseLSTMLayer(5*128, self.enc_hid_dim)
        
        
        self.fc = nn.Linear(self.enc_hid_dim * 2, dec_hid_dim)
        
        #self.ln=nn.LayerNorm(self.enc_hid_dim)
 
        self.dropout_LSTM=nn.Dropout(self.dropout)
        
        
    def forward(self, src, in_data_len, train):
        batch_size = src.shape[0]
        #src = [src len, batch size]
        out = self.layer0(src)  
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)

        # from last layer (out) of CNN we will get torch.Size([batch, channel, h, w])
        #we want to reshape it to make it compatiable for RNN
        out = out.permute(3, 0, 2, 1) # (width, batch, height, channels)
        out.contiguous()
        out = out.reshape(-1, batch_size, 5*128) #(w,batch, (height, channels))
        
        #initial the hidden state and cell state of LSTM
        hx = torch.zeros(batch_size, self.enc_hid_dim).to('cuda:0') # (batch, hidden_size)
        cx = torch.zeros(batch_size, self.enc_hid_dim).to('cuda:0')
        torch.nn.init.kaiming_normal_(hx)
        torch.nn.init.kaiming_normal_(cx)
        
        #give input to forward and backward LSTM
        _, hidden_forwd1= self.forwardlstm(out,(hx,cx))
        _, hidden_backwd1= self.backwardlstm(out,(hx,cx))
        
        #hidden_forwd1[0] contain hidden state and hidden_forwd1[1] contain cell state, both are having same shape of [batch size, hidden dimension]
        #here we are contacting from backward and forward and pass them to linear layer
        #beacuse 256 unit + 256 unit become 512. but our next LSTM layer takes 256 units only so that why we want to make it 256 by passing them to linear layer
        hx= torch.tanh(self.fc(torch.cat((hidden_forwd1[0], hidden_backwd1[0]),1))) 
        cx= torch.tanh(self.fc(torch.cat((hidden_backwd1[1], hidden_forwd1[1]),1)))
        if train:
            hx = self.dropout_LSTM(hx)
            cx = self.dropout_LSTM(cx)
        
        #2nd LSTM layer same process will be done here
        outforwd2, hidden_forwd2= self.forwardlstm(out,(hx,cx))
        outbackwd2, hidden_backwd2= self.backwardlstm(out,(hx,cx))
        output= torch.cat((outforwd2, outbackwd2), 2)
        hx= torch.tanh(self.fc(torch.cat((hidden_forwd2[0], hidden_backwd2[0]),1))) 
        cx= torch.tanh(self.fc(torch.cat((hidden_backwd2[1], hidden_forwd2[1]),1))) #this is referes to W in the paper 
        # but we have to pass both hx and cx to linear layers to beacuse decoder initiliaztion is taking two values, cx and hx 
        
        return output, hx, cx