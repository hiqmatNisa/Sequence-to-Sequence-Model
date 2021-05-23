# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 22:13:09 2020

@author: S3670639
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time
import os
import torch.nn.functional as F
import argparse
from encoder_htr import Encoder
from decoder_htr import Decoder
from attention_htr import Attention
from seq2seq_htr import Seq2Seq
from utils import visualizeAttn,writeGradient, writePredict, writePredict_beam2, writeLoss, HEIGHT, WIDTH, output_max_len, vocab_size, FLIP, WORD_LEVEL, load_data_func, tokens



parser = argparse.ArgumentParser(description='seq2seq net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('start_epoch', type=int, help='load saved weights from which epoch')
args = parser.parse_args()

BATCH_SIZE = 16
learning_rate = 0.001
ENC_HID_DIM = 256
DEC_HID_DIM = 256 # model/encoder.py SUM_UP=False: enc:dec = 1:2  SUM_UP=True: enc:dec = 1:1
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
CurriculumModelID = args.start_epoch
EMBEDDING_SIZE = 64 # IAM
MODEL_SAVE_EPOCH =2
TEACHER_FORCING= True

    
def teacher_force_func(epoch):
    if epoch < 50:
        teacher_rate = 0.5
    elif epoch < 150:
        teacher_rate = (50 - (epoch-50)//2) / 100.
    else:
        teacher_rate = 0.
    return teacher_rate

def all_data_loader():
    data_train, data_valid, data_test = load_data_func()
    train_loader = torch.utils.data.DataLoader(data_train,collate_fn=sort_batch, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(data_valid,collate_fn=sort_batch, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test,collate_fn=sort_batch, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, valid_loader, test_loader

def test_data_loader_batch(batch_size_nuevo):
    _, _, data_test = load_data_func()
    test_loader = torch.utils.data.DataLoader(data_test, collate_fn=sort_batch, batch_size=batch_size_nuevo, shuffle=False, num_workers=2, pin_memory=True)
    return test_loader

def sort_batch(batch):
    n_batch = len(batch)
    train_index = []
    train_in = []
    train_in_len = []
    train_out = []
    for i in range(n_batch):
        idx, img, img_width, label = batch[i]
        train_index.append(idx)
        train_in.append(img)
        train_in_len.append(img_width)
        train_out.append(label)
        
    train_index = np.array(train_index)
    train_in = np.array(train_in, dtype='float32')
    train_out = np.array(train_out, dtype='int64')
    train_in_len = np.array(train_in_len, dtype='int64')

    train_in = torch.from_numpy(train_in)
    train_out = torch.from_numpy(train_out)
    train_in_len = torch.from_numpy(train_in_len)

    train_in_len, idx = train_in_len.sort(0, descending=True)
    train_in = train_in[idx]
    train_out = train_out[idx]
    train_index = train_index[idx]
    #print("index 0 len", train_in_len[0])
    #print(train_in_len)
    
    return train_index, train_in, train_in_len, train_out

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
            
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
    


def train(model, train_loader, optimizer, criterion, clip, epoch, teacher_rate):
    model.train()
    
    epoch_loss = 0
    #with autograd.detect_anomaly():
    
    for num, (train_index, train_in, train_in_len, train_out) in enumerate(train_loader):
            
        train_in, train_out = Variable(train_in).cuda(), Variable(train_out).cuda()
        optimizer.zero_grad()
        output_t = model(train_in, train_out, train_in_len, teacher_rate=teacher_rate, train=True)  
        batch_count_n = writePredict(epoch, train_index, output_t, 'train')
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        
        train_label = train_out.permute(1, 0)[1:].contiguous().view(-1)#remove<GO>
        output = output_t.view(-1, vocab_size)
            
            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, train_label)
            
        loss.backward()
     
        #save last layer decoder fc and encoder  gradient
        for name, param in model.named_parameters():
            if name == 'decoder.fc_out.weight':
                writeGradient(name, torch.linalg.norm(param.grad,2))
            if name == 'encoder.rnn.weight_hh_l2':
                writeGradient(name, torch.linalg.norm(param.grad,2))
            if name == 'encoder.rnn.weight_hh_l1':
                writeGradient(name, torch.linalg.norm(param.grad,2))
            if name == 'encoder.rnn.weight_hh_l0':
                writeGradient(name, torch.linalg.norm(param.grad,2))
            if name == 'encoder.layer2.2.weight':
                writeGradient(name, torch.linalg.norm(param.grad,2))
            if name == 'encoder.layer1.2.weight':
                writeGradient(name, torch.linalg.norm(param.grad,2))
            if name == 'encoder.layer0.2.weight':
                writeGradient(name, torch.linalg.norm(param.grad,2))
        
        #clip the gradient norm by 2
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
        optimizer.step()
            
        epoch_loss += loss.item()
            
    epoch_loss /= (num+1)
        
    return epoch_loss

def evaluate(model, valid_loader, criterion, epoch):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for num, (valid_index, valid_in, valid_in_len, valid_out) in enumerate(valid_loader):
            valid_in, valid_out = Variable(valid_in).cuda(), Variable(valid_out).cuda()
            output_e = model(valid_in, valid_out, valid_in_len, teacher_rate=False, train=False)
            
            batch_count_n = writePredict(epoch, valid_index, output_e, 'valid')

            valid_label = valid_out.permute(1, 0)[1:].contiguous().view(-1)#remove<GO>
            output = output_e.view(-1, vocab_size)
            if LABEL_SMOOTH:
                loss = crit(log_softmax(output), valid_label)
            else:
                loss = criterion(output, valid_label)
            #print("valid batch loss", loss.item())

            epoch_loss += loss.item()
        
        epoch_loss /= (num+1)
    return epoch_loss

def test(test_loader, modelID, showAttn=True):
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(HEIGHT, WIDTH ,ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT).cuda()
    dec = Decoder(vocab_size, EMBEDDING_SIZE, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn).cuda()
    model = Seq2Seq(enc, dec,output_max_len, vocab_size).cuda()
    model_file = 'save_weights/seq2seq-' + str(modelID) +'.model'
    print('Loading ' + model_file)
    model.load_state_dict(torch.load(model_file)) #load

    model.eval()
    total_loss_t = 0
    start_t = time.time()
    with torch.no_grad():
        for num, (test_index, test_in, test_in_len, test_out) in enumerate(test_loader):
            #test_in = test_in.unsqueeze(1)
            test_in, test_out = Variable(test_in).cuda(), Variable(test_out).cuda()
            output_t = model(test_in, test_out, test_in_len, teacher_rate=False, train=False)
            batch_count_n = writePredict(modelID, test_index, output_t, 'test')
            #writePredict_beam2(modelID, test_index, output_t, 'test')
            
            test_label = test_out.permute(1, 0)[1:].contiguous().view(-1)#remove<GO>
            output_t = output_t.view(-1, vocab_size) #torch.Size([batch, 94, 83]) it means there are total 94 outputs, for every output we have 83 choices 
  
            loss = F.cross_entropy(output_t, test_label, ignore_index=tokens['PAD_TOKEN'])
            total_loss_t += loss.item()
    
        total_loss_t /= (num+1)
        #writeLoss(total_loss_t, 'test')
        print('    TEST loss=%.3f' % (total_loss_t))
        

def main(train_loader, valid_loader, test_loader):
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(HEIGHT, WIDTH ,ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT).cuda()
    dec = Decoder(vocab_size, EMBEDDING_SIZE, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn).cuda()
    model = Seq2Seq(enc, dec,output_max_len, vocab_size).cuda()
    model.apply(init_weights)
    
    if CurriculumModelID > 0:
     
        model_file = 'save_weights/seq2seq-' + str(CurriculumModelID) +'.model'
        #model_file = 'save_weights/words/seq2seq-' + str(CurriculumModelID) +'.model'
        print('Loading ' + model_file)
        model.load_state_dict(torch.load(model_file)) #load
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokens['PAD_TOKEN'])
    #criterion=FocalLoss()
    
    N_EPOCHS = 150
    CLIP = 2   


    for epoch in range(N_EPOCHS):
        epoch=epoch+CurriculumModelID
        start_time = time.time()
        teacher_rate = teacher_force_func(epoch) if TEACHER_FORCING else False
        train_loss = train(model, train_loader, optimizer, criterion, CLIP, epoch, teacher_rate)
        writeLoss(train_loss, 'train')
        valid_loss = evaluate(model, valid_loader, criterion, epoch)
        writeLoss(valid_loss, 'valid')
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        
        #save model after every 2 epochs
        if epoch%MODEL_SAVE_EPOCH == 0:
            folder_weights = 'save_weights'
            if not os.path.exists(folder_weights):
                os.makedirs(folder_weights)
            torch.save(model.state_dict(), folder_weights+'/seq2seq-%d.model'%epoch)

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print('train loss=%.4f, valid_loss=%.4f, teacher_rate=%.3f' % (train_loss, valid_loss,teacher_rate))


if __name__ == '__main__':
    print(time.ctime())
    train_loader, valid_loader, test_loader = all_data_loader()
    mejorModelID = main(train_loader, valid_loader, test_loader)
    print(time.ctime())
