import os
import numpy as np
import cv2
import loadData2_vgg as loadData
import torch

HEIGHT = loadData.IMG_HEIGHT
WIDTH = loadData.IMG_WIDTH
output_max_len = loadData.OUTPUT_MAX_LEN
tokens = loadData.tokens
num_tokens = loadData.num_tokens
vocab_size = loadData.num_classes + num_tokens #1 is added for CTC
index2letter = loadData.index2letter
FLIP = loadData.FLIP
WORD_LEVEL = loadData.WORD_LEVEL

load_data_func = loadData.loadData

def writeGradient(pram, value):
    folder_name = 'pred_logs'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_name = folder_name + '/'+pram+'.log'
    with open(file_name, 'a') as f:
        f.write(str(value.item()))
        f.write(' ')

def visualizeAttn(img, first_img_real_len, attn, epoch, count_n, name):
    folder_name = 'imgs'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    img = img[:, :first_img_real_len]
    img = img.cpu().numpy()
    img -= img.min()
    #img *= 255/img.max()
    img = img.astype(np.uint8)
    weights = [img] # (80, 460)
    #for m in attn[:count_n+1]: # also show the last <EOS>
    for m in attn[:count_n]:
        mask_img = np.vstack([m]*10) # (10, 55)
        #mask_img *= 255/mask_img.max()
        mask_img = mask_img.astype(np.uint8)
        mask_img = cv2.resize(mask_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        weights.append(mask_img)
    output = np.vstack(weights)
    if loadData.FLIP:
        output = np.flip(output, 1)
    cv2.imwrite(folder_name+'/'+name+'_'+str(epoch)+'.jpg', output)

def writePredict(epoch, index, pred, flag): # [batch_size, vocab_size] * max_output_len
    folder_name = 'pred_logs'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_prefix = folder_name+'/'+flag+'_predict_seq.'
    #if flag == 'train':
    #    file_prefix = folder_name+'/train_predict_seq.'
    #elif flag == 'valid':
    #    file_prefix = folder_name+'/valid_predict_seq.'
    #elif flag == 'test':
    #    file_prefix = folder_name+'/test_predict_seq.'

    #pred = pred.detach #shape[94, 8, 83] (maxi_out_put len, batch size, number of classes)
    pred2 = torch.topk(pred,1)[1].squeeze(2) # torch.Size([94, 8])  [1] means we are taking indexex of top probalities
    pred2 = pred2.transpose(0, 1) # torch.Size([8,94])
    pred2 = pred2.cpu().numpy()

    batch_count_n = []
    with open(file_prefix+str(epoch)+'.log', 'a') as f:
        for n, seq in zip(index, pred2):
            f.write(n+' ')
            count_n = 0
            for i in seq:
                if i ==tokens['END_TOKEN']:
                    #f.write('<END>')
                    break
                else:
                    if i ==tokens['GO_TOKEN']:
                        f.write('<GO>')
                    elif i ==tokens['PAD_TOKEN']:
                        f.write('<PAD>')
                    else:
                        f.write(index2letter[i-num_tokens])
                    count_n += 1
            batch_count_n.append(count_n)
            f.write('\n')
    return batch_count_n

def writeLoss(loss_value, flag):
    folder_name = 'pred_logs'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    if flag == 'train':
        file_name = folder_name + '/loss_train.log'
    elif flag == 'valid':
        file_name = folder_name + '/loss_valid.log'
    elif flag == 'test':
        file_name = folder_name + '/loss_test.log'
    with open(file_name, 'a') as f:
        f.write(str(loss_value))
        f.write(' ')


def writePredict_beam(epoch, index, pred, flag): # [batch_size, vocab_size] * max_output_len
    folder_name = 'pred_logs'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_prefix = folder_name+'/'+flag+'_predict_seq_beam.'
    #if flag == 'train':
    #    file_prefix = folder_name+'/train_predict_seq.'
    #elif flag == 'valid':
    #    file_prefix = folder_name+'/valid_predict_seq.'
    #elif flag == 'test':
    #    file_prefix = folder_name+'/test_predict_seq.'

    #pred = pred.detach #shape[94, 8, 83] (maxi_out_put len, batch size, number of classes)
    #pred2 = torch.topk(pred,1)[1].squeeze(2) # torch.Size([94, 8])  [1] means we are taking indexex of top probalities
    #pred2 = pred2.transpose(0, 1) # torch.Size([8,94])
    #pred2 = pred2.cpu().numpy()

    batch_count_n = []
    with open(file_prefix+str(epoch)+'.log', 'a') as f:
        f.write(index+' ')
        for n, seq in zip(index, pred):
            count_n = 0
            for i in seq:
                #print(index2letter[i-num_tokens])
                if i ==tokens['END_TOKEN']:
                    #f.write('<END>')
                    break
                else:
                    if i ==tokens['GO_TOKEN']:
                        f.write('')
                    elif i ==tokens['PAD_TOKEN']:
                        f.write('<PAD>')
                    else:
                        f.write(index2letter[i-num_tokens])
                    count_n += 1
            batch_count_n.append(count_n)
            f.write('\n')
    return batch_count_n
def writePredict_beam2(epoch, index, pred, flag): # [batch_size, vocab_size] * max_output_len
    folder_name = 'pred_logs'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_prefix = folder_name+'/'+flag+'_predict_seq_beamwithoutcrossout3.'

    with open(file_prefix+str(epoch)+'.log', 'a') as f:
        f.write(index+' ')
        for i in pred:
                #print(index2letter[i-num_tokens])
            if i ==tokens['END_TOKEN']:
                    #f.write('<END>')
                break
            else:
                if i ==tokens['GO_TOKEN']:
                    f.write('')
                elif i ==tokens['PAD_TOKEN']:
                    f.write('<PAD>')
                else:
                    f.write(index2letter[i-num_tokens])
                
        f.write('\n')
    