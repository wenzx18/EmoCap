import copy
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import sys
import random
from collections import Counter
import json
import argparse
from torch.nn.utils.clip_grad import clip_grad_value_

from text_processing import tokenize_text, untokenize, pad_text, Toks

cuda = True
device = 0

text_train_path = "./datasets/dataset_emotion.json"
token_train_path = "./datasets/dataset_token.json"

model_path = "./code/models/"
model_save_path = "./code/models_seq2seq/"
result_path = model_path + 'result_data.json'
seq_to_seq_test_model_fname = "seq_to_txt_state.tar"
epoch_to_save_path = lambda epoch: model_save_path+"seq_to_txt_state_%d.tar" % int(epoch)

BATCH_SIZE=128
POSITIVE_STYLE = "POSITIVETOKEN"
NEGATIVE_STYLE = 'NEGATIVETOKEN'
COCO_STYLE = "MSCOCOTOKEN"

def get_data(train=True, maxlines = -1, test_style = COCO_STYLE):   #还未设置测试集
    input_text = []
    input_rems_text = []

    js = json.load(open(token_train_path, "r"))
    c = 0
    
    for line in js:
        if maxlines > 0 and c == maxlines:
            break
        sent = line['rm_tokens']
        input_rems_text.append(sent)
        text = line['raw']
        input_text.append(text)
        c+=1

    
    return input_text, input_rems_text

class Encoder(nn.Module):   #编码，input为x_term
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        assert hidden_size % 2 == 0

        self.hidden_size = hidden_size
        self.input_size = input_size
        
        self.hidden_init_tensor = torch.zeros(2, 1, int(self.hidden_size/2), requires_grad=True)
        nn.init.normal_(self.hidden_init_tensor, mean=0, std=0.05)  #正态分布，随机化
        self.hidden_init = torch.nn.Parameter(self.hidden_init_tensor, requires_grad=True)  #设置为可被训练的parameter
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.emb_drop = nn.Dropout(0.2)
        self.gru = nn.GRU(hidden_size, int(hidden_size/2), batch_first=True, bidirectional=True)
        self.gru_out_drop = nn.Dropout(0.2)
        self.gru_hid_drop = nn.Dropout(0.3)
        
    def forward(self, input, hidden, lengths):
        emb = self.emb_drop(self.embedding(input))  #将10004个词转化为对应的512维向量表示
        pp = torch.nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True)    #打包
        out, hidden = self.gru(pp, hidden)  #这里是双向GRU
        out = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)[0]  #[1]表示长度
        out = self.gru_out_drop(out)
        hidden = self.gru_hid_drop(hidden)
        return out, hidden
    
    def initHidden(self, bs):
        return self.hidden_init.expand(2, bs, int(self.hidden_size/2)).contiguous()

class DecoderAttn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, out_bias):
        super(DecoderAttn, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.emb_drop = nn.Dropout(0.2)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.gru_drop = nn.Dropout(0.2)
        self.mlp = nn.Linear(hidden_size*2, output_size)
        if out_bias is not None:
            out_bias_tensor = torch.tensor(out_bias, requires_grad=False)
            self.mlp.bias.data[:] = out_bias_tensor
        self.logsoftmax = nn.LogSoftmax(dim=2)
        
        self.att_mlp = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_softmax = nn.Softmax(dim=2)
    
    def forward(self, input, hidden, encoder_outs):
        emb = self.embedding(input)
        out, hidden = self.gru(self.emb_drop(emb), hidden)
        
        out_proj = self.att_mlp(out)
        enc_out_perm = encoder_outs.permute(0, 2, 1)    #交换维度
        e_exp = torch.bmm(out_proj, enc_out_perm)   #tensor的矩阵乘法
        attn = self.attn_softmax(e_exp)
        
        ctx = torch.bmm(attn, encoder_outs)
        
        full_ctx = torch.cat([self.gru_drop(out), ctx], dim=2)
        
        out = self.mlp(full_ctx)
        out = self.logsoftmax(out)
        return out, hidden, attn

def build_model(enc_vocab_size, dec_vocab_size, dec_bias = None, hid_size=512, loaded_state = None):
    enc = Encoder(enc_vocab_size, hid_size)
    dec = DecoderAttn(dec_vocab_size, hid_size, dec_vocab_size, dec_bias)
    if loaded_state is not None:
        enc.load_state_dict(loaded_state['enc'])
        dec.load_state_dict(loaded_state['dec'])
    if cuda:
        enc = enc.cuda(device=device)
        dec = dec.cuda(device=device)
    return enc, dec

def build_trainers(enc, dec, loaded_state=None):
    learning_rate = 0.001
    lossfunc = nn.NLLLoss(ignore_index=0)

    enc_optim = optim.Adam(enc.parameters(), lr=learning_rate)
    dec_optim = optim.Adam(dec.parameters(), lr=learning_rate)
    if loaded_state is not None:
        enc_optim.load_state_dict(loaded_state['enc_optim'])#load_state->loaded_state?
        dec_optim.load_state_dict(loaded_state['dec_optim'])
    return enc_optim, dec_optim, lossfunc

def generate(enc, dec, enc_padded_text, L=20):
    enc.eval()
    dec.eval()
    with torch.no_grad():
        # run the encoder
        order, enc_pp, enc_lengths = make_packpadded(0, enc_padded_text.shape[0], enc_padded_text)
        hid = enc.initHidden(enc_padded_text.shape[0])
        out_enc, hid_enc = enc(enc_pp, hid, enc_lengths)
        
        hid_enc = torch.cat([hid_enc[0,:, :], hid_enc[1,:,:]], dim=1).unsqueeze(0)

        # run the decoder step by step
        dec_tensor = torch.ones((enc_padded_text.shape[0]), L+1, dtype=torch.long) * Toks.SOS
        if cuda:
            dec_tensor = dec_tensor.cuda(device=device)
        last_enc = hid_enc
        for i in range(L):
            out_dec, hid_dec, attn = dec.forward(dec_tensor[:,i].unsqueeze(1), last_enc, out_enc)
            out_dec[:, 0, Toks.UNK] = -np.inf # ignore unknowns
            #out_dec[torch.arange(dec_tensor.shape[0], dtype=torch.long), 0, dec_tensor[:, i]] = -np.inf
            chosen = torch.argmax(out_dec[:,0],dim=1)
            dec_tensor[:, i+1] = chosen
            last_enc = hid_dec
    
    return dec_tensor.data.cpu().numpy()[np.argsort(order)]

def make_packpadded(s, e, enc_padded_text, dec_text_tensor = None):

    text = enc_padded_text[s:e]
    lengths = np.count_nonzero(text, axis=1)    #每个语句中非零长度
    order = np.argsort(-lengths)    #terms长度从大到小排列
    new_text = text[order]
    new_enc = torch.tensor(new_text)
    if cuda:
        new_enc = new_enc.cuda(device=device)

    
    if dec_text_tensor is not None:
        new_dec = dec_text_tensor[s:e][order].contiguous()  #强制拷贝
        leng = torch.tensor(lengths[order])
        if cuda:
            leng.cuda(device=device)
        return order, new_enc, new_dec, leng
    else:
        leng = torch.tensor(lengths[order])
        if cuda:
            leng.cuda(device=device)
        return order, new_enc, leng

def save_state(enc, dec, enc_optim, dec_optim, dec_idx_to_word, dec_word_to_idx, enc_idx_to_word, enc_word_to_idx, epoch):
    state = {'enc':enc.state_dict(), 'dec':dec.state_dict(),
             'enc_optim':enc_optim.state_dict(), 'dec_optim':dec_optim.state_dict(),
            'dec_idx_to_word':dec_idx_to_word, 'dec_word_to_idx':dec_word_to_idx,
            'enc_idx_to_word':enc_idx_to_word, 'enc_word_to_idx':enc_word_to_idx}
    torch.save(state, epoch_to_save_path(epoch))

def setup_test():
    if not cuda:
        loaded_state = torch.load(model_path+seq_to_seq_test_model_fname,
                map_location='cpu')
    else:
        loaded_state = torch.load(model_path+seq_to_seq_test_model_fname,map_location='cuda:0')

    enc_idx_to_word = loaded_state['enc_idx_to_word']
    enc_word_to_idx = loaded_state['enc_word_to_idx']
    enc_vocab_size = len(enc_idx_to_word)

    dec_idx_to_word = loaded_state['dec_idx_to_word']
    dec_word_to_idx = loaded_state['dec_word_to_idx']
    dec_vocab_size = len(dec_idx_to_word)

    enc, dec = build_model(enc_vocab_size, dec_vocab_size, loaded_state = loaded_state)

    return {'enc': enc, 'dec': dec, 'enc_idx_to_word':enc_idx_to_word, 'enc_word_to_idx':enc_word_to_idx,
        'enc_vocab_size':enc_vocab_size, 'dec_idx_to_word': dec_idx_to_word, 
        'dec_word_to_idx': dec_word_to_idx, 'dec_vocab_size':dec_vocab_size}

def test(setup_data, input_seqs = None):
    style = [NEGATIVE_STYLE,COCO_STYLE,POSITIVE_STYLE]
    #style = [COCO_STYLE]
    input_rems_text = []
    if input_seqs is None:
        for test_style in style:
            _, tmp_rems_text = get_data(train = False, test_style=test_style)
            input_rems_text.append(tmp_rems_text)
    else:
        input_rems_text = []
        slen = len(input_seqs)
        for i in range(slen):
            for test_style in style:
                input_rems_text.append(copy.deepcopy(input_seqs[i]))
                input_rems_text[-1].append(test_style)

    _, _, enc_tok_text, _ = tokenize_text(input_rems_text,
        idx_to_word = setup_data['enc_idx_to_word'], word_to_idx = setup_data['enc_word_to_idx'])
    enc_padded_text = pad_text(enc_tok_text)

    dlen = enc_padded_text.shape[0]/len(style)
    num_batch = int(dlen/BATCH_SIZE)
    if dlen % BATCH_SIZE != 0:
        num_batch+=1
    res = []
    for i in range(num_batch):
        dec_tensor = generate(setup_data['enc'], setup_data['dec'], enc_padded_text[i*len(style)*BATCH_SIZE:(i+1)*len(style)*BATCH_SIZE])
        res.append(dec_tensor)

    all_text = []
    res = np.concatenate(res, axis=0)   #将list转为np
    for row in res:
        utok = untokenize(row, setup_data['dec_idx_to_word'], to_text=True)
        all_text.append(utok)
    return all_text

    #for i in xrange(100):
    #    print "IN :", untokenize(enc_padded_text[i], enc_idx_to_word, to_text=True)
    #    print "GEN:", untokenize(dec_tensor[i], dec_idx_to_word, to_text=True), "\n"

def train():

    input_text, input_rems_text = get_data(train = True)    #input_text大小写都有，input_rem_text转为小写+POS或FrameNet

    dec_idx_to_word, dec_word_to_idx, dec_tok_text, dec_bias = tokenize_text(input_text, lower_case=True, vsize=20000)  #原始语句进行tokenize，得到对应的词频统计
    dec_padded_text = pad_text(dec_tok_text)    #dec_tok_text是每个语句对应的数字向量编码,进行padding和裁剪
    dec_vocab_size = len(dec_idx_to_word)

    enc_idx_to_word, enc_word_to_idx, enc_tok_text, _ = tokenize_text(input_rems_text)  #semantic terms 进行tokenize
    enc_padded_text = pad_text(enc_tok_text)
    enc_vocab_size = len(enc_idx_to_word)

    dec_text_tensor = torch.tensor(dec_padded_text, requires_grad=False)
    if cuda:
        dec_text_tensor = dec_text_tensor.cuda(device=device)
    
    enc, dec = build_model(enc_vocab_size, dec_vocab_size, dec_bias = dec_bias)
    enc_optim, dec_optim, lossfunc = build_trainers(enc, dec)
    
    num_batches = int(enc_padded_text.shape[0] / BATCH_SIZE)

    sm_loss = None
    enc.train()
    dec.train()
    result_dict = {}
    epoch_term = []
    loss_term = []
    for epoch in range(0, 13):
        print("Starting New Epoch: %d" % epoch)
        
        order = np.arange(enc_padded_text.shape[0])
        np.random.shuffle(order)
        enc_padded_text = enc_padded_text[order]
        dec_text_tensor.data = dec_text_tensor.data[order]
        
        for i in range(num_batches):
            s = i * BATCH_SIZE
            e = (i+1) * BATCH_SIZE
            
            _, enc_pp, dec_pp, enc_lengths = make_packpadded(s, e, enc_padded_text, dec_text_tensor)

            enc.zero_grad()
            dec.zero_grad()
            
            hid = enc.initHidden(BATCH_SIZE)    #扩展hidden state到整个batch，1->batch_size

            out_enc, hid_enc = enc.forward(enc_pp, hid, enc_lengths)    #enc_pp为packpadded后的terms对应向量，enc_lengths为terms非零长度
            
            hid_enc = torch.cat([hid_enc[0,:, :], hid_enc[1,:,:]], dim=1).unsqueeze(0)  #合并bidirectional隐藏层向量
            out_dec, hid_dec, attn = dec.forward(dec_pp[:,:-1], hid_enc, out_enc)   #dec_pp为packpadded后的解码语句对应向量，除了最后一列

            out_perm = out_dec.permute(0, 2, 1)
            dec_text_tensor.shape
            loss = lossfunc(out_perm, dec_pp[:,1:])
            
            if sm_loss is None:
                sm_loss = loss.data
            else:
                sm_loss = sm_loss*0.95 + 0.05*loss.data

            loss.backward()
            clip_grad_value_(enc_optim.param_groups[0]['params'], 5.0)  #梯度裁剪
            clip_grad_value_(dec_optim.param_groups[0]['params'], 5.0)
            enc_optim.step()
            dec_optim.step()
            
            #del loss
            if i % 100 == 0:
                print("Epoch: %.3f" % (i/float(num_batches) + epoch,), "Loss:", sm_loss)
                print("GEN:", untokenize(torch.argmax(out_dec,dim=2)[0,:], dec_idx_to_word))
                #print "GEN:", untokenize(torch.argmax(out_dec,dim=2)[1,:], dec_idx_to_word)
                print("GT:", untokenize(dec_pp[0,:], dec_idx_to_word))
                print("IN:", untokenize(enc_pp[0,:], enc_idx_to_word))  #取第一个句子示例

                print(torch.argmax(attn[0], dim=1))
                print("--------------")
                epoch_term.append(i/float(num_batches) + epoch)
                loss_term.append(float(sm_loss))
                
                result_dict['Epoch'] = epoch_term
                result_dict['Loss'] = loss_term
                json.dump(result_dict,open(result_path,'w'))    
        save_state(enc, dec, enc_optim, dec_optim, dec_idx_to_word, dec_word_to_idx, enc_idx_to_word, enc_word_to_idx, epoch)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    args = ap.parse_args()
    args.train = True
    if args.train:
        train()
    else:
        r = setup_test()
        test(r)

if __name__ == "__main__":
    main()
