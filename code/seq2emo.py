from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import json
import random
import pandas as pd
import argparse

from text_processing import tokenize_text, untokenize, pad_text
dataset_path = 'datasets/dataset_emotion.json'
coco_train_path = "datasets/coco_dataset_full_rm_style.json"
emodata_path = 'datasets/dataset_emotion.json'
model_path = "./code/models/"
model_save_path = 'code/models_seq2emo/'
epoch_to_save_path = lambda epoch: model_save_path + 'txt_to_emo_state_%d.tar' % int(epoch)
result_path = model_path + 'result_data.json'
test_path = model_path + 'test_data.json'
txt_to_emo_test_model_fname = 'txt_to_emo_state.tar'

BATCH_SIZE = 128
POSITIVE_STYLE = "POSITIVETOKEN"
NEGATIVE_STYLE = 'NEGATIVETOKEN'
COCO_STYLE = "MSCOCOTOKEN"
rate = 0.01 #测试集、验证集
emotion = [NEGATIVE_STYLE, COCO_STYLE, POSITIVE_STYLE]
cuda = True
device = 1

def get_data(train = True,maxlines = -1):
    
    input_text = []
    input_label = []
    '''
    js = json.load(open(dataset_path,'r'))
    c = 0
    for sen in js:
        input_text.append(sen['tokens'])
        if sen['rm_tokens'][-1] == POSITIVE_STYLE:
            input_label.append(2)
        if sen['rm_tokens'][-1] == NEGATIVE_STYLE:
            input_label.append(0)
        c += 1
        if maxlines >0 and c == maxlines:
            break
    
    lenrm = int(c/2)

    c = 0
    tmp_text = []
    tmp_label = []
    js = json.load(open(coco_train_path, "r"))
    for i, img in enumerate(js["images"]):
        if img["extrasplit"] == "val":
            continue
        if maxlines > 0 and c == maxlines:
            break
        for sen in img["sentences"]:
            tmp_label.append(1)
            c+=1
            if maxlines > 0 and c == maxlines:
                break
            tmp_text.append(sen["tokens"])

    # downsample
    random.seed(0)
    input_text.extend(random.sample(tmp_text,lenrm))
    random.seed(0)
    input_label.extend(random.sample(tmp_label,lenrm))
    '''
    js = json.load(open(emodata_path,'r'))
    c = 0
    for sent in js:
        if train ==True:
            if sent["extrasplit"] == 'test':
                continue
        else:
            if sent['extrasplit'] in ['train','val']:
                continue
        if maxlines > 0 and c == maxlines:
            break
    
        input_label.append(sent['label'])
        input_text.append(sent['tokens'])
        c+=1
    
    return input_text, input_label

def get_file(input_text, input_label):
    data = []
    L = len(input_label)
    order = np.arange(L)
    np.random.seed(0)
    np.random.shuffle(order)
    val = order[:int(rate*L)]
    test = order[int(rate*L):2*int(rate*L)]
    train = order[2*int(rate*L):]
    for i in range(len(input_label)):
        
        line = {}
        line['tokens'] = input_text[i]
        line['label'] = input_label[i]
        if i in val:
            line['extrasplit'] = 'val'
        if i in test:
            line['extrasplit'] = 'test'
        if i in train:
            line['extrasplit'] = 'train'
        data.append(line)
    
    json.dump(data,open(emodata_path,'w'))

    return data

def get_emotion_text(emotion = 'happiness'):
    input_text, _ = get_data()
    emotion_text = []
    emotion_dict = {
    'happiness':['happy','happiness','delight','delightful','delighted','joy','joyful','cheer','cheerful'],
    'sadness':['sad','sadness','sorrow','sorrowful','distress','distressing','distressed','woe','woeful','woefully','heartbroken'],
    'fear':['fear','fearful','fearsome','scare','scared','terrify','terrified','afeared','afeard','apprehensive'],
    'disgust':['disgust','disgusting','hate','detest','aversion','abhorrent','sick','disagreeable','irksome','repugnant'],
    'anger':['anger','angry','indignant','indignation','rage','wrath','wrathful','bile','furious','furiously','resent','resentful'],
    'suprise':['suprise','suprised','suprising','amaze','amazed','amazing','amazement','astonished','astonishment','astonished','astonishing','astound','astounded','astounding']}

    for sen in input_text:
        for emo_word in emotion_dict[emotion]:
            if emo_word in sen:
                emotion_text.append(sen)
                break
    
    return emotion_text

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden_init_tensor = torch.zeros(1, 1, self.hidden_size, requires_grad = True)
        nn.init.normal_(self.hidden_init_tensor, mean=0, std=0.05)
        self.hidden_init = torch.nn.Parameter(self.hidden_init_tensor, requires_grad=True)  #设置为可被训练的parameter


        self.embedding = nn.Embedding(input_size, hidden_size)
        self.emb_drop = nn.Dropout(0.2)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first = True)
        self.gru_drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, int(hidden_size/8))
        self.fc2 = nn.Linear(int(hidden_size/8),output_size)
        self.relu = nn.ReLU()

    def forward(self, input, hidden_in):
        emb = self.embedding(input)
        gru_out, hidden = self.gru(self.emb_drop(emb), hidden_in)
        gru_out = self.gru_drop(gru_out[:,-1,:])
        out = self.fc2(self.relu(self.fc1(gru_out)))
        return out, hidden, gru_out

    def initHidden(self, bs):
        return self.hidden_init.expand(1, bs, self.hidden_size).contiguous()


def build_model(dec_vocab_size,
        hid_size=512, out_size = 3, loaded_state = None): #定义整体框架
    dec = Decoder(dec_vocab_size, hid_size, out_size)
    if loaded_state is not None:
        dec.load_state_dict(loaded_state['dec'])
    if cuda:
        dec = dec.cuda(device=device)
    return dec

def build_trainers(dec, loaded_state=None):
    learning_rate = 0.001
    lossfunc = nn.CrossEntropyLoss()

    dec_optim = optim.Adam(dec.parameters(), lr=learning_rate)
    if loaded_state is not None:
        dec_optim.load_state_dict(loaded_state['dec_optim'])
    return dec_optim, lossfunc

def save_state(dec, dec_optim, dec_idx_to_word, dec_word_to_idx, epoch):
    state = {'dec':dec.state_dict(), 'dec_optim':dec_optim.state_dict(),
    'dec_idx_to_word':dec_idx_to_word, 'dec_word_to_idx':dec_word_to_idx}
    torch.save(state, epoch_to_save_path(epoch))

def generate(dec, enc_text):
    dec.eval()
    with torch.no_grad():
        enc_text_tensor = torch.tensor(enc_text)        
        if cuda:
            enc_text_tensor = enc_text_tensor.cuda(device = device)
        hid = dec.initHidden(len(enc_text))
        out_dec, hid_dec,gru_out = dec.forward(enc_text_tensor, hid)
    
    return out_dec, hid_dec, gru_out  #语义向量

def setup_test():
    if not cuda:
        loaded_state = torch.load(model_path+txt_to_emo_test_model_fname,
                map_location='cpu')
    else:
        loaded_state = torch.load(model_path+txt_to_emo_test_model_fname,map_location='cuda:1')

    dec_idx_to_word = loaded_state['dec_idx_to_word']
    dec_word_to_idx = loaded_state['dec_word_to_idx']
    dec_vocab_size = len(dec_idx_to_word)
    dec = build_model(dec_vocab_size,loaded_state=loaded_state)

    return {'dec': dec, 'dec_idx_to_word': dec_idx_to_word, 
    'dec_word_to_idx': dec_word_to_idx, 'dec_vocab_size':dec_vocab_size}



def test(setup_data, input_seqs = None):
    emotion_pad = json.load(open('code/emotion_pad.json','r'))
    style = [NEGATIVE_STYLE,COCO_STYLE,POSITIVE_STYLE]
    input_text = []
    
    for sen in input_seqs:
        input_text.append(sen.split())
    slen = len(input_seqs)
    # 生成pad词典时用
    #input_text = input_seqs
    #slen = len(input_seqs)
    num_batch = int(slen/BATCH_SIZE)
    if slen % BATCH_SIZE != 0:
        num_batch+=1
    _, _, dec_tok_text, _ = tokenize_text(input_text,
        idx_to_word = setup_data['dec_idx_to_word'], word_to_idx = setup_data['dec_word_to_idx'])
    dec_padded_text = pad_text(dec_tok_text)
    emo_vec = torch.zeros(1,512)
    all_pad = []
    
    emo_size = len(emotion_pad.keys())
    sim = np.zeros((emo_size,1),dtype = 'float32')
    R = np.zeros((emo_size,3),dtype='float32')
    for i in range(num_batch):
        dec_out, _, dec_gru = generate(setup_data['dec'], dec_padded_text[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
        #dec_gru = generate(setup_data['dec'], dec_padded_text[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
        #for j in range(dec_gru.shape[0]):
        #    emo_vec += dec_gru[j].cpu()
        predicted = torch.argmax(dec_out,dim=1).cpu().numpy()
        for vec in dec_gru.cpu().numpy():
            for i,emo in enumerate(emotion_pad):
                a = np.array(emotion_pad[emo]['vector'],dtype = 'float32')
                b = np.array(vec,dtype = 'float32')
                ma = np.linalg.norm(a)
                mb = np.linalg.norm(b)
                #bug
                sim[i] = np.dot(a,b)/(ma*mb)
                R[i] = np.array(emotion_pad[emo]['PAD'],dtype='float32')
            # 公式计算
            idx = np.argsort(-sim.transpose()).transpose()
            pad = sim[idx[0]]*R[idx[0]]
            tmp = 1
            for i in range(emo_size-1):
                tmp *= 1-sim[idx[i]]
            for j in range(emo_size-1):
                pad += tmp*sim[idx[j+1]]*R[idx[j+1]]
            all_pad.append(pad.ravel().tolist())
    return all_pad, predicted#emo_vec/slen
   

def test_emo():
    input_text, input_label = get_data(train=False)
    s2e_data = setup_test()
    _, _, input_tok_text, _ = tokenize_text(input_text,lower_case = True,idx_to_word = s2e_data['dec_idx_to_word'],word_to_idx=s2e_data['dec_word_to_idx'])
    input_padded_text = pad_text(input_tok_text)
    slen = len(input_text)
    num_batch = int(slen/BATCH_SIZE)
    if slen % BATCH_SIZE != 0:
        num_batch+=1
    data = {}
    confusion = np.zeros((len(emotion),len(emotion)))
    for i in range(num_batch):
        s = i * BATCH_SIZE
        e = (i+1) * BATCH_SIZE
        dec_out, _, _ = generate(s2e_data['dec'], input_padded_text[s:e])
        predicted = torch.argmax(dec_out,dim=1).cpu().numpy()
        confusion += confusion_matrix(input_label[s:e],predicted,labels = [0,1,2])
        
    data['confusion matrix'] = confusion.tolist()
    json.dump(data, open(test_path,'w')) 


def train():
    input_text, input_label = get_data()

    dec_idx_to_word, dec_word_to_idx, dec_tok_text, dec_bias = tokenize_text(input_text, lower_case = True, vsize = 20000)

    dec_padded_text = pad_text(dec_tok_text)
    dec_vocab_size = len(dec_idx_to_word)
    class_size = len(emotion)

    dec_text_tensor = torch.tensor(dec_padded_text, requires_grad=False)
    input_label = torch.tensor(input_label,requires_grad=False)
    if cuda:
        dec_text_tensor = dec_text_tensor.cuda(device=device)
        input_label = input_label.cuda(device=device)

    dec = build_model(dec_vocab_size, out_size = class_size)
    dec_optim, lossfunc = build_trainers(dec)

    num_batches = int(dec_padded_text.shape[0] / BATCH_SIZE)

    sm_loss = None
    dec.train()
    result_dict = {}
    epoch_term = []
    loss_term = []
    acc_term = []
    for epoch in range(0,13):
        acc = 0
        print("Starting New Epoch: %d" % epoch)
        
        order = np.arange(dec_padded_text.shape[0])
        np.random.shuffle(order)
        dec_text_tensor.data = dec_text_tensor.data[order]
        input_label.data = input_label.data[order]
        for i in range(num_batches):
            s = i * BATCH_SIZE
            e = (i+1) * BATCH_SIZE

            dec.zero_grad()

            hid = dec.initHidden(BATCH_SIZE)

            out_dec, hid_dec,gru_out = dec.forward(dec_text_tensor[s:e], hid)
            out_emo = torch.argmax(out_dec,dim=1)
            acc += (out_emo == input_label[s:e]).sum().item()
            loss = lossfunc(out_dec, input_label[s:e])

            if sm_loss is None:
                sm_loss = loss.data
            else:
                sm_loss = sm_loss*0.95 + 0.05*loss.data

            loss.backward()
            dec_optim.step()

            if i % 100 == 0:
                print("Epoch: %.3f" % (i/float(num_batches) + epoch,), " Loss:", sm_loss,"Accuracy:%.2f%%" % (100*acc/(i+1)/BATCH_SIZE))
                print("Emotion:",emotion[input_label[s]]," GEN:", emotion[torch.argmax(out_dec,dim=1)[0]])    #生成的
                print("IN:", untokenize(dec_text_tensor[s,:], dec_idx_to_word)) #原始
                
                print("--------------")

                epoch_term.append(i/float(num_batches) + epoch)
                loss_term.append(float(sm_loss))
                acc_term.append(100*acc/(i+1)/BATCH_SIZE)

                result_dict['Epoch'] = epoch_term
                result_dict['Loss'] = loss_term
                result_dict['Accuracy'] = acc_term
                json.dump(result_dict, open(result_path,'w'))

        save_state(dec, dec_optim, dec_idx_to_word, dec_word_to_idx, epoch)


def get_emotion_dict():
    emotion_pad = {
    'happiness':{'PAD':[0.6925,0.3025,0.355]},
    'sadness':{'PAD':[-0.2225,0.0425,-0.175]},
    'fear':{'PAD':[-0.2325,0.325,-0.16]},
    'disgust':{'PAD':[-0.45,0.1,0.1675]},
    'anger':{'PAD':[-0.495,0.275,0.15]},
    'suprise':{'PAD':[0.43,0.4275,0.055]}}
    s2e_data = setup_test()
    for emo in emotion_pad.keys():
        emo_text = get_emotion_text(emo)
        emotion_pad[emo]['vector'] = test(s2e_data, emo_text).numpy().tolist()
        emotion_pad[emo]['sentences'] = len(emo_text)
    
    json.dump(emotion_pad,open('code/emotion_pad.json','w'))

    return emotion_pad

def test_caption():
    df = pd.read_excel('result.xlsx')
    slice = list(range(8,72))
    data=df.iloc[:,slice].values#读取指定列的所有行数据：读取第一列所有数据
    data_sub1 = []
    data_sub2 = []
    score1 = []
    score2 = []
    tmp = data[0].reshape((16,-1))
    for k in tmp:
        data_sub1.append(k[0])
        score1.append(k[1:].tolist())
    tmp = data[1].reshape((16,-1))
    for k in tmp:
        data_sub2.append(k[0])
        score2.append(k[1:].tolist())
    r = setup_test()
    pad1, _ = test(r, data_sub1)
    pad2, _ = test(r, data_sub2)

    pad = json.load(open('emo_material/0-Negative/result.json','r'))
    score1 = np.array(score1)
    score2 = np.array(score2)
    pad1 = np.array(pad1)
    pad2 = np.array(pad2)
    for i in range(len(pad1)):
        tmp1 = 0
        tmp2 = 0
        for k in range(3):
            tmp1 += score1[i][k]*(np.array(pad['PAD'][i*3+k])-pad1[i]/15)
            tmp2 += score2[i][k]*(np.array(pad['PAD'][i*3+k])-pad2[i]/15)
        
        pad1[i] += tmp1
        pad2[i] += tmp2
    
    result = {}
    result['sub1'] = pad1.tolist()
    result['sub2'] = pad2.tolist()

    json.dump(result,open('pad.json','w'))
    return pad1, pad2



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    args = ap.parse_args()
    if args.train:
        train()
    else:
        r = setup_test()
        test(r)
    #train()
    #test_emo()
    #emotion_pad = get_emotion_dict()
    #input_text, input_label = get_data()
    #data = get_file(input_text, input_label)
    #s2e_data = setup_test()

    #test(s2e_data)
    pad1, pad2 = test_caption()


if __name__ == '__main__':
    main()