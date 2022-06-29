from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import sys
import os
import pickle
import argparse
import random
import string
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image

import json

from text_processing import tokenize_text, untokenize, pad_text, Toks

import seq2seq_pytorch as s2s

import seq2emo as s2e

nltk.download('wordnet')
nltk.download('omw-1.4')

cuda = True
device = 0

coco_inception_features_path = "./datasets/coco_train_v3_pytorch.pik"
coco_dataset_path = "./datasets/coco_dataset_full_rm_style.json"
model_path = "./code/models/"
model_save_path = "./code/models_img2txt/"
test_model_fname = "img_to_txt_state.tar"
epoch_to_save_path = lambda epoch: model_save_path+"img_to_txt_state_%d.tar" % int(epoch)    # lambda表达式

BATCH_SIZE=128


class NopModule(torch.nn.Module):
    def __init__(self):
        super(NopModule, self).__init__()
    
    def forward(self, input):
        return input

def get_cnn():  #预训练模型加载、数据预处理
    inception = models.inception_v3(pretrained=True)
    inception.fc = NopModule()
    if cuda:
        inception = inception.cuda(device=device)
    inception.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                normalize,
            ])
    return inception, trans

def has_image_ext(path):    # 判断是否有对应格式图片
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
    ext = os.path.splitext(path)[1]
    if ext.lower() in IMG_EXTENSIONS:
        return True
    return False

def list_image_folder(root):    #展开目录并列举出
    images = []
    dir = os.path.expanduser(root)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if os.path.isdir(d):
            continue
        if has_image_ext(d):
            images.append(d)
    return images

def safe_pil_loader(path, from_memory=False):   #得到RGB
    try:
        if from_memory:
            img = Image.open(path)
            res = img.convert('RGB')
        else:
            with open(path, 'rb') as f:
                img = Image.open(f)
                res = img.convert('RGB')
    except:
        res = Image.new('RGB', (299, 299), color=0)
    return res

class ImageTestFolder(torch.utils.data.Dataset):    #定义数据类
    def __init__(self, root, transform):
        self.root = root
        self.loader = safe_pil_loader
        self.transform = transform

        self.samples = list_image_folder(root)

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(path)
        sample = self.transform(sample)
        return sample, path

    def __len__(self):
        return len(self.samples)

# load images provided across the network
class ImageNetLoader(torch.utils.data.Dataset): #网络中已有图像
    def __init__(self, images, transform):
        self.images = images
        self.loader = safe_pil_loader
        self.transform = transform

    def __getitem__(self, index):
        sample = self.loader(self.images[index], from_memory=True)
        sample = self.transform(sample)
        return sample, ""

    def __len__(self):
        return len(self.images)

def get_image_reader(dirpath, transform, batch_size, workers=4):    #加载dataloader
    image_reader = torch.utils.data.DataLoader(
            ImageTestFolder(dirpath, transform),
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True)
    return image_reader


def get_data(train=True):
    feats = pickle.load(open(coco_inception_features_path,"rb"),encoding='bytes')   #特别注意编码方式，python2和python3不一样

    sents = []
    final_feats = []
    filenames = []
    texts = []
    js = json.load(open(coco_dataset_path, "r"))
    for i, img in enumerate(js["images"]):
        if train and img["extrasplit"] == "val":
            continue
        if (not train) and img["extrasplit"] != "val":
            continue
        if img["filename"].encode() not in feats:
            continue
        if train:
            for sen in img["sentences"]:
                sents.append(sen["rm_style_tokens"])
                final_feats.append(feats[img["filename"].encode()])
                filenames.append(img["filename"])
                texts.append(sen['tokens'])
        else:
            sents.append(img["sentences"][0]["rm_style_tokens"])    #取第一个句子作为GT
            final_feats.append(feats[img["filename"].encode()])
            filenames.append(img["filename"])
            txt = []
            for sen in img["sentences"]:
                txt.append(sen['tokens'])
            texts.append(txt) #对应的语句

    final_feats = np.array(final_feats)

    return final_feats, filenames, sents, texts

class ImgEmb(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ImgEmb, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.mlp = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
    
    def forward(self, input):
        res = self.relu(self.mlp(input))
        return res

class Decoder(nn.Module):   #定义解码器模型
    def __init__(self, input_size, hidden_size, output_size, out_bias=None):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.emb_drop = nn.Dropout(0.5)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.gru_drop = nn.Dropout(0.5)
        self.mlp = nn.Linear(hidden_size, output_size)
        if out_bias is not None:
            out_bias_tensor = torch.tensor(out_bias, requires_grad=False)
            self.mlp.bias.data[:] = out_bias_tensor
        self.logsoftmax = nn.LogSoftmax(dim=2)
    
    def forward(self, input, hidden_in):
        emb = self.embedding(input)
        out, hidden = self.gru(self.emb_drop(emb), hidden_in)
        out = self.mlp(self.gru_drop(out))
        out = self.logsoftmax(out)
        return out, hidden

def build_model(dec_vocab_size, dec_bias = None, img_feat_size = 2048, 
        hid_size=512, loaded_state = None): #定义整体框架
    enc = ImgEmb(img_feat_size, hid_size)
    dec = Decoder(dec_vocab_size, hid_size, dec_vocab_size, dec_bias)
    if loaded_state is not None:
        enc.load_state_dict(loaded_state['enc'])
        dec.load_state_dict(loaded_state['dec'])
    if cuda:
        enc = enc.cuda(device=device)
        dec = dec.cuda(device=device)
    return enc, dec

def build_trainers(enc, dec, loaded_state = None):
    learning_rate = 0.001
    lossfunc = nn.NLLLoss(ignore_index=0)   #忽略标签0
    enc_optim = optim.Adam(enc.parameters(), lr=learning_rate)
    dec_optim = optim.Adam(dec.parameters(), lr=learning_rate)
    if loaded_state is not None:
        enc_optim.load_state_dict(loaded_state['enc_optim'])
        dec_optim.load_state_dict(loaded_state['dec_optim'])
    return enc_optim, dec_optim, lossfunc

def save_state(enc, dec, enc_optim, dec_optim, dec_idx_to_word, dec_word_to_idx, epoch):
    state = {'enc':enc.state_dict(), 'dec':dec.state_dict(),
             'enc_optim':enc_optim.state_dict(), 'dec_optim':dec_optim.state_dict(),
            'dec_idx_to_word':dec_idx_to_word, 'dec_word_to_idx':dec_word_to_idx}
    torch.save(state, epoch_to_save_path(epoch))

def generate(enc, dec, feats, L=20):    #dec
    enc.eval()
    dec.eval()
    with torch.no_grad():
        hid_enc = enc(feats).unsqueeze(0)

        # run the decoder step by step
        dec_tensor = torch.zeros(feats.shape[0], L+1, dtype=torch.long)
        if cuda:
            dec_tensor = dec_tensor.cuda(device=device)
        last_enc = hid_enc
        for i in range(L):
            out_dec, hid_dec = dec.forward(dec_tensor[:,i].unsqueeze(1), last_enc)
            chosen = torch.argmax(out_dec[:,0],dim=1)
            dec_tensor[:, i+1] = chosen
            last_enc = hid_dec
    
    return dec_tensor.data.cpu().numpy()

def setup_test(with_cnn=False): #测试代码
    cnn, trans = get_cnn()
    if not cuda:
        loaded_state = torch.load(model_path+test_model_fname, 
                map_location='cpu')
    else:
        loaded_state = torch.load(model_path+test_model_fname)
    dec_vocab_size = len(loaded_state['dec_idx_to_word'])
    enc,dec = build_model(dec_vocab_size, loaded_state=loaded_state)

    s2s.cuda = cuda
    s2s_data = s2s.setup_test()
    s2e.cuda = cuda
    s2e_data = s2e.setup_test()
    return {'cnn': cnn, 'trans': trans, 'enc':enc, 'dec':dec, 
            'loaded_state':loaded_state, 's2s_data':s2s_data, 's2e_data':s2e_data}

class TestIterator:
    def __init__(self, feats, text, bs = BATCH_SIZE):
        self.feats = feats
        self.text = text
        self.bs = bs
        self.num_batch = feats.shape[0]/bs
        if feats.shape[0] % bs != 0:
            self.num_batch += 1
        self.i = 0

    def __iter__(self):
        self.i = 0
        return self

    def next(self):
        if self.i >= self.num_batch:
            raise StopIteration()   #抛出异常

        s = self.i * self.bs
        e = min((self.i + 1) * self.bs, self.feats.shape[0])
        self.i+=1
        return self.feats[s:e], self.text[s:e]

def rouge1_score(reference, pred):
    rouge_1 = 0
    for grams_reference in reference:
        temp = 0
        ngram_all = len(grams_reference)
        for x in grams_reference:
            if x in pred: temp=temp+1
        rouge_1=max(temp/ngram_all, rouge_1)
    return rouge_1

def test_targets(text_det, text_gt):    #字幕指标
    bleu1 = 0
    bleu3 = 0
    meteor = 0
    rouge = 0
    cider = 0
    spice = 0
    for i in range(len(text_gt)):
        bleu1 += sentence_bleu(text_gt[i], text_det[i], weights = [1,0,0,0])
        bleu3 += sentence_bleu(text_gt[i], text_det[i], weights = [0,0,1,0])
        meteor += meteor_score(text_gt[i], text_det[i])
        rouge += rouge1_score(text_gt[i], text_det[i])
    
    return bleu1, bleu3, meteor, rouge

def test(setup_data, test_folder = None, test_images=None):
    enc = setup_data['enc']
    dec = setup_data['dec']
    cnn = setup_data['cnn']
    trans = setup_data['trans']
    loaded_state = setup_data['loaded_state']
    s2s_data = setup_data['s2s_data']
    s2e_data = setup_data['s2e_data']
    dec_vocab_size = len(loaded_state['dec_idx_to_word'])

    if test_folder is not None:
        # load images from folder
        img_reader = get_image_reader(test_folder, trans, BATCH_SIZE)
        using_images = True
    elif test_images is not None:
        # load images from memory
        img_reader = torch.utils.data.DataLoader(
                ImageNetLoader(test_images, trans),
                batch_size=BATCH_SIZE, shuffle=False,
                num_workers=1, pin_memory=True)
        using_images = True
    else:
        # load precomputed image features from dataset
        feats, filenames, sents, texts = get_data(train=False)
        feats_tensor = torch.tensor(feats, requires_grad=False)
        if cuda:
           feats_tensor = feats_tensor.cuda(device=device)
        img_reader = TestIterator(feats_tensor, sents)
        using_images = False
    
    all_text = []
    all_PAD = []
    filename_test = []
    '''
    slen = len(sents)   #生成文本
    num_batch = int(slen/BATCH_SIZE)
    if slen % BATCH_SIZE != 0:
        num_batch += 1
    
   
    target = {'BLEU1':0,'BLEU3':0,'METEOR':0,'ROUGE':0,'CIDEr':0,'SPICE':0} #指标
    confusion = np.zeros((3,3))
    for i in range(num_batch):

        s = i * BATCH_SIZE
        e = min((i + 1) * BATCH_SIZE, slen)
    
        input = feats_tensor[s:e]
        text_data = sents[s:e]
    '''
    for input, text_data in img_reader:
        
        if using_images:
            if cuda:
                input = input.cuda(device=device)
            with torch.no_grad():
                batch_feats_tensor = cnn(input)
        else:
            batch_feats_tensor = input

        dec_tensor = generate(enc, dec, batch_feats_tensor)

        untok = []
        for j in range(dec_tensor.shape[0]):
            untok.append(untokenize(dec_tensor[j], 
                loaded_state['dec_idx_to_word'], 
                to_text=False))
        
        text = s2s.test(s2s_data, untok)    #生成的风格化字幕，控制只生成中性
        '''
        text_coco = []
        for j in range(e-s):
            text_coco.append(text[j*3+1].split())
        
        bleu1, bleu3, meteor, rouge = test_targets(text, texts[s:e])
        
        target['BLEU1'] += bleu1
        target['BLEU3'] += bleu3
        target['METEOR'] += meteor
        target['ROUGE'] += rouge
        '''
        emo_pad, predicted = s2e.test(s2e_data, text)
        '''
        label = np.ones((3,e-s))
        label[0,:] = 0
        label[2,:] = 2
        label = label.transpose().reshape(1,-1).ravel()
        
        for j in range(e-s):
            if (predicted[3*j:3*(j+1)] == label[3*j:3*(j+1)]).sum() == 3:   #说明生成三种图像字幕均符合
                filename_test.append(filenames[s+j])
        confusion += confusion_matrix(label, predicted, labels = [0, 1, 2])
        '''
        
        for i in range(len(text)):
            if i % 3 == 0:
                if using_images:
                    # text data is filenames
                    print("FN :", text_data[int(i/3)])
                else:
                    # text data is ground truth sentences
                    print("GT :", text_data[int(i/3)])
                print("DET:", ' '.join(untok[int(i/3)]))

            if i%3 == 0:
                print("NEGATIVE:", text[i],"(P,A,D) =",emo_pad[i])
            if i%3 == 1:
                print("COCO:", text[i],"(P,A,D) =",emo_pad[i])
            if i%3 == 2:
                print("POSITIVE:", text[i],"(P,A,D) =",emo_pad[i],"\n")
        all_text.extend(text)
        all_PAD.extend(emo_pad)
    result = {}
    result['text'] = all_text
    result['PAD'] = all_PAD
    '''    
    target['BLEU1'] /= slen
    target['BLEU3'] /= slen
    target['METEOR'] /= slen
    target['ROUGE'] /= slen
    target['CIDEr'] /= slen
    target['SPICE'] /= slen
    #data = {}
    #data['confusion matrix'] = confusion.tolist()
    #json.dump(data, open(model_path+'test_data.json','w'))  #测试集的分类结果
    json.dump(target, open(model_path+'target_data.json','w'))  #图像字幕指标
    '''
    return result

        
def train():

    feats, filenames, sents, texts = get_data(train=True)

    dec_idx_to_word, dec_word_to_idx, dec_tok_text, dec_bias = tokenize_text(sents, vsize = 20000) #sents->texts 改为NIC模型
    dec_padded_text = pad_text(dec_tok_text)    #terms -> vector
    dec_vocab_size = len(dec_idx_to_word)
    
    enc, dec = build_model(dec_vocab_size, dec_bias)
    enc_optim, dec_optim, lossfunc = build_trainers(enc, dec)
    
    feats_tensor = torch.tensor(feats, requires_grad=False)
    dec_text_tensor = torch.tensor(dec_padded_text, requires_grad=False)
    if cuda:
       feats_tensor = feats_tensor.cuda(device=device)
       dec_text_tensor = dec_text_tensor.cuda(device=device) 

    num_batches = int(feats.shape[0] / BATCH_SIZE)

    sm_loss = None
    enc.train()
    dec.train()
    
    result_dict = {}
    epoch_term = []
    loss_term = []
    for epoch in range(0, 13):
        print("Starting New Epoch: %d" % epoch)
        
        order = np.arange(feats.shape[0])
        np.random.shuffle(order)
        del feats_tensor, dec_text_tensor
        if cuda:
            torch.cuda.empty_cache()
        feats_tensor = torch.tensor(feats[order], requires_grad=False)
        dec_text_tensor = torch.tensor(dec_padded_text[order], requires_grad=False)
        if cuda:
           feats_tensor = feats_tensor.cuda(device=device)
           dec_text_tensor = dec_text_tensor.cuda(device=device) 

        for i in range(num_batches):
            s = i * BATCH_SIZE
            e = (i+1) * BATCH_SIZE

            enc.zero_grad()
            dec.zero_grad()

            hid_enc = enc.forward(feats_tensor[s:e]).unsqueeze(0)
            out_dec, hid_dec = dec.forward(dec_text_tensor[s:e,:-1], hid_enc)

            out_perm = out_dec.permute(0, 2, 1)
            loss = lossfunc(out_perm, dec_text_tensor[s:e,1:])  #为什么这里的损失函数是差了一个位置？->因为是GRU，需要由上一时间预测下一时间
            
            if sm_loss is None:
                sm_loss = loss.data
            else:
                sm_loss = sm_loss*0.95 + 0.05*loss.data

            loss.backward()
            enc_optim.step()
            dec_optim.step()
            
            if i % 100 == 0:
                print("Epoch: %.3f" % (i/float(num_batches) + epoch,), "Loss:", sm_loss)
                print("GEN:", untokenize(torch.argmax(out_dec,dim=2)[0,:], dec_idx_to_word))    #生成的
                print("GT:", untokenize(dec_text_tensor[s,:], dec_idx_to_word)) #原始
                print("--------------")
                epoch_term.append(i/float(num_batches) + epoch)
                loss_term.append(float(sm_loss))
                result_dict['Epoch'] = epoch_term
                result_dict['Loss'] = loss_term
                json.dump(result_dict,open(model_save_path + 'result_data.json','w'))    
        save_state(enc, dec, enc_optim, dec_optim, dec_idx_to_word, dec_word_to_idx, epoch)

def run_server(r):
    import werobot

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--test_folder', help="Folder of images to run on")
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()
    args.train = True
    args.test_folder = './emo_material/Positive/'
    global cuda
    if args.cpu:
        cuda=False
    if args.train:
        train()
    else:
        r = setup_test()
        result = test(r, args.test_folder)

        json.dump(result, open(os.path.join(args.test_folder,'result.json'),'w'))
        

if __name__ == "__main__":
    main()
