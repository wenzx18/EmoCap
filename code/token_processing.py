import csv
import enum
import json
from collections import Counter
from sklearn import datasets
import spacy
import nltk
import random
from text_processing import tokenize_text
#from nltk.corpus import stopwords
from nltk.corpus import framenet as fn

emotion_train_path = 'datasets/dataset_emotion.json'
coco_train_path = 'datasets/coco_dataset_full_rm_style.json'
POSITIVE_STYLE = "POSITIVETOKEN"
NEGATIVE_STYLE = 'NEGATIVETOKEN'
COCO_STYLE = "MSCOCOTOKEN"

nlp = spacy.load("en_core_web_sm")
#tc = Counter()

'''
with open(emotion_train_path) as f:
    f_csv = csv.reader(f)
    data = []
    for row in f_csv:
        sen = {}
        sen['sentiment'] = row[0]
        sen['tokens'] = row[5]
        data.append(sen)
'''

def tokenize(text):
    return [tok.text for tok in nlp(text)]

def tokenize_lemma(text):
    return [tok.lemma_ for tok in nlp(text)]

def preprocessd(text):   #nlp类型
    tokens_raw = tokenize(text)
    tokens_lemma = tokenize_lemma(text) #词干化
    tokens_POS = [tok.pos_ for tok in nlp(text)]    #词性标注
    tokens = []

    POS_list = ['PRON','PROPN','CONJ','CCONJ','ADP','ADJ','ADV','PUNCT','DET','INTJ','X','SPACE','SYM','SCONJ']
    for i in range(len(tokens_POS)):
        if tokens_POS[i] in POS_list:   #删除词类型
            continue
        if tokens_raw[i][0] == '@':
            continue
        tok = tokens_lemma[i].lower() + tokens_POS[i]*3
        tokens.append(tok)
    
    return tokens

def tokenize_framenet(toks,framelist):
    tokens = []
    for tok in toks:
        if 'VERB' in tok:   #需要替换的动词，小写
            tok = tok[:-12]
            try:
                fn.frames_by_lemma(r'(?i)'+tok)
            except:
                continue
            else:
                fnlist = fn.frames_by_lemma(r'(?i)'+tok)
                
            if len(fnlist) == 0:
                continue
            idx_min = len(framelist)
            for fnvalue in fnlist:
                if fnvalue.name not in framelist:
                    continue
                idx = framelist.index(fnvalue.name)
                idx_min = min(idx,idx_min)
                if idx_min == idx:
                    tok = 'FRAMENET'+fnvalue.name
            if tok[0:8] == 'FRAMENET':
                tokens.append(tok)
        else:
            tokens.append(tok)
    return tokens

def get_data():
    input_text = []
    js = json.load(open(coco_train_path, "r"))
    for i, img in enumerate(js["images"]):
        
        for sen in img["sentences"]:
            input_text.append(sen["tokens"])
    
    return input_text

def get_FrameNet():
    
    tc = Counter()
    framelist = []
    framenet = []
    js = json.load(open(coco_train_path, "r"))
    for i, img in enumerate(js["images"]):
        for sen in img["sentences"]:
            for tok in sen['rm_style_tokens']:
                if 'FRAMENET' in tok:
                    framelist.append(tok)
    
    tc.update(framelist)
    mc = tc.most_common()
    for tok in mc:
        if tok[1]<200:
            break
        framenet.append(tok[0][8:])
    return framenet
def get_data():

    input_rems_text = json.load(open('datasets/data2.json','r'))
    js = json.load(open('datasets/coco_dataset_full_rm_style.json','r'))    
    lenrm = 193882
    c = 0
    tmp_rems_text = []    
    for i, img in enumerate(js["images"]):
        if img["extrasplit"] == "val":
            continue
        
        for sen in img["sentences"]:
            
            line = {'raw':[],'rm_tokens':[]}    
            line['raw'] = sen['tokens']
            line['rm_tokens'] = sen["rm_style_tokens"] + [COCO_STYLE]
            tmp_rems_text.append(line)

            c+=1

    # downsample


    random.seed(0)
    input_rems_text.extend(random.sample(tmp_rems_text,lenrm))

    return input_rems_text

def main():
    data = get_data()
    #nltk.download('stopwords')
    nltk.download('framenet_v17')
    #stopset = set(stopwords.words('english'))

    js = json.load(open(emotion_train_path, "r"))
    #dec_idx_to_word = json.load(open('code/dec_idx_to_word.json','r'))  #原始语句进行tokenize，得到对应的词频统计
    framelist = get_FrameNet()
    data = []
    text = []
    wordlist = []
    counter = 0
    '''
    for word in dec_idx_to_word:
        if counter >300:
            break
        if 'RES' in word:
            continue
        if word not in stopset:
            counter += 1
            wordlist.append(word)
    
    for i, sen in enumerate(js):
        line = {}
        text = sen['tokens'] #原始语句
        text_tokens = tokenize(text)   #分词
        text_lemma = tokenize_lemma(text)   #词干化

        Str = ''
        for word in text_tokens:
            Str += word
        if len(text_tokens)>20 or len(text_tokens)<4:
            continue
        if len(Str)<10:
            continue
        
        flag = False
        for word in text_lemma:
            if word.lower() in wordlist:
                flag = True
                continue
        if flag == False:
            continue
        
    
    for i, sent in enumerate(js):
        
        line = {'raw':[],'rm_tokens':[]}
        if sent['label'] != 1:
            line['raw'] = sent['tokens']  
    # 剩下的是长度符合要求，且包含常用词的句子
            text = sent['tokens']
            text = ' '.join(text)
            #下面进行词性标注和词干化，得到semantic terms
            line['rm_tokens'] = preprocessd(text)
            if sent['label'] == 0:
                line['rm_tokens'].append(NEGATIVE_STYLE)
            if sent['label'] == 2:
                line['rm_tokens'].append(POSITIVE_STYLE)
            if sent['label'] == 1:
                line['rm_tokens'].append(COCO_STYLE)
            line['rm_tokens'] = tokenize_framenet(line['rm_tokens'],framelist)
            data.append(line)

        if sent['label'] == 1:
            break
        if (i+1) % 10000 == 0:
            print('Valid/Processed/Progress: %d/%d/%.3f%%' %(len(data),i+1,100*(i+1)/len(js)))
    
    ''' 
    with open('datasets/data2.json','w') as f:
        json.dump(data,f)



if __name__ == '__main__':
    main()