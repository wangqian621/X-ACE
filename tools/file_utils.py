import json
import librosa
import random
import torch
from torch.utils.data import Dataset,DataLoader,SequentialSampler
import torch.nn.functional as F
from re import sub

def load_data(file_path):
    with open(file_path,'r') as f:
        datas = json.load(f)
    return datas

def write_data(out_datas,out_file_path):
    out_datas  = json.dumps(out_datas,indent=4)
    with open(out_file_path,'w') as f:
        f.write(out_datas)
        
def load_wav(audio_path):
    sr = 32000
    max_len = sr*10
    try:
        wav , _ = librosa.load(audio_path,sr=sr, mono=True)
        if wav.shape[-1] > max_len:
            max_s = random.randint(0,wav.shape[-1]-max_len)
            wav = wav[max_s:max_s+ max_len]
    except:
        return None
    return wav
def load_anno(audio_name):
    anno_data = load_data("fense/test_data/anno.json")
    try:
        audio_anno = anno_data[audio_name]
    except:
        audio_name = "Y"+audio_name
        audio_anno = anno_data.get(audio_name,None)
    return audio_anno

def text_preprocess(sentence):
    sentence = sentence.lower()
    sentence = sub(r'\s([,.!?;:"](?:\s|$))', r'\1', sentence).replace('  ', ' ')
    sentence = sub('[(,.!?;:|*\")]', ' ', sentence).replace('  ', ' ')
    return sentence
        
        
class ATMDataset(Dataset):
    def __init__(self,preds,ref_wavs):
        self.preds = preds
        self.ref_wavs = ref_wavs
    def __getitem__(self,index):
        wav = load_wav(self.ref_wavs[index])
        caption = text_preprocess(self.preds[index])
        return caption,torch.tensor(wav)
    def __len__(self):
        return len(self.preds)
    def collate_fn(self,batch):
        wavs = []
        caps= []
        max_len_batch = max([sample[-1].shape[-1] for sample in batch])
        for cap,wav in batch:
            caps.append(cap)
            if wav.shape[-1] < max_len_batch:
                pad_len = max_len_batch-wav.shape[-1]
                wav = F.pad(wav,[0,pad_len],'constant',0.0)
            wavs.append(wav)
        wavs = torch.stack(wavs,dim=0)
        return caps,wavs
            