import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import sys
import librosa
import random
import torch
from re import sub
sys.path.append("/home/qwang/text-audio/caption")
from tools.file_utils import load_data,load_wav
with open('/home/qwang/text-audio/caption/spice-data/audiocaps_eval.json') as f:
    anno_audiocaps = json.load(f)
with open('/home/qwang/text-audio/caption/spice-data/clotho_eval.json') as f:
    anno_clotho = json.load(f)
def expand_refs(all_refs_text):
    for idx,sample in enumerate(all_refs_text):
        refs,audio,anno = sample
        i = 0
        while len(refs) < 4:
            refs.append(refs[i])
            i += 1
        all_refs_text[idx] = (refs,audio,anno)
    return all_refs_text
    
def swap_position(arr):
    ret = []
    for i in range(3):
        ret += [v for k,v in enumerate(arr) if k%3==i]
    return ret

def get_former(dataset='clotho',sample_num=99999,human_thresh=-1):
    assert dataset in {'audiocaps', 'clotho'}
    hh_human_truth = []
    hh_preds_text0 = []
    hh_preds_text1 = []
    hh_refs_text0 = []
    hh_refs_text1 = []

    anno = anno_clotho if dataset == 'clotho' else anno_audiocaps
    
    for i,audio in tqdm(enumerate(anno)):
        captions = audio["references"]
        audio_path = f"/home/qwang/text-audio/data_modified/AudioCaps/test/Y{audio['audio_id']}.wav"
        anno_data = load_data("/home/qwang/text-audio/caption/fense/test_data/anno.json")
        audio_name = f"Y{audio['audio_id']}.wav"
        audio_anno = anno_data.get(audio_name,None)
        wav = load_wav(audio_path)
        if wav is None:
            continue
        for facet in ["HC", "HI", "HM"]:
            try:
                truth = np.sum([x for x in audio[facet][-1]])
                if human_thresh>0 and (truth <human_thresh and truth>human_thresh*(-1)):
                    continue
                hh_preds_text0.append(audio[facet][0])
                hh_preds_text1.append(audio[facet][1])
                hh_human_truth.append(truth)
                if facet == "HC":
                    hh_refs_text0.append(([x for x in captions if x != audio[facet][0]],wav,audio_anno))
                    hh_refs_text1.append(([x for x in captions if x != audio[facet][1]],wav,audio_anno))
                elif facet == "HI" or facet == "HM":
                    hh_refs_text0.append(([x for x in captions if x != audio[facet][0]],wav,audio_anno))
                    hh_refs_text1.append(([x for x in captions if x != audio[facet][0]],wav,audio_anno))   
            except:
                continue
        if i>(sample_num-2):
            break
    
    hh_refs_text0 = expand_refs(hh_refs_text0)
    hh_refs_text1 = expand_refs(hh_refs_text1)
 
    hh_preds_text0 = swap_position(hh_preds_text0)
    hh_preds_text1 = swap_position(hh_preds_text1)
    hh_refs_text0 = swap_position(hh_refs_text0)
    hh_refs_text1 = swap_position(hh_refs_text1)
    hh_human_truth = swap_position(hh_human_truth)
    return hh_preds_text0, hh_preds_text1, hh_refs_text0, hh_refs_text1, hh_human_truth

def get_latter(dataset='clotho',sample_num=99999,human_thresh=-1):
    assert dataset in {'audiocaps', 'clotho'}
    mm_human_truth = []
    mm_preds_text0 = []
    mm_preds_text1 = []
    mm_refs_text = []

    anno = anno_clotho if dataset == 'clotho' else anno_audiocaps
    
    for i,audio in tqdm(enumerate(anno)):
        captions = audio["references"]
        audio_path = f"/home/qwang/text-audio/data_modified/AudioCaps/test/Y{audio['audio_id']}.wav"
        anno_data = load_data("/home/qwang/text-audio/caption/fense/test_data/anno.json")
        audio_name = f"Y{audio['audio_id']}.wav"
        audio_anno = anno_data.get(audio_name,None)
        
        wav = load_wav(audio_path)
        if wav is None:
            continue
        for facet in ["MM_1", "MM_2", "MM_3", "MM_4", "MM_5"]:
            try:
                truth = np.sum([x for x in audio[facet][-1]])
                if human_thresh>0 and (truth <human_thresh and truth > human_thresh*(-1)):
                    continue
                mm_preds_text0.append(audio[facet][0])
                mm_preds_text1.append(audio[facet][1])
                mm_refs_text.append((captions,wav,audio_anno))
                mm_human_truth.append(truth)
            except:
                continue
        if i>(sample_num-2):
            break

    return mm_preds_text0, mm_preds_text1, mm_refs_text, mm_human_truth
def kv_search(refs_dict=None,caption=None):
    (refs,audio_paths) = refs_dict
    refs_dict = zip(refs,audio_paths)
    caption = text_preprocess(caption)
    for i,(ref, audio_path) in enumerate(refs_dict):
        for ref_per in ref:
            ref_per = text_preprocess(ref_per)
            if caption in ref_per or ref_per in caption:
                return audio_path
    return "hi"

def text_preprocess(sentence):
    
    # transform to lower case
    sentence = sentence.lower()
    sentence = sentence.replace('. ','')
    sentence = sentence.replace('.','')
    sentence = sentence.replace(" ",'')
    sentence = sentence.replace(",",'')
    # remove any forgotten space before punctuation and double space
    # sentence = sub(r'\s([,.!?;:"](?:\s|$))', r'\1', sentence).replace('', '')

    # # remove punctuations
    # # sentence = sub('[,.!?;:\"]', ' ', sentence).replace('  ', ' ')
    # sentence = sub('[(,.!?;:|*\")]', ' ', sentence).replace('', '')
    return sentence           
def get_cross_former(dataset='clotho'):
    assert dataset in {'audiocaps', 'clotho'}
    hh_human_truth = []
    hh_preds_text0 = []
    hh_preds_text1 = []
    hh_refs_text0 = []
    hh_refs_text1 = []

    anno = anno_clotho if dataset == 'clotho' else anno_audiocaps
    datset_key = {'audiocaps':"AudioCaps","clotho":"Clotho"}
    audio_file_path = f"data_modified/{datset_key[dataset]}/json_files/test.json"
    with open(audio_file_path,'r') as f:
        datas = json.load(f)
    refs = [[data[f"caption_{i}"] for i in range(1,6)] for data in datas]
    audio_paths = [data["audio"] for data in datas] 
    # print(f"ref11111:{len(refs)}")
    refs_dict = (refs,audio_paths)
    anno = anno_clotho if dataset == 'clotho' else anno_audiocaps
    sr = 32000
    max_len = sr * 10
    ref_audios = []
    for audio in tqdm(anno):
        caption = audio["references"][0]
        captions = audio["references"]
        try:
            audio_path = kv_search(refs_dict=refs_dict,caption=caption)
            wav , _ = librosa.load(audio_path,sr=sr, mono=True)
            if wav.shape[-1] > max_len:
                max_s = random.randint(0,wav.shape[-1]-max_len)
                wav = wav[max_s:max_s+ max_len]
        except:
            continue
        for facet in ["HC", "HI", "HM"]:
            try:
                truth = np.sum([x for x in audio[facet][-1]])
                hh_preds_text0.append(audio[facet][0])
                hh_preds_text1.append(audio[facet][1])
                hh_human_truth.append(truth)
                ref_audios.append(torch.tensor(wav))
                if facet == "HC":
                    hh_refs_text0.append([x for x in captions if x != audio[facet][0]])
                    hh_refs_text1.append([x for x in captions if x != audio[facet][1]])
                elif facet == "HI" or facet == "HM":
                    hh_refs_text0.append([x for x in captions if x != audio[facet][0]])
                    hh_refs_text1.append([x for x in captions if x != audio[facet][0]])   
            except:
                continue
    hh_refs_text0 = expand_refs(hh_refs_text0)
    hh_refs_text1 = expand_refs(hh_refs_text1)
    hh_preds_text0 = swap_position(hh_preds_text0)
    hh_preds_text1 = swap_position(hh_preds_text1)
    hh_refs_text0 = swap_position(hh_refs_text0)
    hh_refs_text1 = swap_position(hh_refs_text1)
    ref_audios = swap_position(ref_audios)
    hh_human_truth = swap_position(hh_human_truth)
    return hh_preds_text0, hh_preds_text1, ref_audios,hh_human_truth,hh_refs_text0,hh_refs_text1
def get_cross_latter(dataset='clotho'):
    assert dataset in {'audiocaps', 'clotho'}
    mm_human_truth = []
    mm_preds_text0 = []
    mm_preds_text1 = []
    mm_refs_text = []
    datset_key = {'audiocaps':"AudioCaps","clotho":"Clotho"}
    audio_file_path = f"data_modified/{datset_key[dataset]}/json_files/test.json"
    with open(audio_file_path,'r') as f:
        datas = json.load(f)
    refs = [[data[f"caption_{i}"] for i in range(1,6)] for data in datas]
    audio_paths = [data["audio"] for data in datas] 
    # print(f"ref11111:{len(refs)}")
    refs_dict = (refs,audio_paths)
    anno = anno_clotho if dataset == 'clotho' else anno_audiocaps
    sr = 32000
    max_len = sr * 10
    ref_audios = []
    for audio in tqdm(anno):
        caption = audio["references"][0]
        captions = audio["references"]
        audio_path = kv_search(refs_dict=refs_dict,caption=caption)
        try:
            wav , _ = librosa.load(audio_path,sr=sr, mono=True)
            if wav.shape[-1] > max_len:
                max_s = random.randint(0,wav.shape[-1]-max_len)
                wav = wav[max_s:max_s+ max_len]
        except:
            continue
        for facet in ["MM_1", "MM_2", "MM_3", "MM_4", "MM_5"]:
            try:
                truth = np.sum([x for x in audio[facet][-1]])
                mm_preds_text0.append(audio[facet][0])
                mm_preds_text1.append(audio[facet][1])
                mm_human_truth.append(truth)
                mm_refs_text.append(captions)
                ref_audios.append(torch.tensor(wav))
            except:
                continue

    return mm_preds_text0, mm_preds_text1, ref_audios, mm_human_truth,mm_refs_text
if __name__ == '__main__':
    all_preds_text0, all_preds_text1, all_refs_text0, all_refs_text1, all_human_truth = get_former('audiocaps')
    # all_human_truth应该是给四个人进行评价，取sum
    mm_preds_text0, mm_preds_text1, mm_refs_text, mm_human_truth = get_latter('audiocaps')
    
    print(len(all_human_truth))

    # import pdb; pdb.set_trace()
    # print(np.array(all_refs_text0, dtype=str).shape)
    # print(np.array(mm_preds_text0, dtype=str).shape)

    # np.save('total_preds_audiocaps0.npy', all_preds_text0 + mm_preds_text0)
    # np.save('total_preds_audiocaps1.npy', all_preds_text1 + mm_preds_text1)
