
import json
import sys
import spacy
import torch
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import numpy as np
from nltk.translate import meteor_score
from .file_utils import load_data
from sentence_transformers import util
Prep_Word = [
    'above', 'across','against','along', 'among', 'around',
    'at', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond',
    'down', 'during', 'from', 'in', 'inside', 'into', 'near',
    'on', 'onto', 'out', 'over', 'through', 'toward',
    'under', 'underneath', 'unto', 'up', 'upon', 'within', 'without'
]
time_mapping = {
        "one": ["once", "1 times","one time"],
        "two": ["twice", "two times", "2 times"],
        "three": ["three times", "3 times"],
        "four": ["four times", "4 times"],
        "five": ["five times", "5 times"],
        "six": ["six times", "6 times"],
        "seven": ["seven times", "7 times"],
        "eight": ["eight times", "8 times"],
        "nine": ["nine times", "9 times"],
        "ten": ["ten times", "10 times"],
    }
sys.path.append('./')
def lower_list(input_list):
    return [] if input_list==[] else [input_.lower().strip() for input_ in input_list]
def set_list(input_list):
    return [] if input_list==[] else list(set(input_list))
class attr_map(object):
    def __init__(self,model=None,lemmatizer=None):
        file_path = "data/attr_map.json"
        self.maps = load_data(file_path)
        self.model = model
        self.thres = 0.72
        self.lemmatizer =lemmatizer
        self.wordnet = wordnet
        self.nlp = spacy.load("en_core_web_sm")
        self.not_map = {"man":"woman","male":"female","man":"female","woman":"male"}
        self.attr_map_buffer()
        self.tag_class,self.tag_embs = self.tag_emb_buffer()
        self.tags = lower_list(list(self.maps.keys()))
        self.time_word = ["times","twice","once","one time"]
        self.types = ["Duration","Volume","Pitch","Location","Distance","Change","Tone","Other","Count"]
        # self.type_score_dict = {ty:{"match":0.0,"pred":0.0} for ty in self.types}
    def is_locative_adverb(self,phrase):
        nlp = spacy.load("en_core_web_sm") if self.nlp is None else self.nlp
        doc = nlp("It occurs "+phrase)
        for token in doc:
            if token.dep_ == "prep":
                child_list = [child.tag_ for child in token.children]
                head_flag = (token.head.text =="occurs")
                if len(child_list)>0:
                    for child_ in child_list:
                        if "NN" in child_ and head_flag:
                            return True
        return False
    def istime_word(self,word):
        for time_ in self.time_word:
            if time_ in word:
                return True
        return False
    def in_map_flag(self,word):
        for k,v in self.maps.items():
            word_emb = self.maps[k]["emb"]
            if self.get_sim(pred=word,ref=k,ref_emb=word_emb)>self.thres:
                word_sub_cls = k
                return ("tag",word_sub_cls)
        for prep in Prep_Word:
            if prep in word:
                return ("loc",word)
        for time_ in self.time_word:
            if time_ in word:
                return  ("time",word) 
        return ("special",word)
    def lemma_sentence(self,sentence):
        sentence = sentence.lower()
        tokens = word_tokenize(sentence)
        lemmas = [self.lemmatizer.lemmatize(token,'v') for token in tokens]
        new_sentence = " ".join(lemmas)
        return new_sentence
    def attr_map_buffer(self):
        for attr in self.maps.keys():
            attr_emb = self.get_emb(attr)
            self.maps[attr].update({"emb":attr_emb})
    def tag_emb_buffer(self):
        tag_embs,tag_class = [],[]
        for i,(k,v) in enumerate(self.maps.items()):
            tag_emb = torch.zeros([50,1]).squeeze() if v["emb"] is None else v["emb"]
            tag_embs.append(tag_emb)
            tag_class.append(v["class"])
        tag_embs = torch.stack(tag_embs,dim=0)#[tag_num,emb]
        # print(f"tag:{tag_embs.shape}")
        return tag_class,tag_embs

    def get_emb(self,data:str=""):
        phrase = self.lemma_sentence(data)
        words = phrase.split()
        vectors = [self.model[word] for word in words if word in self.model]
        if len(vectors)==0:
            return None
        tensor_vectors = [torch.from_numpy(np.array(arr,copy=True)) for arr in vectors]
        avg_vector = torch.stack(tensor_vectors).mean(dim=0)
        return avg_vector
    def get_sim(self,pred:str=" ",ref:str=" ",pred_emb=None,ref_emb=None):
        pred_vec = self.get_emb(pred) if pred_emb is None else pred_emb
        ref_vec  = self.get_emb(ref) if ref_emb is None else ref_emb
        if (pred_vec is None) or (ref_vec is None):
            return 0.0
        similarity = F.cosine_similarity(pred_vec,ref_vec, dim=0)
        return similarity.item()
    def get_class(self,flag,display=False):
        type,word = flag
        if type=="tag":
            prime_class,sub_class = self.maps[word]["class"],self.maps[word]["sym"]
        elif type=="loc":
            if display:
                prime_class,sub_class = "Location",word
            else:
                prime_class,sub_class = "loc",word
        elif type=="time":
            if display:
                prime_class,sub_class = "Duration",word 
            else:
                prime_class,sub_class = "time",word
        elif type=="special":
            if display:
                prime_class,sub_class = "Other",word
            else:
                prime_class,sub_class = "special",word
        return prime_class,sub_class
    def get_time_sim(self,pred,ref):
        pred,ref = f"{pred}-type" if type(pred)==int else pred,f"{ref}-type" if type(ref)==int else ref
        pred,ref = pred.lower(),ref.lower()
        little_time = ["one times","two times","three times","once","twice","1 times","2 times","3 times"]
        if (pred=="3-type" and ref in little_time) or (ref=="3-type" and pred in little_time):
            return 1.0
        for number, descriptions in time_mapping.items():
            if {ref, pred}.issubset(set(descriptions)):
                return 1.0
        return 0.0
    def get_type_score(self,attr_pair):
        pred,ref,score = attr_pair
        pred = self.in_map_flag(pred)
        p_prime_class,_ = self.get_class(pred,display=True)
        self.type_score_dict[p_prime_class]["match"] += score
        self.type_score_dict[p_prime_class]["pred"] += 1.0
    def get_attr_match(self,preds=None,refs=None):
        if len(refs)==0 or len(preds)==0:
            return 0
        pred_flags = [self.in_map_flag(pred) for pred in preds]
        ref_flags = [self.in_map_flag(ref) for ref in refs]
        total_score = 0
        for p in pred_flags:
            p_prime_class,p_sub_class = self.get_class(p)
            max_score = 0
            for r in ref_flags:
                r_prime_class,r_sub_class = self.get_class(r)
                if p_prime_class==r_prime_class:
                    if p_prime_class in ["loc","special"]:
                        score = self.get_sim(pred=p_sub_class,ref=r_sub_class)
                    elif p_prime_class=="time":
                        score = self.get_time_sim(pred=p_sub_class,ref=r_sub_class)
                    else:#常规
                        score = (p_sub_class==r_sub_class)
                    if max_score<score:
                        max_score = score
                elif "Duration" in [p_prime_class,r_prime_class] and "time" in [p_prime_class,r_prime_class]:
                    score = self.get_time_sim(pred=p_sub_class,ref=r_sub_class)
            total_score += max_score
        return total_score
    def glove_similarity(self,pred:str=" ",ref:str=" "):
        phrase1 = self.lemma_sentence(pred)
        phrase2 = self.lemma_sentence(ref)
        words1 = phrase1.split()
        words2 = phrase2.split()
        vectors1 = [self.model[word] for word in words1 if word in self.model]
        vectors2 = [self.model[word] for word in words2 if word in self.model]
        if not vectors1 or not vectors2:
            return 0.0
        tensor_vectors1 = [torch.from_numpy(np.array(arr,copy=True)) for arr in vectors1]
        tensor_vectors2 = [torch.from_numpy(np.array(arr,copy=True)) for arr in vectors2]
        avg_vector1 = torch.stack(tensor_vectors1).mean(dim=0)
        avg_vector2 = torch.stack(tensor_vectors2).mean(dim=0)
        similarity = F.cosine_similarity(avg_vector1, avg_vector2, dim=0)
        return similarity.item()
    def _not_map(self,hypo,ref):
        hypo_list = hypo.split(' ')
        ref_list = ref.split(' ')
        for k,v in self.not_map.items():
            for hypo in hypo_list:
                for ref in ref_list:
                    if (k==hypo and v==ref) or (v==hypo and k==ref):
                        return True 
        return False 
    def get_word_match(self,hypo:str=" ",ref:str=" "):
        if self._not_map(hypo,ref):
            return False,0
        hypothesis = list(hypo.lower().split())
        reference = list(ref.lower().split())
        enum_hypothesis_list = [(i,x) for (i,x) in enumerate(hypothesis)]
        enum_reference_list = [(i,x) for (i,x) in enumerate(reference)]
        # exact match
        out,_,_ = meteor_score._match_enums(enum_hypothesis_list=enum_hypothesis_list,enum_reference_list=enum_reference_list)
        if len(out)>=1:
            # print(f"3:{len(out)/max(len(reference),len(hypothesis))}")
            return True,len(out)/max(len(reference),len(hypothesis))+0.3
        #stem match (词干匹配)
        out,_,_ = meteor_score._enum_stem_match(enum_hypothesis_list=enum_hypothesis_list,enum_reference_list=enum_reference_list)
        if len(out)>=1:
            # print(f"2:{len(out)/max(len(reference),len(hypothesis))}")
            return True,len(out)/max(len(reference),len(hypothesis))+0.2
        # wordnet fuzz match
        out,_,_ = meteor_score._enum_wordnetsyn_match(enum_hypothesis_list=enum_hypothesis_list,enum_reference_list=enum_reference_list,
                                    wordnet=self.wordnet)
        if len(out)>=1:
            # print(f"1:{len(out)/max(len(reference),len(hypothesis))}")
            return True,len(out)/max(len(reference),len(hypothesis))+0.1
        return False,0
    def aggregate_pairs(self,preds):
        input_list = []
        input_list.extend(preds)
        input_set = set_list(input_list)
        return preds,input_set
    def get_class_score(self,preds,pred_score,type_score=None):
        preds,input_set = self.aggregate_pairs(preds)
        attr_map_dict = {}
        attr_set_res = []
        attr_embs = []
        for attr in input_set:
            time_flag = self.istime_word(attr)
            if time_flag:
                # attr_map_dict.update({attr:{"class":"Count","emb":self.get_emb(attr)}})
                attr_map_dict.update({attr:"Count"})
                continue
            location_flag = self.is_locative_adverb(attr)
            if location_flag:
                # attr_map_dict.update({attr:{"class":"Location","emb":self.get_emb(attr)}})
                attr_map_dict.update({attr:"Location"})
                continue
            if attr in self.tags:
                # attr_map_dict.update({attr:{"class":self.maps[attr]["class"],"emb":self.get_emb(attr)}})
                attr_map_dict.update({attr:self.maps[attr]["class"]})
                continue
            attr_emb = self.get_emb(attr)
            if attr_emb is not None:
                attr_set_res.append(attr)
                attr_embs.append(attr_emb)
            else:
                attr_map_dict.update({attr:"Other"})
        if len(attr_embs)>0:
            attr_embs = torch.stack(attr_embs,dim=0)
            sim = util.cos_sim(attr_embs,self.tag_embs)#[total_num,tag_num]
            tag_sim,tag_idx = torch.max(sim,dim=1)
            for count,(idx,sim,attr) in enumerate(zip(tag_idx,tag_sim,attr_set_res)):
                tag = self.tag_class[idx.item()]
                # attr_map_dict.update({attr:{"class":tag,"emb":emb}})
                attr_map_dict.update({attr:tag})
        for p,score in zip(preds,pred_score):
            p_class = attr_map_dict[p]
            match,total = type_score[p_class]["match"],type_score[p_class]["total"]
            type_score.update({p_class:{"match":match+score,"total":total+1}})
        return type_score
                    
            
            

        
       
        
def is_locative_adverb(phrase):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(phrase)
        for token in doc:
            # print(f"{token}:{token.dep_}:{token.head}")
            if token.dep_ == "prep":
                print(f"{token}:{token.head.text}")
                head_flag = (token.head.text =="occurs")
                child_list = [child.tag_ for child in token.children]
                if len(child_list)>0:
                    for child_ in child_list:
                        if "NN" in child_ and head_flag:
                            # print(f"child**:{token}:{child_}")
                            return True
        return False
def main2():
    phrases = ["Something occurs on railroad tracks", "It occurs group of people", "It occurs off and on"]
    for phrase in phrases:
        result = is_locative_adverb(phrase)
        print(f"{phrase} 是地点状语吗？ {result}")

def main():
    from nltk.stem import WordNetLemmatizer
    from gensim.models import KeyedVectors
    glove_cp = "/home/qwang/text-audio/glove_cp/glove.6B/glove.6B.50d.txt"
    model = KeyedVectors.load_word2vec_format(glove_cp, binary=False, no_header=True)
    lemmatizer = WordNetLemmatizer()
    mapper = attr_map(model=model,lemmatizer=lemmatizer)
    pred = 'Off and on'
    ref = 'goes up and down'
    flag,level = mapper.get_word_match(hypo=pred,ref=ref)
if __name__=="__main__":
    main2()
