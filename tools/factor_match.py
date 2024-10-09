from tqdm import tqdm
from nltk.corpus import wordnet
from gensim.models import KeyedVectors
import torch
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch.nn.functional as F
import logging
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from .node_map import attr_map,lower_list
from .relation_extractor import rel_search_route
from .file_utils import ATMDataset
from torch.utils.data import DataLoader,SequentialSampler
logger = logging.getLogger(__name__)

# BEFORE_REL = ["before","then","followed","followed by","later","finally","after that"]
# AFTER_REL = ["after","following"]
# AND_REL = ["and","with","as","while","when","the same time","simultaneously"]
# SEQ_REL = ["before","then","followed","followed by","later","finally","after","following"]
TIME_REL = ["before","then","followed","followed by","later","finally","after","following","as","while","when","the same time","simultaneously"]
BeVerbs = ["be", "is", "am", "are", "was", "were", "been", "being"]
prep_word = [
    'about', 'above', 'across', 'after', 'against', 'along', 'among', 'around',
    'at', 'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond',
    'by', 'down', 'during', 'for', 'from', 'in', 'inside', 'into', 'near', 'of',
    'on', 'onto', 'out', 'over', 'through', 'to', 'toward',
    'under', 'underneath', 'until', 'unto', 'up', 'upon', 'with', 'within', 'without'
]

@torch.no_grad()
def lm_sim(hypo:str=" ",ref:str=" ",model=None):
    if model is None:
        model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentence1_representation =torch.tensor(model.encode(hypo,device='cpu'))
    sentence2_representation = torch.tensor(model.encode(ref,device='cpu'))
    cos_sim = util.pytorch_cos_sim(sentence1_representation,sentence2_representation)
    return cos_sim[0][0].numpy()

def none_to_list(input_node):
    if  input_node is None:
        return []
    return input_node
def event_source_concat(event:str=None, source:list=[]):
    if len(source)>0:
        try:
            source = (' and ').join(source)
        except:
            logger.warning(f"source:{source}")
            source = (' and ').join(source)
        event = source + ' ' + event
    return event


def mean_type(F1_type_total):
    F1_type_total_score =  {k:[s[k] for s in F1_type_total if s[k] is not None] for k in F1_type_total[0].keys()} 
    F1_type_total_score = {k:round(np.mean(v)*100,2) if len(v)>0 else None for k,v in F1_type_total_score.items()}
    return  F1_type_total_score

class FactorMatcher:# pred:[sample1,sample2,sample3] ref:[[sample1_ref1,sample1_ref2],[sample2_ref1,sample2_ref2]]
    def __init__(self,model="glove",args=None):
        self.lemmatizer = WordNetLemmatizer()
        self.wordnet = wordnet
        self.model_mode = model
        self.atm = self.load_atm(model)
        self.ttm = self.load_ttm(model)
        self.language_model_choose(model)
        self.other_keys = ["caption","relationship","wav","anno"]
        self.feature_keys = ["source","attr"]
        self.all_keys = ["sound","source","attr","rel"]
        self.beta = getattr(args,"beta",1) if args is not None else 1
        self.anno_num = getattr(args,"anno_num",957) if args is not None else 957
        self.ref_aggr_mode = getattr(args,"ref_aggr_mode","avg") if args is not None else "avg"# for multiple
        self.attr_map = attr_map(model=self.model,lemmatizer=self.lemmatizer) if "type" in model else None
        self.attr_display = True if "display" in model else False
        self.anno = True if "anno" in model else False
        self.in_map_num = 0
        self.total_num = 0
        self.not_match_display = []
        self.not_map = {"man":"woman","male":"female","man":"female","woman":"male"}
        
    def match_flag(self,hypo:str=" ",ref:str=" ",key_mode=""):
        hypo,ref = hypo.lower(),ref.lower()
        if self._not_map(hypo,ref):
            return 0.0
        if hypo == ref:
            return 1.0
        return self.lm_sim(hypo,ref,self.model)
    
    def load_atm(self,model):#load clap model
        if "atm" in model:
            from models.ase_model import ASE
            cp_path = "ckpt/HTSAT-BERT-FT-AudioCaps.pt"
            cp = torch.load(cp_path)
            config = cp["config"]
            model = ASE(config)
            model.load_state_dict(cp["model"],strict=False)
            model.eval()
            device = 'cuda:0'
            model.to(device)
            logger.warning("Loading ATM ...")
            return model
        return None
    def load_ttm(self,model):
        if "ttm" in model:
            model_sb = SentenceTransformer('paraphrase-TinyBERT-L6-v2', device='cuda:1')
            model_sb.eval()
            logger.warning("Loading TTM ...")
            return model_sb
        return None
    def language_model_choose(self,model):
        if model is not  None:
            if "sbert" in model:
                self.lm_sim = lm_sim
                sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
                self.model = sbert_model
                self.thres = 0.85
            elif "glove" in model:
                self.lm_sim = self.glove_similarity
                glove_cp = "ckpt/glove.6B.50d.txt"
                glove_model = KeyedVectors.load_word2vec_format(glove_cp, binary=False, no_header=True)
                self.model= glove_model
                self.thres = 0.75# 0.68
    def _not_map(self,hypo,ref):
        hypo_list = hypo.split(' ')
        ref_list = ref.split(' ')
        for k,v in self.not_map.items():
            for hypo in hypo_list:
                for ref in ref_list:
                    if (k==hypo and v==ref) or (v==hypo and k==ref):
                        return True 
        return False 
    def lemma_sentence(self,sentence):
        tokens = word_tokenize(sentence)
        lemmas = [self.lemmatizer.lemmatize(token,'v') for token in tokens]
        new_sentence = " ".join(lemmas)
        new_sentence = new_sentence.lower()
        return new_sentence
    def word2vec_equal(self,node1, node2):
        similarity = self.word2vec_nlp(node1).similarity(self.word2vec_nlp(node2))
        return similarity
    def glove_similarity(self,hypo:str=" ",ref:str=" ",model=None):
        phrase1,phrase2 = self.lemma_sentence(hypo),self.lemma_sentence(ref)
        words1,words2 = phrase1.split(), phrase2.split()
        vectors1 = [model[word] for word in words1 if word in model]
        vectors2 = [model[word] for word in words2 if word in model]
        if self._not_map(phrase1,phrase2):
            return 0.0
        if phrase2 in phrase1 or phrase1 in phrase2:
            return 1.0
        if not vectors1 or not vectors2:
            return 0.0
        tensor_vectors1 = [torch.from_numpy(np.array(arr,copy=True)) for arr in vectors1]
        tensor_vectors2 = [torch.from_numpy(np.array(arr,copy=True)) for arr in vectors2]
        avg_vector1 = torch.stack(tensor_vectors1).mean(dim=0)
        avg_vector2 = torch.stack(tensor_vectors2).mean(dim=0)
        similarity = F.cosine_similarity(avg_vector1, avg_vector2, dim=0)
        return similarity.item()
    def get_ttm_score(self,pred=None,ref=None,if_sentence=True):
        if not if_sentence:
            pred = "This is the sound of " + pred
        pred_emb = torch.Tensor(self.ttm.encode(pred))
        ref_emb = torch.Tensor(self.ttm.encode(ref))
        score = util.cos_sim(pred_emb,ref_emb).detach().numpy()
        return score
    def rel_match_mode(self,match_sound_pairs:list[str]=[],match_scores=[],pred=None,ref=None):#relationship pred=dict{str:str}
        pred_sounds,ref_sounds =[],[]
        for match_sound_pair in match_sound_pairs:
            pred_sound,ref_sound = match_sound_pair.split('||')
            pred_sounds.append(pred_sound)
            ref_sounds.append(ref_sound)
        route_weights = [match_scores[i]*match_scores[i+1] for i in range(len(match_scores)-1)]
        pred_rels = rel_search_route(sounds=pred_sounds,sound_rel=pred,key="pred")# a-b,b-c,c-d 实际数目-1
        ref_rels = rel_search_route(sounds=ref_sounds,sound_rel=ref,key="ref")
        rel_num = {"pred":len(pred_rels),"ref":len(ref_rels)}
        match_num = sum([route_weight if(pred_rel==ref_rel) else 0 for (pred_rel,ref_rel,route_weight) in zip(pred_rels,ref_rels,route_weights)])
        return match_num,rel_num
    def get_feature_match(self,preds=None,refs=None,key=""):
        if len(refs)==0 or len(preds)==0:
            return 0,[]
        feature_score = 0
        pred_score_list = []
        for pred in preds:
            max_score = 0
            # opt_ref = refs[0]
            for ref in refs:
                score = self.match_flag(pred,ref)
                if score>max_score:
                    max_score = score
                    # opt_ref = ref
            feature_score += max_score
            pred_score_list.append(max_score)
        return feature_score,pred_score_list
    def get_attr_display_match(self,preds=None,refs=None,key=""):
        if len(refs)==0 or len(preds)==0:
            return 0
        feature_score = 0
        attr_pairs = []
        for pred in preds:
            max_score = 0
            match_pair = (pred,refs[0])
            for ref in refs:
                score = self.match_flag(pred,ref)
                if score>max_score:
                    max_score = score
                    match_pair = (pred,ref,score)
            attr_pairs.append(match_pair)
            feature_score += max_score
        return feature_score,attr_pairs
    
    def match_per_sample(self,pred,ref,sample_idx=None,ref_idx=None):
        self.match_num = {"sound":0,"attr":0,"source":0,"rel":0}
        self.pred_num = {"sound":0,"attr":0,"source":0,"rel":0}
        pred_sounds,ref_sounds = list(pred["node"].keys()), list(ref["node"].keys())
        if len(ref_sounds)==0:
            return None
        if len(pred_sounds)==0:
            prec = {}
            for key in self.all_keys:
                prec.update({key:0})
            return prec  
        pred_score_e = 0.0
        match_pairs = []
        match_scores = []
        # Load Annotation
        if self.anno:
            anno_data = ref.get("anno",None)
            if anno_data is not None and anno_data!='':
                ref_node = ref["node"].copy()
                try:
                    anno_sounds = list(anno_data.keys())
                except:
                    logger.warning(f"anno:{ref}")
                    anno_sounds = list(anno_data.keys())
                for ref_e_idx,ref_e in enumerate(ref_sounds):
                    max_score = 0.0
                    max_anno_e = anno_sounds[0]
                    for anno_e_idx,anno_e in enumerate(anno_sounds):
                        score = self.match_flag(ref_e,anno_e)
                        if score>max_score:
                            max_score = score
                            max_anno_e = anno_e
                        if max_score>0.7:
                            ref_node.update({ref_e:anno_data[max_anno_e]})
                ref.update({"node":ref_node})
        
        for pred_e_idx,pred_e in enumerate(pred_sounds):
            max_score = 0.0
            max_ref_e = ref_sounds[0]
            for ref_e_idx,ref_e in enumerate(ref_sounds):
                score = self.match_flag(pred_e,ref_e)
                if score>max_score:
                    max_score = score
                    max_ref_e = ref_e
            match_pair = f"{pred_e}||{max_ref_e}"
            match_pairs.append(match_pair)
            match_scores.append(max_score)
            
            pred_score_e += max_score
            for key in self.feature_keys:
                pred_f,ref_f = pred["node"][pred_e][key],ref["node"][max_ref_e][key]
                pred_f,ref_f = lower_list(pred_f),lower_list(ref_f)
                if key=="source":
                    pred_score_f,_ = self.get_feature_match(preds=pred_f,refs=ref_f,key=key)
                elif key=="attr":
                    if self.attr_map is not None:
                        pred_score_f,pred_score_list = self.get_feature_match(preds=pred_f,refs=ref_f,key=key)
                        if self.cal_mode=="prec":
                            self.prec_type_score = self.attr_map.get_class_score(pred_f,pred_score_list,type_score=self.prec_type_score)
                        else:
                            self.recall_type_score = self.attr_map.get_class_score(pred_f,pred_score_list,type_score=self.recall_type_score)
                    else:
                        pred_score_f,_ = self.get_feature_match(preds=pred_f,refs=ref_f,key=key)
              
                pred_score_f = pred_score_f*max_score
                self.match_num[key] += pred_score_f
                self.pred_num[key] += len(pred_f)
        
        self.match_num["sound"] += pred_score_e
        self.pred_num["sound"] += len(pred_sounds)
        
        if len(match_pairs)>1:
            pred_rels,ref_rels = pred["relationship"], ref["relationship"]
            match_num,rel_num = self.rel_match_mode(match_pairs,match_scores=match_scores,pred=pred_rels,ref=ref_rels)
            self.match_num["rel"] += match_num #* divide_(self.match_num["sound"],self.pred_num["sound"])
            self.pred_num["rel"] += rel_num["pred"]
        
        prec = {}
        for key in self.all_keys:
            prec.update({key:divide_(self.match_num[key],self.pred_num[key])})
        return prec  
    
    @torch.no_grad()
    def get_atm_score(self,preds,ref_wavs,batch_size=32):
        batch_size = min(batch_size,len(preds))
        preds = [p["caption"] for p in preds]
        dataset = ATMDataset(preds=preds,ref_wavs=ref_wavs)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                sampler=SequentialSampler(dataset),
                                collate_fn=dataset.collate_fn)
        scores = np.zeros(len(dataset))
        for batch_id,batch in tqdm(enumerate(dataloader),total=len(dataloader),desc="ATM Matching..."):
            caps,wavs = batch
            pred_emb = torch.Tensor(self.atm.encode_text(caps))
            ref_emb = torch.Tensor(self.atm.encode_audio(torch.tensor(wavs).to(next(self.atm.parameters()).device)))
            tmp = util.cos_sim(pred_emb,ref_emb).detach().diagonal().cpu().numpy()
            try:
                scores[batch_id*batch_size:(batch_id+1)*batch_size] = tmp
            except IndexError:
                scores[batch_id*batch_size:] = tmp
        return scores
    def get_xace_score(self,pred=None,ref=None,ref_wavs=None,batch_size=32,hypo_key=None):
        atm_score = None
        if self.atm is not None:
            atm_score = self.get_atm_score(pred,ref_wavs,batch_size=batch_size)
        if self.attr_map is not None:
            self.recall_type_score = {ty:{"match":0.0,"total":0.0} for ty in self.attr_map.types}
            F1_type_total,prec_type_total,recall_type_total = [],[],[]
        self.hypo_key = hypo_key
        self.preds,self.refs = pred,ref
        F1_total = [0]*len(pred)
        F1_node_total = [0]*len(pred)
        for sample_idx,(pred,ref_list) in tqdm(enumerate(zip(self.preds,self.refs)),total=len(self.preds),desc="Node Matching..."):
            F1_sample = []
            F1_node_sample = []
            if self.attr_map is not None:
                self.prec_type_score = {ty:{"match":0.0,"total":0.0} for ty in self.attr_map.types}
                self.recall_type_score = {ty:{"match":0.0,"total":0.0} for ty in self.attr_map.types}
            for ref_idx,ref in enumerate(ref_list):
                self.cal_mode = "prec"
                prec = self.match_per_sample(pred,ref,sample_idx,ref_idx)
                self.cal_mode = "recall"
                recall = self.match_per_sample(ref,pred,sample_idx,ref_idx)
                if (prec is not None) and (recall is not None):
                    prec_total = np.mean([v for k,v in prec.items() if v is not None])
                    recall_total = np.mean([v for k,v in recall.items() if v is not None])
                    F1_node = {k:2*divide_(v*recall[k],v+recall[k]) for k,v in prec.items()}
                    if "recall" in self.model_mode:
                        F1 = recall_total
                    elif "prec" in self.model_mode:
                        F1 = prec_total
                    else:
                        F1  = 2*divide_(prec_total*recall_total,prec_total+recall_total)
                    F1_sample.append(F1)
                    F1_node_sample.append(F1_node)
                   
            if self.attr_map is not None:
                prec_type_sample = {k:divide_(v["match"],v["total"],pad=None) for k,v in self.prec_type_score.items()}
                recall_type_sample = {k:divide_(v["match"],v["total"],pad=None) for k,v in self.recall_type_score.items()}
                F1_type_sample = {k:F1_non(v,recall_type_sample[k]) for k,v in prec_type_sample.items()}
                F1_type_total.append(F1_type_sample)
                prec_type_total.append(prec_type_sample)
                recall_type_total.append(recall_type_sample)
                
            if self.ref_aggr_mode=='avg':
                F1_sample = np.mean(F1_sample)
                F1_node_sample = {k_n:np.mean([s[k_n] for s in F1_node_sample]) for k_n in F1_node_sample[0].keys()}
               
            elif self.ref_aggr_mode=='max':
                F1_sample = np.max(F1_sample)
                F1_node_sample = {k_n:np.max([s[k_n] for s in F1_node_sample]) for k_n in F1_node_sample[0].keys()}
            F1_total[sample_idx] = F1_sample
            F1_node_total[sample_idx] = F1_node_sample
        F1_total = 0.5*np.array(F1_total)+0.5*atm_score if atm_score is not None else 0.5*np.array(F1_total)
        if self.attr_map is not None:
            self.F1_type_total_score,self.prec_type_total_score,self.recall_type_total_score = mean_type(F1_type_total),mean_type(prec_type_total),mean_type(recall_type_total)
        
        return F1_total,F1_node_total
def divide_(a,b,pad=0):
    return a/b if b>0 else pad
def F1_non(a,b):
    if (a is None) and (b is None):
        return None
    elif a is None:
        return b
    elif b is None:
        return a
    else:
        return a*b*2/(a+b) if (a+b)>0 else 0

if __name__=="__main__":
    pass
    