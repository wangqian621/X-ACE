import numpy as np
import pandas as pd
import torch
import sys
sys.path.append('/home/qwang/text-audio/caption')
sys.path.append('/home/qwang/text-audio/caption/fense')
from Node_Match_LLM_anno2 import NodeMatcher
from node_mapper import Node_Map
import logging
logger = logging.getLogger(__name__)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dataloader import get_former, get_latter
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


def cosine_similarity(input, target):
    from torch.nn import CosineSimilarity
    cos = CosineSimilarity(dim=0, eps=1e-6)
    return cos(input, target).item()


def get_llm_node_score(all_preds_text=None, all_refs_text=None,args=None,cand_idx=None,hm=None,matcher=None):
    dataset = args.dataset
    mapper = Node_Map(dataset)
    model = "glove"
    hypo_node = [mapper._get_node(text) for text in all_preds_text]
    ref_nodes = [[mapper._get_node(ref,audio,anno) for ref in text_sample] for (text_sample,audio,anno) in all_refs_text]
    
    node_matcher = matcher if matcher is not None else NodeMatcher(model=model,args=args)
    total_score =  torch.Tensor(node_matcher.recall_averager(pred=hypo_node,ref=ref_nodes))
    # total_score =  torch.Tensor(node_matcher.recall_average)
    logger.warning(f"in map num:{node_matcher.in_map_num}/{node_matcher.total_num}")
    return total_score,hypo_node,ref_nodes

def get_accuracy(machine_score, human_score, threshold=0):
    cnt = 0
    # threshold = 0.001*np.average([abs(t) for t in machine_score])
    # threshold = 1e-6
    N = np.sum([x!=0 for x in human_score]) if threshold==0 else len(human_score)
    for i, (ms, hs) in enumerate(zip(machine_score, human_score)):
        if ms*hs > 0 or abs(ms-hs) < threshold:
            cnt += 1
    return cnt / N
def print_accuracy(machine_score, human_score,len_=None):
    results = []
    if len_ is not None:
        H_inter = int(len_/3)
    else:
        H_inter = 250
    for i, facet in enumerate(['HC', 'HI', 'HM', 'MM']):
        if facet != 'MM':
            sub_score = machine_score[i*H_inter:(i+1)*H_inter]#其余指标间隔为250
            sub_truth = human_score[i*H_inter:(i+1)*H_inter]
        else:
            sub_score = machine_score[i*H_inter:]#最后250个
            sub_truth = human_score[i*H_inter:]
        # logger.warning(f"{facet}:{sub_score[:10]}-{sub_truth[:10]}")
        acc = get_accuracy(sub_score, sub_truth)
        results.append(round(acc*100, 1))
        print(facet,  "%.1f" % (acc*100))
    acc = get_accuracy(machine_score, human_score)
    results.append(round(acc*100, 1))
    print("total acc: %.1f" % (acc*100))
    return results
def get_score_list(per_file_metric, metric):
            if metric == 'SPICE':
                return [v[metric]['All']['f'] for k,v in per_file_metric.items()]
            else:
                return [v[metric] for k,v in per_file_metric.items()]
def shrink(arr, repeat=5):
    print(f"arr:{np.array(arr).shape}")    
    return np.array(arr).reshape(-1, repeat).mean(axis=1).tolist()
def get_false_idx(score0,score1,human_score,threshold=0,node=None,text=None):
    score = score0-score1
    false_idx = {}
    for i, (ms, hs) in enumerate(zip(score, human_score)):
        if ms*hs > 0 or abs(ms-hs) < threshold:
            continue
        temp = {str(i):{"0":score0[i],"1":score1[i],"human":hs}}
        # hypo0,hypo1,ref0,ref1 = node["hypo0"],node["hypo1"],node["ref0"],node["ref1"]
        print(f"{temp}\n")
        for k,v in node.items():
            if "ref" in k:
                if "t" in k:
                    temp = v[i]
                else:
                    temp = [v_[i] for v_ in v]
                print(f"{k}:{temp}\n")
            else:
                print(f"{k}:{v[i]}\n")
        print("**********************************")
        print("**********************************")
        # false_idx.update({str(i):{"0":score0[i],"1":score1[i],"human":hs}})
    # print(false_idx)
    return false_idx
def get_caption_len(input):
    return [len(list(sample[0].split(' '))) for sample in input]
def metric_test(beta):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern",default="LLM")
    args = parser.parse_args()
    args.beta = beta
    args.eval_mode = "max"
    args.sample_num = 9999
    args.return_sample =True
    args.anno_num = 9999
    args.display_cap_len = True
    logger.warning(f"anno_num:{args.anno_num}")
    human_thresh = -1 #-1
    for dataset in ['audiocaps']:#'audiocaps'
        args.dataset = dataset
        model = "glove-anno-atm"
        logger.warning(f"model:{model}")
        matcher = NodeMatcher(model=model,args=args)
        hh_preds_text0, hh_preds_text1, hh_refs_text0, hh_refs_text1, hh_human_truth = get_former(dataset,sample_num=args.sample_num,human_thresh=human_thresh)# 750[1],750[1],750[4],750[4],750[1]
        mm_preds_text0, mm_preds_text1, mm_refs_text, mm_human_truth = get_latter(dataset,sample_num=args.sample_num,human_thresh=human_thresh)#921
        # hh0_len,hh1_len,mm0_len,mm1_len = get_caption_len(hh_preds_text0),get_caption_len(hh_preds_text1),get_caption_len(mm_preds_text0),get_caption_len(mm_preds_text1)
        
        total_score0, total_score1, total_score = {}, {}, {}
        total_human_truth = hh_human_truth + mm_human_truth #concate 750+921=1671 这里的score应该是pred0-pred1的相对得分
        
        # Ours hh and mm
        logger.warning("-----------------Calucate hh score-----------------")
        hm = "hh"
        ours_hh_0_score,hypo_node0,refs_node0 = get_llm_node_score(all_preds_text=hh_preds_text0, all_refs_text=hh_refs_text0,args=args,cand_idx=0,hm=hm,matcher=matcher)
        ours_hh_1_score,hypo_node1,refs_node1 = get_llm_node_score(all_preds_text=hh_preds_text1, all_refs_text=hh_refs_text1,args=args,cand_idx=1,hm=hm,matcher=matcher)
        # node = {"hypo0_t":hh_preds_text0,"hypo0":hypo_node0,"ref0_t":hh_refs_text0,"ref0":refs_node0,"hypo1_t":hh_preds_text1,"hypo1":hypo_node1,"ref1_t":hh_refs_text1,"ref1":refs_node1}
        # get_false_idx(score0=ours_hh_0_score,score1=ours_hh_1_score,human_score=hh_human_truth,node=node)
        # sys.exit()
        logger.warning("-----------------Calucate mm score-----------------")
        hm = "mms"
        ours_mm_0_score,hypo_node0,refs_node0 = get_llm_node_score(all_preds_text=mm_preds_text0, all_refs_text=mm_refs_text,args=args,cand_idx=0,hm=hm,matcher=matcher)
        ours_mm_1_score,hypo_node1,refs_node1 = get_llm_node_score(all_preds_text=mm_preds_text1, all_refs_text=mm_refs_text,args=args,cand_idx=1,hm=hm,matcher=matcher)
        # node = {"hypo0_t":mm_preds_text0,"hypo0":hypo_node0,"ref0_t":mm_refs_text,"ref0":refs_node0,"hypo1_t":mm_preds_text1,"hypo1":hypo_node1,"ref1_t":mm_refs_text,"ref1":refs_node1}
        # get_false_idx(score0=ours_mm_0_score,score1=ours_mm_1_score,human_score=hh_human_truth,node=node)
        # sys.exit()
        metric = "Ours"
        total_score0[metric] = torch.cat([ours_hh_0_score,ours_mm_0_score])
        total_score1[metric] = torch.cat([ours_hh_1_score, ours_mm_1_score])
        total_score[metric] = total_score0[metric] - total_score1[metric]
                
                
        results = []
        logger.warning(f"Dataset is :{dataset}")
        for metric in total_score:
            print(metric)
            tmp = print_accuracy(total_score[metric], total_human_truth,len_=len(ours_hh_0_score))
            results.append(tmp)

        df = pd.DataFrame(results, columns=['HC', 'HI', 'HM', 'MM', 'total'])
        df.index = [x for x in total_score]
        df.to_csv(f"pics/results_{dataset}_{args.eval_mode}_{args.sample_num}_{args.beta}.csv")

if __name__ =="__main__":
    # fense_test()
    beta = 1
    metric_test(beta)