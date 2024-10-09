import sys
sys.path.append('./')
from tools.factor_extractor import FactorExtractor
from tools.factor_match import FactorMatcher
from tools.file_utils import load_data,write_data
import numpy as np
import os
def anno2ref():
   
    ref_path = "data/test_audiocaps.json"
    anno_path = "data/test_audiocaps_annotation.json"
    ref_datas = load_data(ref_path) # from reference caption
    anno_data = load_data(anno_path) # from annotation
    ref_wavs = [d["audio"] for d in ref_datas]

    for d in ref_datas:
        [d[f"caption_{i}"] for i in range(1,6)]
    refs = [[]]*len(ref_datas)
    for idx,d in enumerate(ref_datas):
        anno = anno_data[d["id"]]
        new_d = [0]*5
        for i in range(1,6):
            d[f"caption_{i}"].update({"anno":anno})
            new_d[i-1] = d[f"caption_{i}"]
        refs[idx] = new_d
    batch_size = min(len(ref_wavs),32)
    return refs,ref_wavs,batch_size

def multi_refs(factor_extractor,refs,reference_path):
    if os.path.exists(reference_path):
        caption2node_map = load_data(reference_path)
        return caption2node_map
    ref_num = len(refs[0])
    new_ref = [[None]*ref_num for _ in range(len(refs))]
    for i in range(ref_num):
        ref_i = factor_extractor.inference([r[i] for r in refs])
        for j in range(len(ref_num)):
            new_ref[j][i] = ref_i[j]
    write_data(new_ref,reference_path)
    return new_ref
def get_total_score(hs,ns,key_modes=["sound","source","attr","rel"]):#holistic score/ node score
    hos_s = round(np.mean(hs)*100,2)
    node_s = { k: 0.0 for k in key_modes}
    for key in key_modes:
        node_s[key] = round(np.mean([n[key] for n in ns])*100,2)
    node_s.update({"total":hos_s})
    return node_s
def get_evaluation(preds: list[str]=None,
                   refs: list[list[str]]=None,
                   ref_wavs:list[str]=None,
                   model_mode:str="glove"):
    # Replace it with the downloaded ckpt path
    model_path = "ckpt/XACE-Llama-3-8B-Instruct"
    tokenizer_path = "ckpt/XACE-Llama-3-8B-Instruct"
    
    # Specify the storage location for the factor entity output from prediction
    pred_path = "data/pred.json"
    
    # Factor Extraction
    factor_extractor = FactorExtractor(model_path=model_path,
                                       tokenizer_path=tokenizer_path)
    if refs is None:
        refs,ref_wavs,batch_size = anno2ref()
    else:
        reference_path = "data/reference_for_others.json" #
        if len(refs[0])>1:
            refs = multi_refs(factor_extractor,refs,reference_path)
           
        else:
            refs = factor_extractor.inference(refs,reference_path)
        print( "Factor entity extraction for the reference caption has been done.")
        batch_size = min(32,len(refs))
    assert len(preds)==len(refs),"The number of samples for the prediction is different from that for the reference label"
    preds = factor_extractor.inference(preds,pred_path)
    print( "Factor entity extraction for the predicted caption has been done.")
    if ref_wavs is None:
        model_mode = model_mode.replace('-atm','')
        print("[Drop ATM Mode] The reference audios do not exist.")
    
    # Factor Matching
    
    matcher = FactorMatcher(model=model_mode)
    hs,ns = matcher.get_xace_score(pred=preds,ref=refs,ref_wavs=ref_wavs,
                                   batch_size=batch_size,
                                   )
    if matcher.attr_map is not None:
        attr_category_F1 = matcher.F1_type_total_score
        print(f"For attibute categories: \n {attr_category_F1}")
    total_score = get_total_score(hs,ns)
    return total_score

def main():
    # Please load the wav files located at the "audio" key in data/test_audiocaps.json in order and output the list of captions
    preds = load_data("data/pred_caption.json") 
    
    model_mode = "glove-atm-anno"
    # "atm" can be optionally added, representing audio-text cross-modal similarity calculation
    # "anno" represents human annotations provided in the X-ACE Benchmark to expand the reference data
    # "type" can be optionally added, e.g., model_mode = "glove-atm-anno-type", indicating a more detailed evaluation of the attribute factor, which will take more time
    
    total_score = get_evaluation(preds=preds,
                                 refs=None, # If eval is on the AudioCaps dataset, please specify as None
                                 ref_wavs=None, # If eval is on the AudioCaps dataset, please specify as None
                                 model_mode=model_mode)
    print(f"total score:{total_score}")
    
if __name__=="__main__":
    main()


