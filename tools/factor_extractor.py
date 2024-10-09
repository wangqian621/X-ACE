import torch
import re
from transformers import AutoTokenizer,AutoModelForCausalLM
from .reformat_data import Reformator
from .relation_extractor import get_relation
from .file_utils import load_data,write_data
from tqdm import tqdm
import json
import os

def elimate_quotes(input_str):# We've-> We ve
    def replace_quotes(match):
        return match.group(0).replace("'", " ")
    pattern = r"(?<=\w)'(?=\w)"
    fixed_str = re.sub(pattern, replace_quotes, input_str)
    return fixed_str
def replace_other_quotes(input_str):
    remove_chars = r'[\n@！#￥^]'
    cleaned_text = re.sub(remove_chars, '', input_str)
    return cleaned_text
class FactorExtractor(object):
    def __init__(self,model_path,tokenizer_path,batch_size=64):
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                        load_in_8bit=True, 
                                        device_map='auto', 
                                        torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.instruction = "### Instrcution:\n\n \
    Please help me to extract 3 type of entities in the audio caption: sound event (the sound activities),\
    source (who generate the sounds), attribute (auditory attribute of the sounds). You need to \
    to extract dict of which keys are sound events, value are attribute list toward sounds and source (If no, list should be replaced by None). \
    You should pay attention on the sound event (e.g. speak, cries..) should be copied from the original caption. ###Input Caption:\n{caption}\n\n \
    Please return the entities extracted from above caption in the format of json, not the code!\n\n### Response:\n\n"
        self.batch_size = batch_size
        self.reformator = Reformator() # format check and rectify
    def str2token(self,input_strs):
        token_batch = self.tokenizer(input_strs, 
                                     return_tensors="pt",
                                     padding=True,
                                     truncation=True).to("cuda")
        return token_batch
    def map_dict(self,input_str):
        return self.instruction.format_map({"caption":input_str})
    
    def inference(self,
                  caption_list:list[str],
                  path="data/xxxxxxxx.json"):
        # caption_list is a list of generated captions from the user
        # path is the JSON file that converts captions into factor entities
        if os.path.exists(path):
            caption2node_map = load_data(path)
            return caption2node_map
        
        self.batch_size = min(self.batch_size,len(caption_list))
        caption2node_map = []
        caption_list = list(map(replace_other_quotes,caption_list))
        for batch_idx in tqdm(range(0,len(caption_list),self.batch_size)):
            try:
                input_batch = caption_list[batch_idx:batch_idx+self.batch_size]
            except IndexError:
                input_batch = caption_list[batch_idx:]
            # LLM inference to extract sound event/ source / attribute entities
            node_batch = self.inference_batch(input_batch)
            for node,caption in zip(node_batch,input_batch):
                # Relationship entities inference
                relation = get_relation(caption=caption,node=node)
                caption2node_map.append({"caption":caption,"node":node,"relationship":relation})
        write_data(caption2node_map,path)
        return caption2node_map
    def inference_batch(self,input_strs=None):
        if input_strs is None:
            input_strs = [
            "An explosion and then gunfire and a speech followed by another explosion",
            "Four gunshots occur followed by a man screaming and then an explosion",
            "Heavy footsteps walking on a hard surface followed by a man grunting then two explosions",
            "Loud footsteps followed by a scream and explosions",
            "Wind blowing followed by heavy footsteps on a solid surface then a man groaning proceeded by two explosions"
        ]
        input_strs = input_strs.copy()
        input_prompts = list(map(self.map_dict,input_strs))
        batch = self.str2token(input_prompts)
        self.model.eval()
        with torch.no_grad():
            batch_output = self.tokenizer.batch_decode(self.model.generate(**batch, max_new_tokens=400,pad_token_id=self.tokenizer.eos_token_id),
                                                       skip_special_tokens=True)
        for idx,output in enumerate(batch_output):
            output = output.split("### Response:\n\n")[-1]
            try:
                output = eval(elimate_quotes(output))
                output = self.reformator.reformat_per_node(output) # check format
            except:
                output = None
            if output is not None:
                input_strs[idx] = output
        return input_strs
def get_audiocaps():
    file_path = "/home/qwang/text-audio/caption/spice-data/audiocaps_eval.json"
    out_file_path = "data/audiocaps_map.json"
    datas = load_data(file_path)
    new_dict = {}
    other_keys = ["audio_id","raw_name","audio_path"]
    for d in datas:
        for k,v in d.items():
            if k in other_keys:
                continue
            if v is not None:
                if k!="references":
                    v = v[:2]
                
                for cap in v:
                    if cap not in new_dict:
                        new_dict.update({cap:{"node":None,"relationship":None,"audio":d.get("audio_path",None)}})
    input_strs = list(new_dict.keys())
    map_dict = get_inference(input_strs)
    write_data(map_dict,out_file_path=out_file_path)
    return map_dict
    
            
                
                
                
        
        
def get_inference(input_strs=None):
    model_path = "/home/qwang/text-audio/checkpoint/XACE-Llama-3-8B-Instruct"
    tokenizer_path = "/home/qwang/text-audio/checkpoint/Meta-Llama-3-8B-Instruct"
    factor_extractor = FactorExtractor(model_path=model_path,tokenizer_path=tokenizer_path)
    if input_strs is None:
        input_strs = [
            "An explosion and then gunfire and a speech followed by another explosion",
            "Four gunshots occur followed by a man screaming and then an explosion",
            "Heavy footsteps walking on a hard surface followed by a man grunting then two explosions",
            "Loud footsteps followed by a scream and explosions",
            "Wind blowing followed by heavy footsteps on a solid surface then a man groaning proceeded by two explosions"
        ]
    
    node = factor_extractor.inference(input_strs)
    return node

if __name__ == "__main__":
    # get_inference()
    # 传递给reformat_per_node方法
    # output = self.reformator.reformat_per_node(correct_dict)
    pass
