BEFORE_REL = ["before","then","followed","followed by","later","finally"]
AFTER_REL = ["after","following"]
AND_REL = ["and","with","as","while","when","the same time","simultaneously"]
SEQ_REL = ["before","then","followed","followed by","later","finally","after","following"]


def rel_conbination(pointer_rel:str="",rel:str=""):
    if (pointer_rel in BEFORE_REL and rel in AFTER_REL) or (rel in BEFORE_REL and pointer_rel in AFTER_REL):
        return "Unknown" # One is "before" and the other is "after", unable to determine the relationship.
    if pointer_rel in SEQ_REL: #if pointer_rel/rel is sequential relation, decided by pointer_rel/rel
        return pointer_rel
    return rel
def rel_route_flip(sound_k,rel_v):
    former,latter = sound_k.split('|&|')
    if rel_v=="before":
        rel_v = "after"
    elif rel_v =="after":
        rel_v = "before"
    return {f"{latter}|&|{former}":rel_v}
def rel_search_route(sounds=None,sound_rel:dict=None,key="pred"):
    pointer_rels=[]
    sound_rel_copy = sound_rel.copy()
    for (sound_k,rel_v) in sound_rel.items():
        sound_rel_copy.update(rel_route_flip(sound_k,rel_v))
    for i in range(len(sounds)-1):# A-B to B-C to C-D
        a,b = sounds[i],sounds[i+1]
        pointer_a = a
        pointer_rel = "and"
        for sound_k,rel_v in sound_rel_copy.items():
            if pointer_a in sound_k.split('|&|')[0]:
                pointer_rel = rel_conbination(pointer_rel=pointer_rel,rel=rel_v)
                if b in list(sound_k.split('|&|'))[-1]: #End
                    break
                pointer_a = list(sound_k.split('|&|'))[-1]
        pointer_rels.append(pointer_rel)
    return pointer_rels

class RelationExtractor(object):
    def __init__(self,caption:str=None,event_list:list[str]=None):
        self.caption = caption.lower()
        event_list = [event.lower() for event in event_list]
        
        self.caption_word_list = list(self.caption.split(' '))
        event_list = [event if len(list(event.split(' ')))==1 else event.split(' ')[-1] for event in event_list]
        self.event_idx_list = sorted(list(set([self.event_locator(event) for event in event_list])))
        
    def event_locator(self,event):
        for idx,word in enumerate(self.caption_word_list):
            if event in word:
                return idx
        return None
    def relationship_extract(self,last_idx=0,cur_idx=0):
        assert last_idx<cur_idx,f"last:{last_idx},cur:{cur_idx},event_idx:{self.event_idx_list}"
        for k in range(last_idx+1,cur_idx):
            word = self.get_word(k)
            if word in BEFORE_REL:
                return "before"
            elif word in AFTER_REL:
                return "after"
        return "and"
    def relation_node(self):
        relation_list = {}
        if len(self.event_idx_list) > 1:
            for j in range(len(self.event_idx_list)-1):
                last_idx  = self.event_idx_list[j]                                    
                cur_idx = self.event_idx_list[j+1]
                relation_flag = self.relationship_extract(last_idx,cur_idx)
                last_name = self.get_word(last_idx)
                cur_name = self.get_word(cur_idx)
                a_rel_b = {f"{last_name}|&|{cur_name}":relation_flag}
                relation_list.update(a_rel_b)
        return relation_list
    def get_word(self,index):
        return self.caption_word_list[index]
    
def get_relation(caption=None,node=None):
    event_names = list(node.keys())
    caption = caption.replace(',','').replace('.','')
    event_in_cap_num = sum([1 if key_e.lower() in caption.lower() else 0 for key_e in event_names])
    key_total_num = len(event_names)
    if event_in_cap_num == key_total_num:
        rel = RelationExtractor(caption=caption,event_list=event_names)
        rel_list_per_sample = rel.relation_node()
        return rel_list_per_sample
    return {}
    