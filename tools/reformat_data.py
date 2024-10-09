# def valid
BEFORE_REL = ["before","then","followed","followed by","later","finally","after that"]
AFTER_REL = ["after","following"]
AND_REL = ["and","with","as","while","when","the same time","simultaneously"]
SEQ_REL = ["before","then","followed","followed by","later","finally","after","following"]
TIME_REL = ["before","then","followed","followed by","later","finally","after","following","as","while","when","the same time","simultaneously"]
BeVerbs = ["be", "is", "am", "are", "was", "were", "been", "being"]
prep_word = [
    'about', 'above', 'across', 'after', 'against', 'along', 'among', 'around',
    'at', 'before', 'behind', 'below', 'beneath', 'beside', 'between', 'beyond',
    'by', 'down', 'during', 'for', 'from', 'in', 'inside', 'into', 'near', 'of',
    'on', 'onto', 'out', 'over', 'through', 'to', 'toward',
    'under', 'underneath', 'until', 'unto', 'up', 'upon', 'with', 'within', 'without'
]
class Reformator(object):
    def __init__(self):
        self.source_forbid =  ["someone","object","none","null","somebody","something"]
    def reformat_per_node(self,node):
        for event,child in node.items():
            new_child = {}
            for key in ["source","attr"]:
                ent  = self.reformat(child[key],key)
                new_child.update({key:ent})
            node.update({event:new_child})
        return node
                
    def reformat(self,data,data_mode):# preproess for entity list
        self.data_mode = data_mode
        if data is None or data==[]:
            return []
        if type(data)==str:
            data = [d for d in data.split(',')]
        elif type(data) in [int,float]:
            data = [str(data)]
        data = list(map(self.reformat_valid_str,data))
        data = [d for d in data if d is not None]
        return data
    def reformat_valid_str(self,data):
        data  = data.lower()
        if self.data_mode=="source":
            if len(data)<3:
                return None
            if data in self.source_forbid:
                return None
        elif self.data_mode=="attr":
            if len(data)<3:
                return None
            elif type(data)==str and self.check_time_attr(data):
                return None
        return data
            
    def check_time_attr(self,unit):
        unit_split = unit.split(' ')
        for word in unit_split:
            if word in TIME_REL:
                return True
            if (word in prep_word or word in BeVerbs) and len(list(unit_split))<2:
                return True
            if word.lower() in ["null","none","false","unknown"]:
                return True
        return False
            
            
        
        
        
        