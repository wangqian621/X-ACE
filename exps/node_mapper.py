import json
import sys
sys.path.append("/home/qwang/text-audio/caption")
from openai import  OpenAI
import concurrent.futures
from tqdm import tqdm
import logging
from collections import OrderedDict
logger = logging.getLogger(__name__)
class Node_Map(object):
    def __init__(self,dataset='audiocaps'):
        file_path = f"data/{dataset}_map.json"
        self.data = self.load_data(file_path)
    def load_data(self,file_path):
        with open (file_path,'r') as f:
            datas = json.load(f)
        return datas
    def _get_node(self,text,audio=None,anno=None):
        try:
            data = json.loads(self.data[text])
        except:
            data = self.data[text]
        data.update({"caption":text,"wav":audio,"anno":anno})
        return data
