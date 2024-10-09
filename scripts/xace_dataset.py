# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json

import torch
from torch.utils.data import Dataset

def get_xace_dataset(
    dataset_config, tokenizer, partition="train", max_words=30):
    return XACEDataset(dataset_config,  tokenizer= tokenizer,max_words=max_words)
PROMPT_DICT = {
"prompt":(
    "### Instrcution:\n\n \
    Please help me to extract 3 type of entities in the audio caption: sound event (the sound activities),\
    source (who generate the sounds), attribute (auditory attribute of the sounds). You need to \
    to extract dict of which keys are sound events, value are attribute list toward sounds and source (If no, list should be replaced by None). \
    You should pay attention on the sound event (e.g. speak, cries..) should be copied from the original caption. ###Input Caption:\n{caption}\n\n \
    Please return the entities extracted from above caption in the format of json.\n\n### Response:\n\n"
)
}

class XACEDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train",max_words=30):
        self.ann = json.load(open(dataset_config.data_path))
        if partition == "train":
            self.ann = self.ann[1000:]
        else:
            self.ann = self.ann[:1000]

        self.tokenizer = tokenizer
        self.max_words = max_words

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  

        ann = self.ann[index]
        node = str(ann["node"])
        prompt = PROMPT_DICT["prompt"].format_map(ann)
        example = prompt + node #label
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask":example_mask,
        }