"""
    script for torch datasets

"""

import torch
from torch.utils.data import Dataset


class ValueDataset(Dataset):
    """
    create Dataset for intent detection
    """

    def __init__(self, data, tokenizer):
        """
        init of intent dataset
        Args:
            data: list of dictionary of utt and labels
            tokenizer: bert pretrained model
        """
        self.samples = data.to_dict('records')
        self.tokenizer = tokenizer
        self.cls_token = tokenizer.encode(tokenizer.cls_token, add_special_tokens=False)[0]
        self.sep_token = tokenizer.encode(tokenizer.sep_token, add_special_tokens=False)[0]

    def __len__(self):
        return len(self.samples)

    def tokenize(self, input_text):
        return self.tokenizer(input_text, truncation=True, add_special_tokens=False).input_ids

    def __getitem__(self, idx):
        sample = self.samples[idx]
        conclusion = [self.sep_token] + self.tokenize(sample['Conclusion'])
        stance = [self.sep_token] + self.tokenize(sample['Stance'])
        premise = self.tokenize(sample['Premise'])
        sequence_ids = [self.cls_token] + premise + stance + conclusion + [self.sep_token]
        arg_id = sample['Argument ID']
        labels = [b for a, b in sample.items() if a not in ['Argument ID',
                                                           "Conclusion",
                                                           "Stance",
                                                           "Premise",
                                                           "Usage"]]
        sequence_ids = torch.tensor(sequence_ids)
        return sequence_ids, torch.sign(sequence_ids), labels, arg_id


class ToxicDataset(Dataset):
    """
    create Dataset for intent detection
    """

    def __init__(self, data, tokenizer):
        """
        init of intent dataset
        Args:
            data: list of dictionary of utt and labels
            tokenizer: bert pretrained model
        """
        self.samples = data.to_dict('records')
        self.tokenizer = tokenizer
        self.cls_token = tokenizer.encode(tokenizer.cls_token, add_special_tokens=False)[0]
        self.sep_token = tokenizer.encode(tokenizer.sep_token, add_special_tokens=False)[0]

    def __len__(self):
        return len(self.samples)

    def tokenize(self, input_text):
        return self.tokenizer(input_text, truncation=True, add_special_tokens=False).input_ids

    def __getitem__(self, idx):
        sample = self.samples[idx]
        premise = self.tokenize(sample['comment_text'])
        sequence_ids = [self.cls_token] + premise + [self.sep_token]
        labels = [b for a, b in sample.items() if a not in ['id',
                                                           "comment_text"]]
        sequence_ids = torch.tensor(sequence_ids)
        return sequence_ids, torch.sign(sequence_ids), labels
