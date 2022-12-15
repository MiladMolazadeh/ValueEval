"""
    Training Model
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler, AdamW, AutoModel
from torch.utils.data import DataLoader
import numpy as np
# =================== In project packages =====================
from configuration import BaseConfig
from data_loader import ValueDataset, load_values_from_json, format_dataset, collate_fn
from utils import compute_metrics
from models import ValueClassifier

if __name__ == '__main__':
    _config = BaseConfig().get_config()
    _values = load_values_from_json(_config.values_path)
    TRAIN_DATA = format_dataset(_config.train_arguments_path,
                                _config.train_labels_path,
                                _values)
    VALID_DATA = format_dataset(_config.validation_arguments_path,
                                _config.validation_labels_path,
                                _values)
    tokenizer = AutoTokenizer.from_pretrained(_config.bert_model)

    TRAIN_DL = DataLoader(ValueDataset(TRAIN_DATA, tokenizer), shuffle=True,
                          collate_fn=collate_fn(),
                          batch_size=_config.batch_size)

    VALID_DL = DataLoader(ValueDataset(VALID_DATA, tokenizer), shuffle=True,
                          collate_fn=collate_fn(),
                          batch_size=_config.batch_size)
    # TRAIN_DL = [d for dl in [TRAIN_DL, VALID_DL] for d in dl]
    num_labels = len(_values['2'])

    classifier = ValueClassifier(_config.bert_model, config=_config)
    classifier.train(TRAIN_DL, VALID_DL)
