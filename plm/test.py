"""
    Test Result On TestSet
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from configuration import BaseConfig
from data_loader import test_dataset, load_values_from_json, format_dataset, collate_fn, ValueDataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from plm.utils import compute_metrics


def get_results(data, partition_name):
    """
    Get Result in TSV format
    """
    dl = DataLoader(ValueDataset(data, TOKENIZER), shuffle=False,
                    collate_fn=collate_fn(),
                    batch_size=CONFIG.batch_size)

    result_csv = [['Argument ID'] + COLUMNS]
    predicts_list, labels = [], []
    with torch.no_grad():
        for batch in dl:
            batch, arg_ids = batch
            batch = {k: v.to(CONFIG.device) for k, v in batch.items()}
            outputs = MODEL(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'])
            logits = outputs.logits
            predicts = torch.sigmoid(logits).cpu().detach().numpy().tolist()
            predicts = torch.from_numpy(np.asarray(predicts))
            predicts = 1 * (predicts > 0.5).numpy()
            for preds, aid in zip(predicts, arg_ids):
                result_csv.extend([[aid] + preds.tolist()])

            if batch["labels"].shape[-1] > 0:
                predicts_list.extend(torch.sigmoid(logits).cpu().detach().numpy().tolist())
                labels.extend(batch['labels'].cpu().detach().numpy().tolist())
        if len(predicts_list) > 0:
            f1 = compute_metrics((np.asarray(predicts_list),
                                  np.asarray(labels)))
            print(f"{partition_name}: score-f1: {f1}")
    result = pd.DataFrame(result_csv[1:], columns=result_csv[0])
    os.makedirs("../results", exist_ok=True)
    result.to_csv(f"../results/result-{partition_name}.tsv", index=False, sep='\t')


if __name__ == '__main__':
    CONFIG = BaseConfig().get_config()
    VALUES = load_values_from_json(CONFIG.values_path)
    COLUMNS1 = pd.read_csv(CONFIG.train_labels_path, encoding='utf-8', sep='\t', header=0).columns.tolist()
    COLUMNS = ['Achievement', 'Benevolence: caring', 'Benevolence: dependability', 'Conformity: interpersonal', 'Conformity: rules', 'Face', 'Hedonism', 'Humility', 'Power: dominance', 'Power: resources', 'Security: personal', 'Security: societal', 'Self-direction: action', 'Self-direction: thought', 'Stimulation', 'Tradition', 'Universalism: concern', 'Universalism: nature', 'Universalism: objectivity', 'Universalism: tolerance']

    TRAIN_DATA = format_dataset(CONFIG.train_arguments_path,
                                CONFIG.train_labels_path,
                                VALUES)
    VALID_DATA = format_dataset(CONFIG.validation_arguments_path,
                                CONFIG.validation_labels_path,
                                VALUES)
    TEST_DATA = test_dataset(CONFIG.test_arguments_path)

    ZHIHU_DATA = format_dataset(CONFIG.zhihu_arguments_path,
                                CONFIG.zhihu_labels_path,
                                VALUES)
    TOKENIZER = AutoTokenizer.from_pretrained(CONFIG.bert_model)
    MODEL = AutoModelForSequenceClassification.from_pretrained(CONFIG.save_path).to(CONFIG.device)

    get_results(TEST_DATA, "test")
    get_results(TRAIN_DATA, "train")
    get_results(VALID_DATA, "valid")
    get_results(ZHIHU_DATA, "zhihu")
