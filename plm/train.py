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
                          batch_size=32)

    VALID_DL = DataLoader(ValueDataset(VALID_DATA, tokenizer), shuffle=True,
                          collate_fn=collate_fn(),
                          batch_size=32)
    num_labels = len(_values['2'])

    classifier = ValueClassifier(_config.bert_model, config=_config)
    classifier.train(TRAIN_DL, VALID_DL)








    model = AutoModelForSequenceClassification.from_pretrained(_config.bert_model,
                                                               num_labels=num_labels).to(_config.device)
    loss_fct = torch.nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_training_steps = len(TRAIN_DL) * _config.num_epochs
    lr_scheduler = get_scheduler(name="linear",
                                 optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=num_training_steps)


    def loss_fn(outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs,
                                            targets)


    model.train()
    optimizer = AdamW(params=model.parameters(),
                      lr=1e-05,
                      weight_decay=0.01,
                      correct_bias=True
                      )
    for epoch in range(_config.num_epochs):
        labels, predicts = [], []

        for batch in TRAIN_DL:
            batch = {k: v.to(_config.device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'])
            logits = outputs.logits
            optimizer.zero_grad()
            loss = loss_fn(logits,
                           batch['labels'].float())
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           1.0)
            lr_scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

            predicts.extend(torch.sigmoid(logits).cpu().detach().numpy().tolist())
            labels.extend(batch['labels'].cpu().detach().numpy().tolist())
            if len(labels) % 100 == 0:
                f1 = compute_metrics((np.asarray(predicts),
                                      np.asarray(labels)))
                print(f"STEP {len(labels)}", f1)
