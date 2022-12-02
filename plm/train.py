"""
    Training
"""
import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification, get_scheduler, AdamW, AutoModel
from torch.utils.data import DataLoader
from data_loader import ValueDataset, load_values_from_json, format_dataset, collate_fn, ToxicDataset
from configuration import BaseConfig
from sklearn.metrics import f1_score
import numpy as np


def accuracy_thresh(y_pred, y_true, thresh=0.5, sigmoid=True):
    """Compute accuracy of predictions"""
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)
    if sigmoid:
        y_pred = y_pred.sigmoid()

    return ((y_pred > thresh) == y_true.bool()).float().mean().item()


def f1_score_per_label(y_pred, y_true, thresh=0.5, sigmoid=True):
    """Compute label-wise and averaged F1-scores"""
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)
    if sigmoid:
        y_pred = y_pred.sigmoid()
    y_true = y_true.cpu().bool().numpy()
    print(1 * (y_pred > thresh).numpy().sum())
    print((1 * y_true).sum())

    y_pred = (y_pred > thresh).numpy()

    f1_scores = round(f1_score(y_true, y_pred, zero_division=0, average='macro'), 2)

    return f1_scores


def compute_metrics(eval_pred):
    """Custom metric calculation function for MultiLabelTrainer"""
    predictions, labels = eval_pred
    f1scores = f1_score_per_label(predictions, labels, sigmoid=False)
    return {'accuracy_thresh': accuracy_thresh(predictions, labels),
            'f1-score': f1scores}


if __name__ == '__main__':
    _config = BaseConfig().get_config()
    _values = load_values_from_json(_config.values_path)
    _dataset = format_dataset(_config.data_path, _config.arguments_path, _values)
    tokenizer = AutoTokenizer.from_pretrained(_config.bert_model)

    _dataset = ValueDataset(_dataset, tokenizer)
    train_dl = DataLoader(_dataset, shuffle=True,
                          collate_fn=collate_fn(),
                          batch_size=32)
    num_labels = len(_values['2'])

    model = AutoModelForSequenceClassification.from_pretrained(_config.bert_model,
                                                               num_labels=num_labels).to(_config.device)
    loss_fct = torch.nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_training_steps = len(train_dl) * _config.num_epochs
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

        for batch in train_dl:
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

