import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    get_scheduler, AdamW, AutoModelForMaskedLM
from .base import Base
from plm.utils import mask_tokens


class ValueClassifier:
    """
        Classifier for Value models
    """

    def __new__(cls, model_path: str, config, logger=None, classifier_type=None):
        """
        """
        return cls.factory(classifier_type=classifier_type)(model_path=model_path,
                                                            config=config)

    @staticmethod
    def factory(classifier_type: str = None):
        """
        factory for mapping method to class
        :param classifier_type:
        :return:
        """
        model_dict = {
            'value-bert': ValueBert,
        }
        return model_dict.get(classifier_type,
                              ValueBert)


class ValueBert(Base):
    """
    Value Classification Using Bert
    """

    def __init__(self, model_path: str, config):
        super(ValueBert).__init__()
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification. \
            from_pretrained(model_path, num_labels=config.num_values)
        self.word_embedding = AutoModelForMaskedLM.from_pretrained(model_path)
        self.word_embedding.to(config.device)
        self.bert_freeze(self.model)
        self.config = config
        self.optimizer = AdamW(params=self.model.parameters(),
                               lr=1e-05,
                               weight_decay=0.01,
                               correct_bias=True
                               )
        self.sim = Similarity(self.config.simTemp)

    def get_tokenizer(self):
        """get tokenizer of classifier"""
        return self.tokenizer

    def get_model(self):
        """get model of classifier"""
        return self.model

    def loss_fn(self, outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs,
                                            targets)

    def train(self, train_dl, valid_dl=None) -> None:
        """
        training model
        :param train_dl: torch validation data loader
        :param valid_dl: valid validation data loader
        :return:
        """
        num_training_steps = len(train_dl) * self.config.num_epochs
        progress_bar = tqdm(range(num_training_steps))
        lr_scheduler = get_scheduler(name="linear",
                                     optimizer=self.optimizer, num_warmup_steps=0,
                                     num_training_steps=num_training_steps)
        print("training phase")
        self.model.to(self.config.device)
        self.model.train()
        for epoch in range(self.config.num_epochs):
            labels, predicts = [], []

            for batch in train_dl:
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                outputs = self.model(input_ids=batch['input_ids'],
                                     attention_mask=batch['attention_mask'],
                                     output_hidden_states=True)
                logits = outputs.logits
                self.optimizer.zero_grad()
                loss = self.loss_fn(logits,
                                    batch['labels'].float())
                if self.config.mlm_loss:
                    mask_ids, mask_lb = mask_tokens(batch['input_ids'].cpu(), self.tokenizer)
                    mlm_batch = {'input_ids': mask_ids.to(self.config.device),
                                 'attention_mask': batch['attention_mask']}

                    loss_mlm = self.mlmForward(mlm_batch,
                                               mask_lb.to(self.config.device))
                    print(loss_mlm)
                    loss += loss_mlm
                if self.config.lossCorRegWeight > 0:
                    last_hidden_state = outputs.hidden_states[-1]
                    covLoss = self.loss_covariance(last_hidden_state[:, 0])
                    # print("covloss",
                    #       covLoss)
                    # print("loss", loss)
                    loss += 0.04 * covLoss

                # dropout-based contrastive learning

                if self.config.lossContrastiveWeight > 0.0:
                    lossDropoutCLLoss = self.calculateDropoutCLLoss(self.model,
                                                                    batch,
                                                                    beforeBatchNorm=True)
                    loss = loss + self.config.lossContrastiveWeight * lossDropoutCLLoss

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               1.0)
                lr_scheduler.step()
                self.optimizer.step()
                self.optimizer.zero_grad()

                predicts.extend(torch.sigmoid(logits).cpu().detach().numpy().tolist())
                labels.extend(batch['labels'].cpu().detach().numpy().tolist())
                progress_bar.update(1)
            self.evaluate_multilabel(dl=valid_dl, usage='VALIDATION')
            self.evaluate_multilabel(dl=train_dl, usage='TRAIN')


class Similarity(torch.nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp
