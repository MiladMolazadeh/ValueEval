"""
    Base Class for Classification
"""
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import f1_score
from utils import compute_metrics


class Base:
    """
        Base class for Classifier
    """

    def __init__(self) -> None:
        """
            Define child
        """
        self.model = None
        self.config = None
        self.logger = None

    def train(self, train_dl, valid_dl) -> None:
        """
        Args:
            train_dl: torch train data loader
            valid_dl: torch valid data loader
        Returns:
        """

    def ce_loss(self, targets, predicts):
        """
        Cross Entropy Loss
        :param targets: target labels
        :param predicts: model prediction logits
        :return:
        """

    def mlm_loss(self, targets, predicts):
        """
        MLM Loss
        :param targets: target labels
        :param predicts: model prediction logits
        :return:
        """

    def contrastive_isotropy(self):
        """
        contrastive Isotropy

        :return:
        """

    def correlation_isotropy(self):
        """
        Correlation Isotropy
        :return:
        """
    def loss_bce(self, outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs,
                                            targets)

    def evaluate_sequence(self, valid_dl) -> float:
        """
        evaluate model for sequence classification
        :param valid_dl: torch validation dataloader
        :return: f1 macro score
        """
        predictions, target = [], []
        progress_bar = tqdm(range(len(valid_dl)))
        for batch in valid_dl:
            batch = {k: v.to(self.config.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
                prediction = torch.argmax(outputs.logits, dim=-1)
                predictions.extend(list(prediction.cpu().numpy()))
                target.extend(list(batch["labels"].cpu().numpy()))
            progress_bar.update(1)
        return f1_score(predictions, target, average='macro')

    def bert_freeze(self, model, freeze_layer_count=8, emgrad=True):
        if freeze_layer_count:
            # We freeze here the embeddings of the model
            for param in model.deberta.embeddings.parameters():
                param.requires_grad = emgrad

            if freeze_layer_count != -1:
                # if freeze_layer_count == -1, we only freeze the embedding layer
                # otherwise we freeze the first `freeze_layer_count` encoder layers
                for layer in model.deberta.encoder.layer[:freeze_layer_count]:
                    for param in layer.parameters():
                        param.requires_grad = False
        return model

    def getUttEmbeddings(self, X, beforeBatchNorm):
        # BERT forward
        outputs = self.word_embedding(**X, output_hidden_states=True)

        # extract [CLS] for utterance representation
        if beforeBatchNorm:
            CLSEmbedding = outputs.hidden_states[-1][:, 0]
        else:
            CLSEmbedding = outputs.hidden_states[-1][:, 0]
            CLSEmbedding = self.BN(CLSEmbedding)
            CLSEmbedding = self.dropout(CLSEmbedding)

        return CLSEmbedding

    def forwardEmbedding(self, X, beforeBatchNorm=False):
        # get utterances embeddings
        CLSEmbedding = self.getUttEmbeddings(X, beforeBatchNorm=beforeBatchNorm)

        return CLSEmbedding

    def duplicateInput(self, X):
        """

        @brief duplicate input for contrastive learning

        @param X, a dict {'input_ids':[batch, Len], 'token_type_ids': [batch, Len], 'attention_mask':[batch, Len]}

        @return  X, a dict {'input_ids':[2*batch, Len], 'token_type_ids': [2*batch, Len], 'attention_mask':[2*batch, Len]}

        """
        batchSize = X['input_ids'].shape[0]

        X_duplicate = {}
        X_duplicate['input_ids'] = X['input_ids'].unsqueeze(1).repeat(1, 2, 1).view(batchSize * 2, -1)
        # X_duplicate['token_type_ids'] = X['token_type_ids'].unsqueeze(1).repeat(1, 2, 1).view(batchSize * 2, -1)
        X_duplicate['attention_mask'] = X['attention_mask'].unsqueeze(1).repeat(1, 2, 1).view(batchSize * 2, -1)

        return X_duplicate

    def loss_ce(self, logits, Y):
        loss = torch.nn.CrossEntropyLoss()
        output = loss(logits, Y)
        return output

    def calculateDropoutCLLoss(self, model, X, beforeBatchNorm=False):

        """
        @brief calculate dropout-based contrastive loss

        @param model   model
        @param X       X, a dict {'input_ids':[batch, Len], 'token_type_ids': [batch, Len], 'attention_mask':[batch, Len]}
        @param beforeBatchNorm: get embeddings before or after batch norm

        @return  a loss value, tensor        # duplicate input
        """
        batch_size = X['input_ids'].shape[0]
        X_dup = self.duplicateInput(X)

        # get raw embeddings
        batchEmbedding = self.forwardEmbedding(X_dup, beforeBatchNorm=beforeBatchNorm)
        batchEmbedding = batchEmbedding.view((batch_size, 2, batchEmbedding.shape[1]))  # (bs, num_sent, hidden)

        # Separate representation
        z1, z2 = batchEmbedding[:, 0], batchEmbedding[:, 1]

        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        logits = cos_sim

        labels = torch.arange(logits.size(0)).long().to(model.device)
        lossVal = self.loss_ce(logits, labels)

        return lossVal

    def mlmForward(self, X, Y):
        outputs = self.word_embedding(**X, labels=Y)

        return outputs.loss

    def loss_covariance(self, embeddings):
        # covariance
        meanVector = embeddings.mean(dim=0)
        centereVectors = embeddings - meanVector

        # estimate covariance matrix
        featureDim = meanVector.shape[0]
        dataCount = embeddings.shape[0]
        covMatrix = ((centereVectors.t()) @ centereVectors) / (dataCount - 1)

        # normalize covariance matrix
        stdVector = torch.std(embeddings, dim=0)
        sigmaSigmaMatrix = (stdVector.unsqueeze(1)) @ (stdVector.unsqueeze(0))
        normalizedConvMatrix = covMatrix / sigmaSigmaMatrix

        deltaMatrix = normalizedConvMatrix - torch.eye(featureDim).to(self.config.device)

        covLoss = torch.norm(deltaMatrix)  # Frobenius norm

        return covLoss

    def evaluate_multilabel(self, dl, usage):
        predicts, labels = [], []
        with torch.no_grad():
            for batch in dl:
                batch, arg_ids = batch

                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                outputs = self.model(input_ids=batch['input_ids'],
                                     attention_mask=batch['attention_mask'])
                logits = outputs.logits

                predicts.extend(torch.sigmoid(logits).cpu().detach().numpy().tolist())
                labels.extend(batch['labels'].cpu().detach().numpy().tolist())
        f1 = compute_metrics((np.asarray(predicts),
                              np.asarray(labels)))
        print(f"{usage}: score-f1: {f1}")
        return f1

    def save_model(self, path):
        """
        save model as transformers hugging-face format
        :param path: path to model directory
        :return:
        """
        self.model.save_pretrained(path)


