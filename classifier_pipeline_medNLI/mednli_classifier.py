# -*- coding: utf-8 -*-
import logging as log
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
import io
# import tensorflow as tf

import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoModel,RobertaForMaskedLM
from transformers.modeling_longformer import LongformerSelfAttention

import pytorch_lightning as pl
from tokenizer import Tokenizer
from torchnlp.encoders import LabelEncoder
from torchnlp.utils import collate_tensors, lengths_to_mask
from utils import mask_fill

import pytorch_lightning.metrics.functional.classification as metrics
from loguru import logger

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from mednli_data_utils import *


def plot_confusion_matrix(cm, class_names, model):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes

    credit: https://towardsdatascience.com/exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12
    """

    cm = cm.cpu().detach().numpy() 
    font = FontProperties()
    font.set_family('serif')
    font.set_name('Times New Roman')
    font.set_style('normal')

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    figure.savefig(f'experiments/{model}/test_mtx.png')


class RobertaLongSelfAttention(LongformerSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        return super().forward(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)

class RobertaLongForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.roberta.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = RobertaLongSelfAttention(config, layer_id=i)


class MedNLIClassifier(pl.LightningModule):
    """
    
    :param hparams: ArgumentParser containing the hyperparameters.
    """
    
    class MedNLIDataModule(pl.LightningDataModule):
        def __init__(self, hparams):
            super().__init__()
            self.hparams = hparams
            if self.hparams.transformer_type == 'longformer':
                self.hparams.batch_size = 1

        def setup(self, stage=None):
            mednli_train, mednli_dev, mednli_test = load_mednli()
            self.train_dataset, self.val_dataset, self.test_dataset = MedNLIDataset(hparams,mednli_train,self.tokenizer),MedNLIDataset(hparams,mednli_dev,self.tokenizer),MedNLIDataset(hparams,mednli_test,self.tokenizer)
            logger.info('MedNLI JSONs loaded...')

        def train_dataloader(self) -> DataLoader:
            logger.warning('Loading training data...')
            return DataLoader(
                dataset=self.train_dataset,
                shuffle=True,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.loader_workers,
            )
        def val_dataloader(self) -> DataLoader:
            logger.warning('Loading validation data...')
            return DataLoader(
                dataset=self.val_dataset,
                shuffle= False,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.loader_workers,
            )
        def test_dataloader(self) -> DataLoader:
            logger.warning('Loading testing data...')
            return DataLoader(
                dataset=self.test_dataset,
                shuffle= False,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.loader_workers,
            )



    def __init__(self, hparams: Namespace) -> None:
        super(MedNLIClassifier,self).__init__()

        self.hparams = hparams
        self.label_vocab_size = 3
        self.batch_size = hparams.batch_size

        if self.hparams.transformer_type  == 'longformer' or self.hparams.transformer_type == 'roberta-long':
            self.tokenizer = AutoTokenizer.from_pretrainedenizer(
                pretrained_model=self.hparams.encoder_model,
                max_tokens = self.hparams.max_tokens_longformer)
            self.tokenizer.max_len = 4096
 
        else: self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model=self.hparams.encoder_model,
            max_tokens = 512)

        
        # Build Data module
        self.data = self.DataModule(self)
        
        # build model
        self.__build_model()

        # Loss criterion initialization.
        self.__build_loss()

        if hparams.nr_frozen_epochs > 0:
            self.freeze_encoder()
        else:
            self._frozen = False
        self.nr_frozen_epochs = hparams.nr_frozen_epochs

        self.test_conf_matrices=[]


    def __build_model(self) -> None:
        """ Init transformer model + tokenizer + classification head."""

        if self.hparams.transformer_type == 'roberta-long':
            self.transformer= RobertaLongForMaskedLM.from_pretrained(
                self.hparams.encoder_model,
                output_hidden_states=True,
                gradient_checkpointing=True
            )

        elif self.hparams.transformer_type == 'longformer':
            self.transformer = AutoModel.from_pretrained(
                self.hparams.encoder_model,
                output_hidden_states=True,
                gradient_checkpointing=True, #critical for training speed.
            )

        else: #BERT
            self.transformer = AutoModel.from_pretrained(
                self.hparams.encoder_model,
                output_hidden_states=True,
                )

        logger.warning(f'model is {self.hparams.encoder_model}')

        if self.hparams.transformer_type == 'longformer':
            logger.warning('Turnin ON gradient checkpointing...')

            self.transformer = AutoModel.from_pretrained(
            self.hparams.encoder_model,
            output_hidden_states=True,
            gradient_checkpointing=True, #critical for training speed.
                )

        else: self.transformer = AutoModel.from_pretrained(
            self.hparams.encoder_model,
            output_hidden_states=True,
                )
        
        # set the number of features our encoder model will return...
        self.encoder_features = 768

        # Classification head
        if self.hparams.single_label_encoding == 'default':
            self.classification_head = nn.Sequential(
                nn.Linear(self.encoder_features, self.encoder_features * 2),
                nn.Tanh(),
                nn.Linear(self.encoder_features * 2, self.encoder_features),
                nn.Tanh(),
                nn.Linear(self.encoder_features, self.label_vocab_size),
            )

        elif self.hparams.single_label_encoding == 'graphical':
            logger.critical('Graphical embedding not yet implemented!')
            # self.classification_head = nn.Sequential(
                #TODO
            # )


    def __build_loss(self):
        """ Initializes the loss function/s. """
        #FOR SINGLE LABELS --> MSE (linear regression) LOSS (like a regression problem)
        # For multiple POSSIBLE discrete single labels, CELoss
        # for many possible categoricla labels, binary cross-entropy (logistic regression for all labels.)
        self._loss = nn.CrossEntropyLoss()

        # self._loss = nn.MSELoss()

    def unfreeze_encoder(self) -> None:
        """ un-freezes the encoder layer. """
        if self._frozen:
            log.info(f"\n-- Encoder model fine-tuning")
            for param in self.transformer.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_encoder(self) -> None:
        """ freezes the encoder layer. """
        for param in self.transformer.parameters():
            param.requires_grad = False
        self._frozen = True

    def predict(self, sample: dict) -> dict: #BUG
        """ Predict function.
        :param sample: dictionary with the text we want to classify.

        Returns:
            Dictionary with the input text and the predicted label.
        """
        if self.training:
            self.eval()

        with torch.no_grad():
            model_input, _ = self.prepare_sample([sample], prepare_target=False)
            model_out = self.forward(**model_input)
            logits = model_out["logits"].numpy()
            predicted_labels = [
                self.data.label_encoder.index_to_token[prediction]
                for prediction in np.argmax(logits, axis=1)
            ]
            sample["predicted_label"] = predicted_labels[0]

        return sample

    def forward(self, tokens, lengths):
        """ Usual pytorch forward function.
        :param tokens: text sequences [batch_size x src_seq_len]
        :param lengths: source lengths [batch_size]

        Returns:
            Dictionary with model outputs (e.g: logits)
        """
        tokens = tokens[:, : lengths.max()]
        # When using just one GPU this should not change behavior
        # but when splitting batches across GPU the tokens have padding
        # from the entire original batch
        mask = lengths_to_mask(lengths, device=tokens.device)

        # Run BERT model.
        word_embeddings = self.transformer(tokens, mask)[0]

        # Average Pooling
        word_embeddings = mask_fill(
            0.0, tokens, word_embeddings, self.tokenizer.padding_index
        )
        sentemb = torch.sum(word_embeddings, 1)
        sum_mask = mask.unsqueeze(-1).expand(word_embeddings.size()).float().sum(1)
        sentemb = sentemb / sum_mask

        return {"logits": self.classification_head(sentemb)}


    def loss(self, predictions: dict, targets: dict) -> torch.tensor:
        """
        Computes Loss value according to a loss function.
        :param predictions: model specific output. Must contain a key 'logits' with
            a tensor [batch_size x 1] with model predictions
        :param labels: Label values [batch_size]

        Returns:
            torch.tensor with loss value.
        """
        return self._loss(predictions["logits"], targets["labels"])


    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ 
        Runs one training step. This usually consists in the forward function followed
            by the loss function.
        
        :param batch: The output of your dataloader. 
        :param batch_nb: Integer displaying which batch this is

        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss_val = self.loss(model_out, targets)
        self.log('loss',loss_val)
        return loss_val



    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ Similar to the training step but with the model in eval mode.

        Returns:
            - dictionary passed to the validation_end function.
        """
        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss_val = self.loss(model_out, targets)

        y = targets["labels"]
        y_hat = model_out["logits"]

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)


        self.log('val_loss',loss_val)

        f1 = metrics.f1_score(labels_hat, y,class_reduction='weighted')
        prec =metrics.precision(labels_hat, y,class_reduction='weighted')
        recall = metrics.recall(labels_hat, y,class_reduction='weighted')
        acc = metrics.accuracy(labels_hat, y,class_reduction='weighted')

        self.log('val_prec',prec)
        self.log('val_f1',f1)
        self.log('val_recall',recall)
        self.log('val_acc_weighted', acc)
        # self.log('val_cm',cm)
        


    def test_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ 
        Runs one training step. This usually consists in the forward function followed
            by the loss function.
        
        :param batch: The output of your dataloader. 
        :param batch_nb: Integer displaying which batch this is

        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        inputs, targets = batch
        model_out = self.forward(**inputs)
        loss_val = self.loss_fn(model_out, targets)  
            
        self.log('test_loss',loss_val)


        y_hat=model_out['logits']
        labels_hat = torch.argmax(y_hat, dim=1)
        y=targets['labels']


        f1 = metrics.f1_score(labels_hat,y,    class_reduction='weighted')
        prec =metrics.precision(labels_hat,y,  class_reduction='weighted')
        recall = metrics.recall(labels_hat,y,  class_reduction='weighted')
        acc = metrics.accuracy(labels_hat,y,   class_reduction='weighted')

        self.log('test_batch_prec',prec)
        self.log('test_batch_f1',f1)
        self.log('test_batch_recall',recall)
        self.log('test_batch_weighted_acc', acc)

        cm = metrics.confusion_matrix(pred = labels_hat,target=y,normalize=False)
        self.test_conf_matrices.append(cm)    

    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        parameters = [
            {"params": self.classification_head.parameters()},
            {
                "params": self.transformer.parameters(),
                "lr": self.hparams.encoder_learning_rate,
            },
        ]
        optimizer = optim.Adam(parameters, lr=self.hparams.learning_rate)
        return [optimizer], []

    def on_epoch_end(self):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()
    
    @classmethod
    def add_model_specific_args(
        cls, parser: ArgumentParser
    ) -> ArgumentParser:
        """ Parser for Estimator specific arguments/hyperparameters. 
        :param parser: argparse.ArgumentParser

        Returns:
            - updated parser
        """
        parser.add_argument(
            "--encoder_model",
            default= 'bert-base-uncased',# 'emilyalsentzer/Bio_ClinicalBERT',# 'allenai/biomed_roberta_base',#'simonlevine/biomed_roberta_base-4096-speedfix', # 'bert-base-uncased',
            type=str,
            help="Encoder model to be used.",
        )

        parser.add_argument(
            "--transformer_type",
            default='bert', #'longformer', roberta-long
            type=str,
            help="Encoder model /tokenizer to be used (has consequences for tokenization and encoding; default = longformer).",
        )

        parser.add_argument(
            "--single_label_encoding",
            default='default',
            type=str,
            help="How should labels be encoded? Default for torch-nlp label-encoder...",
        )
        
        parser.add_argument(
            "--max_tokens_longformer",
            default=4096,
            type=int,
            help="Max tokens to be considered per instance..",
        )
        parser.add_argument(
            "--max_tokens",
            default=512,
            type=int,
            help="Max tokens to be considered per instance..",
        )
        parser.add_argument(
            "--encoder_learning_rate",
            default=1e-05,
            type=float,
            help="Encoder specific learning rate.",
        )
        parser.add_argument(
            "--learning_rate",
            default=3e-05,
            type=float,
            help="Classification head learning rate.",
        )
        parser.add_argument(
            "--nr_frozen_epochs",
            default=0,
            type=int,
            help="Number of epochs we want to keep the encoder model frozen.",
        )
        # parser.add_argument(
        #     "--train_csv",
        #     default="data/intermediary-data/notes2diagnosis-icd-train.csv",
        #     type=str,
        #     help="Path to the file containing the train data.",
        # )
        # parser.add_argument(
        #     "--dev_csv",
        #     default="data/intermediary-data/notes2diagnosis-icd-validate.csv",
        #     type=str,
        #     help="Path to the file containing the dev data.",
        # )
        # parser.add_argument(
        #     "--test_csv",
        #     default="data/intermediary-data/notes2diagnosis-icd-test.csv",
        #     type=str,
        #     help="Path to the file containing the dev data.",
        # )
        parser.add_argument(
            "--loader_workers",
            default=8,
            type=int,
            help="How many subprocesses to use for data loading. 0 means that \
                the data will be loaded in the main process.",
        )
        return parser
