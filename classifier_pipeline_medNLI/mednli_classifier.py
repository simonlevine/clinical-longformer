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
#from longformer.longformer import LongformerSelfAttention

import pytorch_lightning as pl
from tokenizer import Tokenizer
from torchnlp.encoders import LabelEncoder
from torchnlp.utils import collate_tensors, lengths_to_mask
from utils import mask_fill

import pytorch_lightning.metrics.functional as metrics
from loguru import logger
from pytorch_lightning.metrics.classification.confusion_matrix import ConfusionMatrix
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from mednli_data_utils import *




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
    Sample model to show how to use a Transformer model to classify sentences.
    
    :param hparams: ArgumentParser containing the hyperparameters.
    """

    # *************************
    class DataModule(pl.LightningDataModule):
        def __init__(self, classifier_instance):
            super().__init__()
            self.hparams = classifier_instance.hparams


            if self.hparams.transformer_type == 'longformer':
                self.hparams.batch_size = 1
            self.classifier = classifier_instance

            self.transformer_type = self.hparams.transformer_type

            
             # Label Encoder
            self.label_encoder = LabelEncoder(
                ['contradiction', 'entailment', 'neutral'], 
                reserved_labels=[]
            )

            self.label_encoder.unknown_index = None

        def get_mednli_data(self, raw_data:list) -> list: #REWRITE
            """ Reads a comma separated value file.

            :param path: path to a csv file.
            
            :return: List of records as dictionaries
            """
        
            df = pd.DataFrame(raw_data)
            df = df.rename(columns={0:'premise', 1:'hypothesis',2:'label'})

            df["text"] = df["premise"].astype(str) + df["hypothesis"].astype(str)
            df["label"] = df["label"].astype(str)
            df = df.drop(['premise','hypothesis'],axis=1)
            return df.to_dict("records")


        def setup(self, stage=None):
            self.mednli_train, self.mednli_dev, self.mednli_test = load_mednli()  
            
        def train_dataloader(self) -> DataLoader:
            """ Function that loads the train set. """
            logger.warning('Loading training data...')
            self._train_dataset = self.get_mednli_data(self.mednli_train)
            return DataLoader(
                dataset=self._train_dataset,
                sampler=RandomSampler(self._train_dataset),
                batch_size=self.hparams.batch_size,
                collate_fn=self.classifier.prepare_sample,
                num_workers=self.hparams.loader_workers,
            )

        def val_dataloader(self) -> DataLoader:
            logger.warning('Loading validation data...')

            """ Function that loads the validation set. """
            self._dev_dataset = self.get_mednli_data(self.mednli_dev)
            return DataLoader(
                dataset=self._dev_dataset,
                batch_size=self.hparams.batch_size,
                collate_fn=self.classifier.prepare_sample,
                num_workers=self.hparams.loader_workers,
            )

        def test_dataloader(self) -> DataLoader:
            logger.warning('Loading testing data...')

            """ Function that loads the validation set. """
            self._test_dataset = self.get_mednli_data(self.mednli_test)

            return DataLoader(
                dataset=self._test_dataset,
                batch_size=self.hparams.batch_size,
                collate_fn=self.classifier.prepare_sample,
                num_workers=self.hparams.loader_workers,
            )

#    ****************

    def __init__(self, hparams: Namespace) -> None:
        super(MedNLIClassifier,self).__init__()

        self.hparams = hparams
        self.batch_size = hparams.batch_size

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

        self.confusion_matrix= ConfusionMatrix(num_classes=3)

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

        # Tokenizer
        if self.hparams.transformer_type  == 'longformer' or self.hparams.transformer_type == 'roberta-long':
            self.tokenizer = Tokenizer(
                pretrained_model=self.hparams.encoder_model,
                max_tokens = self.hparams.max_tokens_longformer)
            self.tokenizer.max_len = 4096
 

        else: self.tokenizer = Tokenizer(
            pretrained_model=self.hparams.encoder_model,
            max_tokens = 512)

           #others:
             #'emilyalsentzer/Bio_ClinicalBERT' 'simonlevine/biomed_roberta_base-4096-speedfix'

        # Classification head
        if self.hparams.single_label_encoding == 'default':
            self.classification_head = nn.Sequential(

                nn.Linear(self.encoder_features, self.encoder_features * 2),
                nn.Tanh(),
                nn.Linear(self.encoder_features * 2, self.encoder_features),
                nn.Tanh(),
                nn.Linear(self.encoder_features, self.data.label_encoder.vocab_size),

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

    def predict(self, sample: dict) -> dict:
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

    def prepare_sample(self, sample: list, prepare_target: bool = True) -> (dict, dict):
        """
        Function that prepares a sample to input the model.
        :param sample: list of dictionaries.
        
        Returns:
            - dictionary with the expected model inputs.
            - dictionary with the expected target labels.
        """
        sample = collate_tensors(sample)
        
        tokens, lengths = self.tokenizer.batch_encode(sample["text"])

        inputs = {"tokens": tokens, "lengths": lengths}

        if not prepare_target:
            return inputs, {}

        # Prepare target:
        try:
            targets = {"labels": self.data.label_encoder.batch_encode(sample["label"])}
            return inputs, targets
        except RuntimeError:
            raise Exception("Label encoder found an unknown label.")

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

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)


        self.log('loss',loss_val)

        # can also return just a scalar instead of a dict (return loss_val)
        return loss_val


    
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
        loss_val = self.loss(model_out, targets)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
            
        self.log('test_loss',loss_val)


        y_hat=model_out['logits']
        labels_hat = torch.argmax(y_hat, dim=1)
        y=targets['labels']


        f1 = metrics.f1(labels_hat,y, average='weighted',         num_classes=3)
        prec =metrics.precision(labels_hat,y,  class_reduction='weighted',num_classes=3)
        recall = metrics.recall(labels_hat,y,  class_reduction='weighted',num_classes=3)
        acc = metrics.accuracy(labels_hat,y,   class_reduction='weighted',num_classes=3)

        self.confusion_matrix.update(labels_hat,y)
        self.log('test_batch_prec',prec)
        self.log('test_batch_f1',f1)
        self.log('test_batch_recall',recall)
        self.log('test_batch_weighted_acc', acc)



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

        if self.on_gpu:
            val_acc = val_acc.cuda(loss_val.device.index)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)
            val_acc = val_acc.unsqueeze(0)


        self.log('val_loss',loss_val)

        f1 = metrics.f1(labels_hat, y,average='weighted',num_classes=3)
        prec = metrics.precision(labels_hat, y,class_reduction='weighted',num_classes=3)
        recall = metrics.recall(labels_hat, y,class_reduction='weighted',num_classes=3)
        acc = metrics.accuracy(labels_hat, y,class_reduction='weighted',num_classes=3)

        self.log('val_prec',prec)
        self.log('val_f1',f1)
        self.log('val_recall',recall)
        self.log('val_acc_weighted', acc)
        

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
            default= 'simonlevine/bioclinical-roberta-long', #'emilyalsentzer/Bio_ClinicalBERT',# 'allenai/biomed_roberta_base',', # 'bert-base-uncased',
            type=str,
            help="Encoder model to be used.",
        )

        parser.add_argument(
            "--transformer_type",
            default='roberta-long', #'bert', roberta, or roberta-long
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

        parser.add_argument(
            "--loader_workers",
            default=8,
            type=int,
            help="How many subprocesses to use for data loading. 0 means that \
                the data will be loaded in the main process.",
        )
        return parser
