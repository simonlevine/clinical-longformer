
import json
import csv
import torch
from torch.utils.data import Dataset,DataLoader

from loguru import logger
import pytorch_lightning as pl

from torchnlp.utils import collate_tensors, lengths_to_mask

from transformers import AutoTokenizer



def load_mednli(datadir='../data/mednli/'):
    filenames = [
        'mli_train_v1.jsonl',
        'mli_dev_v1.jsonl',
        'mli_test_v1.jsonl',
    ]

    filenames = [datadir+f  for f in filenames]

    mednli_train, mednli_dev, mednli_test = [read_mednli(f) for f in filenames]

    return mednli_train, mednli_dev, mednli_test


def read_mednli(filename) -> list:
    data = []

    with open(filename, 'r') as f:
        for line in f:
            example = json.loads(line)

            premise = (example['sentence1'])
            hypothesis = (example['sentence2'])
            label = example.get('gold_label', None)
            data.append((premise,hypothesis,label))

    print(f'MedNLI file loaded: {filename}, {len(data)} examples')
    return data


class MedNLIDataset(torch.utils.data.Dataset):
    LABEL_TO_ID = {'contradiction': 0, 'entailment': 1, 'neutral': 2}

    def __init__(self, hparams,mednli_data):
        self.hparams=hparams
        self.premises, self.hypotheses, labels = zip(*mednli_data)
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.encoder_model)
        self.labels = [MedNLIDataset.LABEL_TO_ID[l] if l is not None else -1 for l in labels]


        if self.hparams.transformer_type  == 'longformer' or self.hparams.transformer_type == 'roberta-long':
            self.tokenizer = AutoTokenizer.from_pretrainedenizer(
                pretrained_model=self.hparams.encoder_model,
                max_tokens = self.hparams.max_tokens_longformer)
            self.tokenizer.max_len = 4096
 
        else: self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model=self.hparams.encoder_model,
            max_tokens = 512)


    def __getitem__(self, index):
        premise = self.premises[index]
        hypothesis = self.hypotheses[index]
        label = self.labels[index]
        encoded_inputs = self.tokenizer(premise, hypothesis, truncation=True)
        return encoded_inputs, label

    def __len__(self):
        return len(self.labels)


class MedNLIDataModule(pl.LightningDataModule):
        def __init__(self, hparams):
            super().__init__()
            self.hparams = hparams
            if self.hparams.transformer_type == 'longformer':
                self.hparams.batch_size = 1

        def setup(self, stage=None):
            mednli_train, mednli_dev, mednli_test = load_mednli()
            self.train_dataset, self.val_dataset, self.test_dataset = MedNLIDataset(hparams,mednli_train),MedNLIDataset(hparams,mednli_dev),MedNLIDataset(hparams,mednli_test)
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
