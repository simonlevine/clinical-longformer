import torch
from torch import nn
from typing import List
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelWithLMHead, AdamW, get_linear_schedule_with_warmup
import logging
import os
from functools import lru_cache
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from argparse import Namespace
from sklearn.metrics import classification_report
from torch_lr_finder import LRFinder


def main():
    tokenizer = AutoTokenizer.from_pretrained("marrrcin/PolBERTa-base-polish-cased-v1")
    model     = AutoModelWithLMHead.from_pretrained("marrrcin/PolBERTa-base-polish-cased-v1")

    classifier = PolBERTaSentimentModel(AutoModelWithLMHead.from_pretrained("marrrcin/PolBERTa-base-polish-cased-v1").roberta, 3)
    X = torch.tensor(enc["input_ids"]).unsqueeze(0)
    attn = torch.tensor(enc["attention_mask"]).unsqueeze(0)


    train_path, val_path, test_path = [f"dataset_conll/all.text.{d}.txt" for d in ("train", "dev", "test")]

    df = pd.read_csv(train_path, sep="__label__", header=None, names=["text", "label"], engine="python")


    hparams_tmp = Namespace(
    train_path=train_path,
    val_path=val_path,
    test_path=test_path,
    batch_size=8,
    warmup_steps=100,
    epochs=3,
    lr=lr,
    accumulate_grad_batches=1,
    )

    # finding_LR:

    module = TrainingModule(hparams_tmp)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(module.parameters(), lr=5e-7)
    lr_finder = LRFinder(module, optimizer, criterion, device="cuda")
    lr_finder.range_test(module.train_dataloader(), end_lr=100, num_iter=100, accumulation_steps=hparams_tmp.accumulate_grad_batches)
    lr_finder.plot()
    lr_finder.reset()

    #atual training
    hparams = Namespace(
    train_path=train_path,
    val_path=val_path,
    test_path=test_path,
    batch_size=8,
    warmup_steps=100,
    epochs=3,
    lr=lr,
    accumulate_grad_batches=1
    )
    module = TrainingModule(hparams)

    trainer = pl.Trainer(gpus=1, max_epochs=hparams.epochs, progress_bar_refresh_rate=10,
                     accumulate_grad_batches=hparams.accumulate_grad_batches)

    trainer.fit(module)


    #evaluation...

    with torch.no_grad():
    progress = ["/", "-", "\\", "|", "/", "-", "\\", "|"]
    module.eval()
    true_y, pred_y = [], []
    for i, batch_ in enumerate(module.test_dataloader()):
        (X, attn), y = batch_
        batch = (X.cuda(), attn.cuda())
        print(progress[i % len(progress)], end="\r")
        y_pred = torch.argmax(module(batch), dim=1)
        true_y.extend(y.cpu())
        pred_y.extend(y_pred.cpu())
        
    print("\n" + "_" * 80)
    print(classification_report(true_y, pred_y, target_names=label2int.keys(), digits=6))





class MyRobertaModel(nn.Module):
    def __init__(self, base_model, n_classes, base_model_output_size=768, dropout=0.05):
        super().__init__()
        self.base_model = base_model
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_model_output_size, base_model_output_size),
            Mish(),
            nn.Dropout(dropout),
            nn.Linear(base_model_output_size, n_classes)
        )
        
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=0.02)
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def forward(self, input_, *args):
        X, attention_mask = input_
        hidden_states, _ = self.base_model(X, attention_mask=attention_mask)
        
        # here I use only representation of <s> token, but you can easily use more tokens,
        # maybe do some pooling / RNNs... go crazy here!
        return self.classifier(hidden_states[:, 0, :])



class TokenizersCollateFn:
    def __init__(self, max_tokens=512):
        t = ByteLevelBPETokenizer(
            "tokenizer/vocab.json",
            "tokenizer/merges.txt"
        )
        t._tokenizer.post_processor = BertProcessing(
            ("</s>", t.token_to_id("</s>")),
            ("<s>", t.token_to_id("<s>")),
        )
        t.enable_truncation(max_tokens)
        t.enable_padding(max_length=max_tokens, pad_id=t.token_to_id("<pad>"))
        self.tokenizer = t

    def __call__(self, batch):
        encoded = self.tokenizer.encode_batch([x[0] for x in batch])
        sequences_padded = torch.tensor([enc.ids for enc in encoded])
        attention_masks_padded = torch.tensor([enc.attention_mask for enc in encoded])
        labels = torch.tensor([x[1] for x in batch])
        
        return (sequences_padded, attention_masks_padded), labels


class PolEmo2Dataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data_column = "text"
        self.class_column = "class"
        self.data = pd.read_csv(path, sep="__label__", header=None, names=[self.data_column, self.class_column],
                               engine="python")

    def __getitem__(self, idx):
        return self.data.loc[idx, self.data_column], label2int[self.data.loc[idx, self.class_column]]

    def __len__(self):
        return self.data.shape[0]


class TrainingModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.model = PolBERTaSentimentModel(AutoModelWithLMHead.from_pretrained("marrrcin/PolBERTa-base-polish-cased-v1").roberta, 4)
        self.loss = nn.CrossEntropyLoss()
        self.hparams = hparams

    def step(self, batch, step_name="train"):
        X, y = batch
        loss = self.loss(self.forward(X), y)
        loss_key = f"{step_name}_loss"
        tensorboard_logs = {loss_key: loss}

        return { ("loss" if step_name == "train" else loss_key): loss, 'log': tensorboard_logs,
               "progress_bar": {loss_key: loss}}

    def forward(self, X, *args):
        return self.model(X, *args)

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def validation_end(self, outputs: List[dict]):
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"val_loss": loss}
        
    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")
    
    def train_dataloader(self):
        return self.create_data_loader(self.hparams.train_path, shuffle=True)

    def val_dataloader(self):
        return self.create_data_loader(self.hparams.val_path)

    def test_dataloader(self):
        return self.create_data_loader(self.hparams.test_path)
                
    def create_data_loader(self, ds_path: str, shuffle=False):
        return DataLoader(
                    PolEmo2Dataset(ds_path),
                    batch_size=self.hparams.batch_size,
                    shuffle=shuffle,
                    collate_fn=TokenizersCollateFn()
        )
        
    @lru_cache()
    def total_steps(self):
        return len(self.train_dataloader()) // self.hparams.accumulate_grad_batches * self.hparams.epochs

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.hparams.lr)
        lr_scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.hparams.warmup_steps,
                    num_training_steps=self.total_steps(),
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]