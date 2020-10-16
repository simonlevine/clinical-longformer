import pytorch_lightning as pl
import torch
import torch.nn as nn
import transformers
from centerloss import *
from transformers_for_hierarchical import RobertaForHierarchicalClassification

def main():    
    classCount = 14
    model = LightningChestXrayCnnClassifierInceptionV3(classCount)
    trainer = pl.Trainer(max_epochs=15)
    trainer.fit(model)

    hparams = { 
        'train_data_path' : '../data/train_data/',
        'test_data_path'  : '../data/test_data/',
        'val_data_path'   : '../data/val_data/', 

        'verif_data_path' : '../data/verification_data',
        'learning_rate'   : 0.001,
        'batch_size'      : 200,
        'max_tokens'
        }

    model = SiameseModel(hparams)
    if torch.cuda.is_available():
    trainer = pl.Trainer(gpus=1,fast_dev_run=False,max_epochs=10)
    else:
    trainer = pl.Trainer(fast_dev_run=True) #,max_epochs=1)
    trainer.fit(model)


class HierarchicalRobertaModel(nn.Module):
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


class Model(pl.LightningModule):
  
  def __init__(
      self, huggingface_model_url:str, icd_hierarchy_json_fp:str, hparams:dict):

    super(Model,self).__init__()

    self.tokenizer = transformers.AutoTokenizer.from_pretrained(huggingface_model_url)
        # ******* elongated roberta == longformer in all but name! *****
    self.encoder = transformers.AutoModel.from_pretrained(huggingface_model_url)
    # the encoder is a classifier w/ loss
    # self.encoder.classifier = transformers.RobertaForHierarchicalClassification
    self.max_tokens = max_tokens

    if freeze_transformer:
        for param in self.encoder.parameters():
            param.requires_grad = False


    self.intermediary_layers = nn.Sequential(
        Linear(transformer_pooled_outputs)
    )
    
    None  # <- overwritten by subclass
    self.output_layer = None        # <- overwritten by subclass
    self.loss_fn = None             # <- overwritten by subclass

    self.loss_center = CenterLoss(num_classes='FIRST LEVEL OF HIERARCHY', feat_dim=?, use_gpu=True)
    self.loss_layers = torch.nn.CrossEntropyLoss()


    def discharge_summary_encoder(self, discharge_summary):
        '''
        Takes in a single discharge summary.
        Returns a single encoded discharge summary, ready for use as a Transformer input.
        '''
        encoded = self.tokenizer.encode_plus(
            text = discharge_summary,
            text_pair = None,
            add_special_tokens=True,
            max_length=self.max_tokens,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True,
        )

        input_ids, token_type_ids = encoded['input_ids'], encoded['token_type_ids']
        return input_ids, token_type_ids



    def configure_optimizers(self):
        optimizer_centerloss = torch.optim.SGD(center_loss.parameters(), lr=0.5) #alpha = 1
        optimizer_crossentropy = torch.optim.Adam()

    # ... elided for simplicity

    def train_dataloader(self):
        # ... elided for simplicity

    def training_step(self, batch, batch_idx):
        x, y = batch

        input = input.to(dtype=torch.float32, device=self.device)
        varInput = torch.autograd.Variable(input)
        varTarget1 = torch.autograd.Variable(input)
        varTarget2 = torch.autograd.Variable(y)

        varOutput1, varOutput2 = self.module.forward(varInput)
        classifierOut1, classifierOut2 = varOutput2

        # Transformer model center loss:
        lossvalue1 = self.loss1(varOutput1, varTarget1)
        # weighting between main and aux branch of inception model:
        lossvalue2 = 0.8 * self.loss2(classifierOut1, varTarget2) + 0.2 * self.loss2(classifierOut2, varTarget2)  

        losses = [loss_center,loss_val1,loss_val2,loss_val3,loss_val4,loss_val5,loss_val6,loss_val7]
        # weighting btween MSE and BCE respectively:
        loss = sum(losses)

        output = {
        'loss': loss,  # required
        }
        return output


  def get_dataloader(self):
    # Load datasets (this runs download script for the first run)
    train = 
    test = 
    
    # Just split train dataset into train and val, so that we can use val for early stopping.
    train, val = split(train) #NOTE

    batch_size = self.hparams['batch_size']

    # Now the BERT Tokenizer comes!
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    
    # The actual preprocessing is in this `preprocess` function. (it is defined above.)
    preprocessor = partial(preprocess, tokenizer)

    # Apply the preprocessing and make pytorch dataloaders.
    train_dataloader = DataLoader(
            train.map(preprocessor),
            sampler=RandomSampler(train),
            batch_size=batch_size
            )
    val_dataloader = DataLoader(
            val.map(preprocessor),
            sampler=SequentialSampler(val),
            batch_size=batch_size
            )
    test_dataloader = DataLoader(
            test.map(preprocessor),
            sampler=SequentialSampler(test),
            batch_size=batch_size
            )

    return train_dataloader, val_dataloader, test_dataloader



if __name__ == "__main__":
    main()



