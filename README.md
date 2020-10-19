# 11785-project (***WIP***)

Sandbox: https://colab.research.google.com/drive/1n0vaflnGpRpEnRAgwkCOLqVFRr1p1S3L?usp=sharing

# To do:
- Update logging for PT-lighting: see https://pytorch-lightning.readthedocs.io/en/latest/logging.html
- elongate bioclinical_BERT
- get multilabel training working on BERT
- get ROBERTA working,single and multilabel case
- build a hierarchical classifier head
- get x-transformer trained on our best model.


## Running the code

### Data

MIMIC-III data requires credentialed access. This pipeline requires:

- D_ICD_DIAGNOSES.csv.gz (if descriptions of ICDs are wanted)
- D_ICD_PROCEDURES.csv.gz
- NOTEEVENTS.csv.gz (contains, among others, discharge summary texts)
- DIAGNOSES_ICD.csv.gz (contains actual patient visit code assignment)
- PROCEDURES_ICD.csv.gz

We also provide a General Equivalence Mapping if translating codes to ICD10 is desired (these data are relevant but quite old, MIMIC-IV is in the works).

### Preprocess the Data

Data should be in the main working directory, in the "data" folder.

In preprocessing_pipeline, run:
- format_notes.py 
  - cleans discharge summaries of administrative language, etc.
  - experiments with Dask, multiprocessing, etc., were not fruitful. Vectorized operations here still take a few minutes.
  
- format_data_for_training.py , OR, format_data_for_for_multilabel.py
  - depending on your use-case
  
 Finally, in longformer_gen, if you don't want to pull our models from https://huggingface.co/simonlevine: 
 
 - roberta_to_longformer.py
    - if you want to convert your own Roberta to Longformer (with the Longformer speed-fix)
 - elongate_bert.py
    - if you want to model global document token dependencies using BERT.

### Run training

Run the pytorch-lightning trainer from either training_multilabel.py or training_onelabel.py.

# Background


This project is the course project for Fall 2020, 11-785: Deep Learning, at Carnegie Mellon University.

Here, we benchmark various Transformer-based encoders (BERT, RoBERTa, Longformer) on electronic health records data from the MIMIC-III data lake.
We also benchmark our new pipeline against another novel pipeline developed in 2020 by Simon Levine and Jeremy Fisher, auto-icd-Transformers.
- The x-transformer pipeline uses the state-of-the-art extreme multilabel classification with label clustering and PIFA-TFIDF features to amke good use of label descriptions. ICD code labels not only have 'ICD9_CODE' labels, but also 'LONG_TITLE' labels with rich semantic information.
- However, this ignores the natural ICD hierarchy, hence this project.

Though our pipeline can handle any category of ICU note events present in MIMIC, we focus on Discharge Summaries with ICD-9-CM and ICD-9-PCS codes as outputs.
Data preprocessing scripts are included in this repository with novel logic to filter out administrative language and extraneous characters.

We use base-BERT-uncased as a baseline transformer. We then try bio-clinical-BERT, biomed-RoBERTa, and finally our bespoke "Longformer" biomed-RoBERTa-4096-speedfix.
This latter model is simply allenAI's biomed-roberta with global attention, such that documents of up to 4096 token length are able to be used without truncation, a critical aspect of free-text EHR data).

We choose not to pre-train the Longformer on MIMIC data as
1) This is costly
2) The corpus is likely not sufficient to have dramatic increases in performance, per AllenAI, and
3) out focus is more on developing a good, new classifier head as high fidelity encoders are largely a solved/solvable issue.


## Transformers

For each transformer, we build a classifier head. As a baseline,, we use

nn.Linear(self.encoder_features, self.encoder_features * 2),
  nn.Tanh(),
  nn.Linear(self.encoder_features * 2, self.encoder_features),
  nn.Tanh(),
  nn.Linear(self.encoder_features, self.data.label_encoder.vocab_size),


where encoder features = 768, vocab_size is simply the number of unique ICDs in the training set.

## Labels and Label Manipulation

We ignore label descriptions (see above: this direction was attempted previously by Simon Levine and Jeremy Fisher, adapting the X-transformer pipeline).

### One Label
For ICD-10-CM and ICD-10-PCS, the single-label case is first analyzed. That is, ICD "SEQ_NUM==1" is looked at, as this is generally the most subjectively important code assigned per patient visit.

As such, single labels are then modeled hierarchcally, as ICDs are by nature a hierarchical scheme with a classifier head learning class-hierarchy rather than a flat representation.
This should produce better results due to the severe sparsity of some labels in MIMIC and in life.

Lastly, we attempt a continuous embedding of single labels using Jeremy Fisher's excellent ICD-codex project, predicting a vector and learning the nearest-neighbor ICD.

### Multi-Label

For the multi-label case, the label space grows. Hence, an "extreme multilabel" classifier is required.
We begin with one-hot encoded ICD codes per instance, with our base 3-layer dense classifier.

We then move to more exotic methods for classification in this domain, implementing a small selection of hierarchical classification schemes in the (extreme) multilabel case.


