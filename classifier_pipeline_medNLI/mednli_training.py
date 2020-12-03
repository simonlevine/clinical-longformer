"""
Runs a model on a single node across N-gpus.
"""
import argparse
import os
from datetime import datetime
from pathlib import Path

from mednli_classifier import MedNLIClassifier as Classifier
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger
from torchnlp.random import set_seed
# from mednli_data_utils import MedNLIDataModule
import matplotlib.pyplot as plt

import itertools
'''
The trivial solution to Pr = Re = F1 is TP = 0. So we know precision, recall and F1 can have the same value in general
'''


def main(hparams) -> None:
    """
    Main training routine specific for this project
    :param hparams:
    """
    set_seed(hparams.seed)
    # ------------------------
    # 1 INIT LIGHTNING MODEL AND DATA
    # ------------------------

    model = Classifier(hparams)
    

    tb_logger = TensorBoardLogger(
        save_dir='mednli_experiments/',
        version="version_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S"),
        name=f'{hparams.encoder_model}',
    )
    # ------------------------
    # 5 INIT TRAINER
    # ------------------------
    trainer = Trainer(
        logger=tb_logger,
        gpus=hparams.gpus,
        log_gpu_memory="all",
        fast_dev_run=hparams.fast_dev_run,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        max_epochs=hparams.max_epochs,
        default_root_dir=f'./classifier_pipeline_medNLI/{hparams.encoder_model}'
    )

    # ------------------------
    # 6 START TRAINING
    # ------------------------

    trainer.fit(model, model.data)
    trainer.test(model, model.data.test_dataloader())

    cm=model.confusion_matrix.compute().detach().cpu().numpy()
    model_name = model.hparams.encoder_model

    np.save(f'mednli_experiments/{model.hparams.encoder_model}/test_confusion_matrix.npy',cm)
    plot_confusion_matrix(cm,class_names=['entailment','neutral','contradiction'],model=model.hparams.encoder_model)




def plot_confusion_matrix(cm, class_names, model):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes

    credit: https://towardsdatascience.com/exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12
    """

    # cm = cm.cpu().detach().numpy() 
    # font = FontProperties()
    # font.set_family('serif')
    # font.set_name('Times New Roman')
    # font.set_style('normal')

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
    figure.savefig(f'mednli_experiments/{model}/test_mtx.png')


if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    parser = argparse.ArgumentParser(
        description="Minimalist Transformer Classifier",
        add_help=True,
    )
    parser.add_argument("--seed", type=int, default=3, help="Training seed.")
    parser.add_argument(
        "--save_top_k",
        default=1,
        type=int,
        help="The best k models according to the quantity monitored will be saved.",
    )

    # Early Stopping
    parser.add_argument(
        "--monitor", default="val_acc", type=str, help="Quantity to monitor."
    )

    parser.add_argument(
        "--metric_mode",
        default="max",
        type=str,
        help="If we want to min/max the monitored quantity.",
        choices=["auto", "min", "max"],
    )
    parser.add_argument(
        "--patience",
        default=5,
        type=int,
        help=(
            "Number of epochs with no improvement "
            "after which training will be stopped."
        ),
    )

    parser.add_argument(
        "--max_epochs",
        default=10,
        type=int,
        help="Limits training to a max number number of epochs",
    )

    parser.add_argument(
        '--fast_dev_run',
        default=False,
        type=bool,
        help='Run for a trivial single batch and single epoch.'
    )

    # Batching
    parser.add_argument(
        "--batch_size", default=12, type=int, help="Batch size to be used."
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        default=2,
        type=int,
        help=(
            "Accumulated gradients runs K small batches of size N before "
            "doing a backwards pass."
        ),
    )

    # gpu args
    parser.add_argument("--gpus", type=int, default=1, help="How many gpus")

    # each LightningModule defines arguments relevant to it
    parser = Classifier.add_model_specific_args(parser)
    hparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hparams)