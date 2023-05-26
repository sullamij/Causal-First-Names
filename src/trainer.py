from sklearn.metrics import classification_report
import torch
from torch import nn
import os
import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, AutoTokenizer, RobertaModel, RobertaConfig, AdamW
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

from transformers.modeling_outputs import SequenceClassifierOutput
#from transformers.modeling_utils import post_init
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from itertools import chain


class Trainer(object):
    """
    Trainer for training a multiple choice classification model
    """

    def __init__(self, model, optimizer, model_name, device="cpu"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name

    def _print_summary(self):
        print(self.model)
        print(self.optimizer)

    def train(self, loader):
        """
        Run a single epoch of training
        """
        self.model.train()  # Run model in training mode
        loss = None

        epoch_true_labels = []
        epoch_preds = []
        for i, batch in tqdm(enumerate(loader)):
            # clear gradient
            self.optimizer.zero_grad()
            # input_ids shape: (batch_size, num_choices, sequence_length)
            input_ids = batch[0]['input_ids'].to(self.device)
            # input_ids shape: (batch_size, num_choices, sequence_length)
            attention_mask = batch[0]['attention_mask'].to(self.device)
            # labels shape: (batch_size, )
            labels = batch[1].to(self.device)

            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 labels=labels)
            loss, logits = outputs[0], outputs[1]

            epoch_true_labels.extend(labels.tolist())
            epoch_preds.extend(torch.argmax(nn.Softmax(dim=1)(logits), dim=1).tolist())

            # back propagation
            loss.backward()
            # do gradient descent
            self.optimizer.step()

            # Just returning the last loss
        return loss, epoch_true_labels, epoch_preds

    def evaluate(self, loader):
        """
        Evaluate the model on a validation set.
        Only do batch size = 1.
        """

        self.model.eval()  # Run model in eval mode (disables dropout layer)
        loss = None

        epoch_true_labels = []
        epoch_preds = []
        with torch.no_grad():  # Disable gradient computation - required only during training
            for i, batch in tqdm(enumerate(loader)):
                # input_ids shape: (batch_size, num_choices, sequence_length)
                input_ids = batch[0]['input_ids'].to(self.device)
                # input_ids shape: (batch_size, num_choices, sequence_length)
                attention_mask = batch[0]['attention_mask'].to(self.device)
                # labels shape: (batch_size, )
                labels = batch[1].to(self.device)

                outputs = self.model(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     labels=labels)
                loss, logits = outputs[0], outputs[1]

                epoch_true_labels.extend(labels.tolist())
                epoch_preds.extend(torch.argmax(nn.Softmax(dim=1)(logits), dim=1).tolist())

        # Just returning the last loss
        return loss, epoch_true_labels, epoch_preds

    def get_model_dict(self):
        return self.model.state_dict()

    def run_training(self, train_loader, valid_loader, save_location, dataset, n_epochs=3):
        # Useful for us to review what experiment we're running
        # Normally, you'd want to save this to a file
        # self._print_summary()
        losses_valid = []
        losses_train = []
        best_valid = float("inf")
        for i in range(n_epochs):
            saved = False
            target_names = None
            if dataset == 'hs':
                target_names = ['Ending Option 1', 'Ending Option 2', 'Ending Option 3', 'Ending Option 4']
            elif dataset == 'siqa':
                target_names = ['Answer A', 'Answer B', 'Answer C']
            elif dataset == 'anli':
                target_names = ['Hypothesis 1', 'Hypothesis 2']
            elif dataset == 'mnli':
                target_names = ['entailment', 'neutral', 'contradiction']
            elif dataset == 'sst':
                target_names = ['negative', 'positive']

            epoch_loss_train, labels, preds = self.train(train_loader)
            print("Train eval")
            print(classification_report(labels, preds, target_names=target_names))

            epoch_loss_valid, labels, preds = self.evaluate(valid_loader)
            print("Valid eval")
            print(classification_report(labels, preds, target_names=target_names))

            if epoch_loss_valid < best_valid:
                best_valid = epoch_loss_valid
                cur_dir = '..'
                torch.save(self.get_model_dict(),
                           os.path.join(cur_dir, save_location, f'model-{self.model_name}-checkpoint-epoch{i + 1}.pt'))
                saved = True
            if not saved:
                cur_dir = '..'
                torch.save(self.get_model_dict(),
                           os.path.join(cur_dir, save_location, f'model-{self.model_name}-checkpoint-epoch{i + 1}.pt'))

            losses_train.append(epoch_loss_train.tolist())
            losses_valid.append(epoch_loss_valid.tolist())
            print(f"Epoch {i}")
            print(f"Train loss: {epoch_loss_train}")
            print(f"Valid loss: {epoch_loss_valid}")

        train_epoch_idx = range(len(losses_train))
        valid_epoch_idx = range(len(losses_valid))
        # sns.lineplot(epoch_idx, all_losses)
        sns.lineplot(train_epoch_idx, losses_train)
        sns.lineplot(valid_epoch_idx, losses_valid)
        plt.show()