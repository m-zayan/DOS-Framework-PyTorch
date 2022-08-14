import os, gc

import random as pyrand

import pandas as pd
from tqdm import tqdm
import numpy as np

import torch

from external.metrics import LossAccumulator, ConfusionMatrix


# -------------------------------------------------------------------------------------------------------------------


def get_device():

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    return device

# -------------------------------------------------------------------------------------------------------------------


def set_seed(torch_seed=0, np_seed=None, py_seed=None):

    os.environ['PYTHONHASHSEED'] = str(py_seed) if py_seed else str(torch_seed)
    pyrand.seed(py_seed if py_seed else torch_seed)
    np.random.seed(np_seed if np_seed else torch_seed)
    torch.manual_seed(torch_seed)

# -------------------------------------------------------------------------------------------------------------------


class Handler:

    def __init__(self, model, optimizer, num_classes, alpha=1.0, device=None):

        self.num_classes = num_classes

        self.model = model
        self.optimizer = optimizer

        self.alpha = alpha

        self.loss_accum = LossAccumulator()
        self.cm = ConfusionMatrix(num_classes=num_classes)

        self.device = device if device else get_device()

    # ================================================================================================================

    def train(self, train_dataset, valid_dataset, num_epochs, grad_accum_step=1):

        self.model.train()

        for epoch in range(num_epochs):

            self.loss_accum.reset()
            self.cm.reset()

            pbar = tqdm(enumerate(train_dataset))

            size = len(train_dataset)

            for idx, (inputs, targets) in pbar:

                images = inputs.to(self.device)
                labels = targets.to(self.device)

                logits, (loss, aux_loss) = self.model(images, labels)

                loss = (loss + self.alpha * aux_loss) / grad_accum_step
                loss.backward()

                if ((idx + 1) % grad_accum_step == 0) or (idx + 1 == size):

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                with torch.no_grad():

                    self.loss_accum.update(loss.item() * grad_accum_step)

                    total_loss = self.loss_accum.compute()
                    metrics_log = self.evaluate_step(logits, labels)

                log = 'train | {}/{}: loss = {:.4f}, '.format(idx + 1, size, total_loss)
                log += metrics_log

                pbar.set_description(log)

                del inputs, targets, logits
                gc.collect()

            self.evaluate(valid_dataset)

    # ================================================================================================================

    def evaluate_step(self, logits, labels):

        y = labels.cpu()
        y = torch.as_tensor(y, dtype=torch.long)

        y_pred = torch.softmax(logits, dim=1).cpu().argmax(dim=1)

        self.cm.update(y_pred, y)

        score = self.cm.f1_score(average='macro')

        metrics_log = 'f1 score = {:.4f}'.format(score)

        return metrics_log

    # ================================================================================================================

    def evaluate(self, dataset, name='valid', device=None):

        self.model.eval()

        self.loss_accum.reset()
        self.cm.reset()

        pbar = tqdm(enumerate(dataset))

        size = len(dataset)

        for idx, (inputs, targets) in pbar:

            images = inputs.to(device)
            labels = targets.to(device)

            logits, (loss, aux_loss) = self.model(images, labels)
            loss = loss + self.alpha * aux_loss

            self.loss_accum.update(loss.item())

            total_loss = self.loss_accum.compute()
            metrics_log = self.evaluate_step(logits, labels)

            log = '{} | {}/{}: loss = {:.4f}, '.format(name, idx + 1, size, total_loss) + metrics_log

            pbar.set_description(log)

            del inputs, targets, logits
            gc.collect()

    # ================================================================================================================

    def summary(self, dataset, csv_path, device=None):

        self.model.eval()

        self.loss_accum.reset()
        self.cm.reset()

        pbar = tqdm(enumerate(dataset))

        y, y_pred = [], []

        for idx, (inputs, targets) in pbar:

            images = inputs.to(device)
            labels = targets.to(device).cpu()

            logits = self.model(images)
            predictions = torch.softmax(logits, dim=1).cpu().argmax(axis=-1)

            y.append(labels)
            y_pred.append(predictions)

            del inputs, targets, logits
            gc.collect()

        y = torch.concat(y, dim=0)
        y_pred = torch.concat(y_pred, dim=0)

        self.cm.update(y, y_pred)

        scores = self.cm.f1_score(average='macro', reduce=False).detach().numpy()
        frequencies = np.bincount(y.detach().numpy(), minlength=None)

        summary_dict = {'label': [], 'frequency': [], 'score': []}

        for label, score in enumerate(scores):

            summary_dict['label'].append(label)
            summary_dict['frequency'].append(frequencies[label])
            summary_dict['score'].append(score)

        pd.DataFrame(summary_dict).to_csv(csv_path, index=False, encoding='utf-8')

    # ================================================================================================================

    def save_checkpoint(self, path):

        torch.save(self.model.state_dict(), path)

    # ================================================================================================================

    def load_checkpoint(self, path):

        self.model.load_state_dict(torch.load(path))
