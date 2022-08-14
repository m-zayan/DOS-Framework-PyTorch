import copy

import torch

# -------------------------------------------------------------------------------------------------------------------


class Accumulator:

    def __init__(self, **kwargs):

        self.state = {}
        self.state.update(copy.deepcopy(kwargs))

        self.reset()

    def reset(self):

        raise NotImplementedError

    def update(self, *args, **kwargs):

        raise NotImplementedError

    def compute(self):

        raise NotImplementedError


# -------------------------------------------------------------------------------------------------------------------

class LossAccumulator(Accumulator):

    def __init__(self):

        super().__init__()

    def reset(self):

        self.state.update({'step': 0, 'loss': 0.0})

    def update(self, loss):

        self.state['loss'] += loss
        self.state['step'] += 1

    def compute(self):

        return self.state['loss'] / self.state['step']

# -------------------------------------------------------------------------------------------------------------------


class ConfusionMatrix(Accumulator):

    def __init__(self, num_classes, epsilon=1e-8):

        self.epsilon = torch.as_tensor(epsilon, dtype=torch.float32)

        super().__init__(num_classes=num_classes)

    @property
    def num_classes(self):

        return self.state['num_classes']

    def reset(self):

        cm = torch.zeros((self.num_classes, self.num_classes), dtype=torch.long)

        self.state.update({'num_examples': 0, 'cm': cm})

    def update(self, y_pred, y):

        batch_size = y.size(0)

        y_pred = y_pred.flatten()
        y = y.flatten()

        flat_index = self.num_classes * y + y_pred

        cm = torch.bincount(flat_index, minlength=self.num_classes ** 2)

        self.state['num_examples'] += batch_size
        self.state['cm'] += cm.view(self.num_classes, self.num_classes)

    def compute(self, average=False):

        if average:

            cm = self.state['cm'].float() / float(self.state['num_examples'])

            return cm

        return self.state['cm']

    def true_positive(self, average=False):

        cm = self.compute(average)

        return torch.diag(cm, diagonal=0)

    def total_positive(self, average=False):

        cm = self.compute(average)

        return torch.sum(cm, dim=0)

    def false_positive(self, average=False):

        return self.total_positive(average) - self.true_positive(average)

    def total_negative(self, average=False):

        cm = self.compute(average)

        return torch.sum(cm, dim=1)

    def false_negative(self, average=False):

        return self.total_negative(average) - self.true_positive(average)

    def precision(self, average='macro'):

        assert average in {'macro', 'micro'}, f'average={average}, is undefined or not yet supported'

        true = self.true_positive(False)
        total = self.total_positive(False)

        if average == 'micro':

            true = true.sum()
            total = total.sum()

        return true / torch.maximum(total, self.epsilon)

    def recall(self,  average='macro'):

        true = self.true_positive(False)
        total = self.total_negative(False)

        if average == 'micro':

            true = true.sum()
            total = total.sum()

        return true / torch.maximum(total, self.epsilon)

    def f1_score(self, average='macro', reduce=True):

        p = self.precision(average)
        r = self.recall(average)

        score = (2.0 * p * r) / torch.maximum(p + r, self.epsilon)

        return score.mean() if reduce else score

# -------------------------------------------------------------------------------------------------------------------
