import os
from multiprocessing import cpu_count
from tqdm import tqdm
import numpy as np

from torch.utils.data import Dataset, Subset, DataLoader
import torchvision


# --------------------------------------------------------------------------------------------------------------------

PARSE_CHUNK_SIZE = 16

DATASET_DIR = os.getenv('DATASET_DIR')


# --------------------------------------------------------------------------------------------------------------------

def extend_loader(loader):

    def load(**kwargs):

        kwargs['split'] = 'train' if kwargs.get('train') else 'test'
        kwargs.pop('train')

        dataset = loader(**kwargs)

        if not hasattr(dataset, 'targets'):

            workers = cpu_count()

            dataset_loader = DataLoader(dataset, batch_size=PARSE_CHUNK_SIZE, shuffle=True,
                                        num_workers=workers, collate_fn=None)
            targets = []

            pbar = tqdm(dataset_loader)

            for _, target in pbar:

                targets.extend(target)

                pbar.set_description('parse targets')

            dataset.targets = targets
            dataset.classes = np.unique(targets)

        return dataset

    return load

# --------------------------------------------------------------------------------------------------------------------


DATASET_LOADER = {'cifar10': torchvision.datasets.CIFAR10,
                  'mnist': torchvision.datasets.MNIST,
                  'svhn': extend_loader(torchvision.datasets.SVHN)}

# --------------------------------------------------------------------------------------------------------------------

pil_to_tensor = torchvision.transforms.ToTensor()

# --------------------------------------------------------------------------------------------------------------------


class DatasetResampler(Dataset):

    def __init__(self, src, targets, classes, seed=None):

        self.src = src
        self.subset = src

        self.targets = targets
        self.classes = classes

        self.groups = None
        self.frequency = None
        self.weights = None

        self.random = np.random.RandomState(seed=seed)

        self.reset()
        self.reset_weights()

    def reset(self):

        targets = np.unique(self.targets)

        self.groups = {}

        for target in targets:

            self.groups[target] = self.cluster_set(target)

    def reset_weights(self):

        num_classes = len(self.classes)
        num_examples = len(self.subset)

        self.frequency = {key: len(value) for key, value in self.groups.items()}
        self.weights = {key: (num_classes * self.frequency[key]) / num_examples for key in self.groups}

    def cluster_set(self, target):

        indices = np.where(self.targets == target)[0]

        return indices

    def get_resample_frequency(self, target, weight):

        old_frequency = self.frequency[target]
        new_frequency = (weight / self.weights[target]) * old_frequency

        return int(np.round(new_frequency))

    def sample_indices(self):

        indices = []

        for key in self.groups:

            indices.extend(self.groups[key])

        return indices

    def resample_step(self, target, weight):

        count = self.get_resample_frequency(target, weight)
        replace = (count > self.frequency[target])

        self.groups[target] = self.random.choice(self.groups[target], size=count, replace=replace)
        self.subset = Subset(self.src, self.sample_indices())

        self.weights[target] *= weight

    def resample(self, resample_dict):

        if resample_dict is None:

            return self

        for target, weight in resample_dict.items():

            self.resample_step(target, weight)

        return self

    def __getitem__(self, index):

        example = self.subset[index]

        return example

    def __len__(self):

        return len(self.subset)


# --------------------------------------------------------------------------------------------------------------------

def get_splits(name, download=False, resample_dict=None, use_valid=True, valid_size=0.1, seed=None):

    assert 0 < valid_size < 1, 'valid_size must be in range (0, 1)'

    root = DATASET_DIR + '/' + name
    loader = DATASET_LOADER[name]

    train_dataset = loader(root=root, train=True, transform=pil_to_tensor, target_transform=None, download=download)
    test_dataset = loader(root=root, train=False, transform=pil_to_tensor, target_transform=None, download=download)

    train_targets = np.array(train_dataset.targets)
    test_targets = np.array(test_dataset.targets)

    classes = set(train_dataset.classes).union(set(test_dataset.classes))
    classes = np.array(list(classes))

    splits = {}

    if use_valid:

        random = np.random.RandomState(seed=seed)

        n = len(train_dataset)
        sample_size = int(valid_size * n)

        indices = np.arange(n)
        random.shuffle(indices)

        train_indices, valid_indices = indices[:-sample_size], indices[-sample_size:]

        train_targets, valid_targets = train_targets[train_indices], train_targets[valid_indices]
        train_dataset, valid_dataset = Subset(train_dataset, train_indices), Subset(train_dataset, valid_indices)

        splits['train'] = DatasetResampler(train_dataset, targets=train_targets, classes=classes, seed=seed)
        splits['valid'] = DatasetResampler(valid_dataset, targets=valid_targets, classes=classes, seed=seed)
        splits['test'] = DatasetResampler(test_dataset, targets=test_targets, classes=classes, seed=seed)

    else:

        splits['train'] = DatasetResampler(train_dataset, targets=train_targets, classes=classes, seed=seed)
        splits['test'] = DatasetResampler(test_dataset, targets=test_targets, classes=classes, seed=seed)

    splits['train'] = splits['train'].resample(resample_dict)

    return splits

# --------------------------------------------------------------------------------------------------------------------


# noinspection PyArgumentList
def get_imbalanced_weights(unique_targets, minority_count=4, min_weight=0.001, max_weight=0.01, seed=0):

    random = np.random.RandomState(seed=seed)
    num_classes = len(unique_targets)

    random_states = random.rand(num_classes)
    selected = random_states.argsort()[-minority_count:]

    minority_set = unique_targets[selected]

    weights = random.rand(minority_count)
    weights = weights * (max_weight - min_weight) + min_weight

    resample_dict = dict(zip(minority_set, weights))

    return resample_dict


# --------------------------------------------------------------------------------------------------------------------
