import numpy as np

import torch
from torch.utils.data import DataLoader

from dos.dataset import get_splits, get_imbalanced_weights
from dos.baseline import Model
from experiments import Handler, set_seed

seed = 0

set_seed(seed)

unique_targets = np.arange(10)
resample_dict = get_imbalanced_weights(unique_targets, minority_count=4, min_weight=0.05, max_weight=0.1)
k_dict = {target: 4 for target in resample_dict}

print('resample_dict = {}\n{}\n k_dict = {}\n{}\n'.format(resample_dict, '='*150, k_dict, '=' * 150))

splits = get_splits(name='mnist', download=False, resample_dict=resample_dict,
                    use_valid=True, valid_size=0.1, seed=seed)


train_dataset = DataLoader(splits['train'], batch_size=16, shuffle=True, num_workers=4, collate_fn=None)
valid_dataset = DataLoader(splits['valid'], batch_size=16, shuffle=False, num_workers=4, collate_fn=None)
test_dataset = DataLoader(splits['test'], batch_size=16, shuffle=False, num_workers=4, collate_fn=None)

model = Model(k_dict=k_dict, num_classes=10, num_channels=1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

handler = Handler(model=model, optimizer=optimizer, num_classes=10, alpha=1.0)

handler.train(train_dataset=train_dataset, valid_dataset=valid_dataset, num_epochs=5, grad_accum_step=1)
handler.evaluate(test_dataset, name='test')

handler.summary(train_dataset, csv_path='./train_summary.csv')
handler.summary(test_dataset, csv_path='./test_summary.csv')

handler.save_checkpoint(path='./ckpt.pth')
