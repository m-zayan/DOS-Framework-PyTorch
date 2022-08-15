import torch
from torch import nn

from external import common

# --------------------------------------------------------------------------------------------------------------------


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, act=None, use_norm=False, downsample=False):

        super(ConvBlock, self).__init__()

        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same')]

        if use_norm:

            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(act if act else nn.ReLU(inplace=True))

        if downsample:

            layers.append(nn.MaxPool2d(kernel_size=(2, 2)))

        self.block = nn.Sequential(*layers)

    def forward(self, x):

        return self.block(x)


# --------------------------------------------------------------------------------------------------------------------

class FCBlock(nn.Module):

    def __init__(self, in_features, out_features, act=None, use_norm=False):

        super(FCBlock, self).__init__()

        layers = [nn.Linear(in_features, out_features)]

        if use_norm:

            layers.append(nn.BatchNorm1d(out_features))

        layers.append(act if act else nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):

        return self.block(x)

# --------------------------------------------------------------------------------------------------------------------


class DeepFeatureKNN(nn.Module):

    def __init__(self, distance_fn=None):

        super(DeepFeatureKNN, self).__init__()

        self.distance_fn = distance_fn

    def forward(self, embeddings, subset, k, return_distances=False):

        assert k > 1, 'k must be greater than 1'

        embeddings_flat = common.permute_for_gather(embeddings, axis=[0], last=True)
        embeddings_flat = common.flatten_for_gather(embeddings_flat, axis=[0], last=True)
        embeddings_flat = embeddings_flat.unsqueeze(2)

        subset_flat = common.permute_for_gather(subset, axis=[0], last=True)
        subset_flat = common.flatten_for_gather(subset_flat, axis=[0], last=True)
        subset_flat = subset_flat.unsqueeze(1)

        if self.distance_fn:

            distances = self.distance_fn(embeddings_flat, subset_flat)

        else:

            distances = ((embeddings_flat - subset_flat) ** 2.0).sum(0)

        top_k = torch.topk(distances, k=k, largest=False, sorted=True, dim=0)

        indices = torch.moveaxis(top_k.indices, 0, -1)
        distances = torch.moveaxis(top_k.values, 0, -1)

        matches = embeddings[indices]

        if return_distances:

            return matches, distances

        return matches

# --------------------------------------------------------------------------------------------------------------------


class BasicEncoder(nn.Module):

    def __init__(self, num_channels, hidden_sizes, fc_hidden_sizes, use_skip=True, rate=2, use_norm=False):

        super(BasicEncoder, self).__init__()

        conv_blocks = []
        fc_blocks = []
        skip_blocks = []

        in_features = num_channels

        for idx, dim in enumerate(hidden_sizes):

            is_nxt = ((idx + 1) % rate == 0)

            conv_blocks.append(ConvBlock(in_features, dim, downsample=is_nxt, use_norm=use_norm))

            if use_skip and is_nxt:

                skip_blocks.append(ConvBlock(hidden_sizes[(idx + 1) - rate], dim, downsample=True, use_norm=use_norm))

            in_features = dim

        for dim in fc_hidden_sizes:

            fc_blocks.append(FCBlock(in_features, dim))
            in_features = dim

        self.rate = rate
        self.use_skip = use_skip

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.skip_blocks = nn.ModuleList(skip_blocks) if use_skip else None
        self.fc_blocks = nn.ModuleList(fc_blocks)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):

        y = x

        for idx, block in enumerate(self.conv_blocks):

            is_nxt = ((idx + 1) % self.rate == 0)

            x = block(y)

            if self.use_skip and is_nxt:

                y = x + self.skip_blocks[(idx + 1) // self.rate - 1](y)

            else:

                y = x

        y = self.avg_pool(y)
        y = torch.flatten(y, start_dim=1, end_dim=-1)

        for block in self.fc_blocks:

            y = block(y)

        return y


# -------------------------------------------------------------------------------------------------------------------

class ClassificationHead(nn.Module):

    def __init__(self, in_features, num_classes):

        super(ClassificationHead, self).__init__()

        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):

        return self.classifier(x)


# -------------------------------------------------------------------------------------------------------------------

class DeepFeatureOverload(nn.Module):

    def __init__(self, k_dict):

        super(DeepFeatureOverload, self).__init__()

        self.k_dict = k_dict
        self.selector = DeepFeatureKNN(distance_fn=None)

    def overload(self, x, labels, embeddings, logits=None):

        batch_size = x.size(0)
        groups = {}

        for i in range(batch_size):

            yi = int(labels[i].detach().cpu().numpy().squeeze())

            if yi not in groups:

                groups[yi] = common.TorchDict({})

                groups[yi]['x'] = []
                groups[yi]['embeddings'] = []

                if logits is not None:

                    groups[yi]['logits'] = []

            groups[yi]['x'].append(x[i])
            groups[yi]['embeddings'].append(embeddings[i])

            if logits is not None:

                groups[yi]['logits'].append(logits[i])

        for yi in groups:

            k = self.k_dict[yi] if yi in self.k_dict else 0

            groups[yi]['x'] = torch.stack(groups[yi]['x'], dim=0)
            groups[yi]['embeddings'] = torch.stack(groups[yi]['embeddings'], dim=0)

            if logits is not None:

                groups[yi]['logits'] = torch.stack(groups[yi]['logits'], dim=0)

            if 0 < k <= len(groups[yi]['embeddings']):

                out = self.selector(groups[yi]['embeddings'], groups[yi]['embeddings'], k=k, return_distances=True)

                groups[yi]['neighbors'], groups[yi]['distances'] = out

            else:

                size = groups[yi]['embeddings'].size(0)

                groups[yi]['neighbors'] = groups[yi]['embeddings'].unsqueeze(1)
                groups[yi]['distances'] = torch.zeros((size, 1), dtype=torch.float32)

        for yi in groups:

            groups[yi] = groups[yi].to(device=x.device)

        return common.TorchDict(groups).to(device=x.device)

    def forward(self, x, labels, embeddings, logits=None):

        groups = self.overload(x, labels, embeddings, logits)

        return groups

# -------------------------------------------------------------------------------------------------------------------
