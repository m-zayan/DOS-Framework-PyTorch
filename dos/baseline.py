from torch import nn

from .modules import DeepFeatureOverload, BasicEncoder, ClassificationHead
from .losses import micro_cluster_loss


# -------------------------------------------------------------------------------------------------------------------

class Model(nn.Module):

    def __init__(self, k_dict, num_classes=10, num_channels=3, hidden_sizes=None, fc_hidden_sizes=None,
                 use_skip=True, rate=2, use_norm=False):

        super(Model, self).__init__()

        if hidden_sizes is None:

            hidden_sizes = [32, 64, 64, 128]

        if fc_hidden_sizes is None:

            fc_hidden_sizes = [256, 256]

        self.encoder = BasicEncoder(num_channels=num_channels, hidden_sizes=hidden_sizes,
                                    fc_hidden_sizes=fc_hidden_sizes, use_skip=use_skip,
                                    rate=rate, use_norm=use_norm)

        self.overload = DeepFeatureOverload(k_dict=k_dict)

        self.classifier = ClassificationHead(in_features=fc_hidden_sizes[-1], num_classes=num_classes)

    def forward(self, x, labels=None):

        embeddings = self.encoder(x)
        logits = self.classifier(embeddings)

        if labels is not None:

            groups = self.overload(x, labels, embeddings, logits)

            loss, aux_loss = micro_cluster_loss(groups)

            return logits, (loss, aux_loss)

        else:

            return logits

# -------------------------------------------------------------------------------------------------------------------
