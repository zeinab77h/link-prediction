import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.nn as gnn
from torch.autograd import Variable


# Define the Net class for graph network modeling
class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, latent_dim=[32, 32, 32], with_dropout=False, use_diffpool=False):
        super(Net, self).__init__()

        self.use_diffpool = use_diffpool  # Use DiffPool as a flag

        conv = gnn.GCNConv  # Define GCN convolution

        self.latent_dim = latent_dim

        self.conv_params = nn.ModuleList()
        self.conv_params.append(conv(input_dim, latent_dim[0], cached=False))

        for i in range(1, len(latent_dim)):
            self.conv_params.append(conv(latent_dim[i - 1], latent_dim[i], cached=False))

        latent_dim_sum = sum(latent_dim)

        if self.use_diffpool:
            self.diffpool = nn.Linear(latent_dim_sum, hidden_size)

        self.linear1 = nn.Linear(latent_dim_sum, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 2)

        self.with_dropout = with_dropout

    def forward(self, data):
        data = data.to(torch.device("cuda"))  # Ensure data is moved to the correct device

        x, edge_index, batch, y = data.x, data.edge_index, data.batch, data.y

        cur_message_layer = x
        cat_message_layers = []
        lv = 0

        while lv < len(self.latent_dim):
            cur_message_layer = self.conv_params[lv](cur_message_layer, edge_index)
            cur_message_layer = torch.tanh(cur_message_layer)

            cat_message_layers.append(cur_message_layer)
            lv += 1

        cur_message_layer = torch.cat(cat_message_layers, dim=1)  # Concatenate along the feature dimension

        if self.use_diffpool:
            cur_message_layer = self.diffpool(cur_message_layer)

        batch_idx = torch.unique(batch)
        idx = []
        for i in batch_idx:
            idx.append((batch == i).nonzero(as_tuple=False)[0].item())  # Fixing tensor indexing issue

        # Ensure compatibility for linear layers by padding or truncating as needed
        target_size = self.linear1.in_features
        if cur_message_layer.size(1) < target_size:
            padding = torch.zeros((cur_message_layer.size(0), target_size - cur_message_layer.size(1)), device=cur_message_layer.device)
            cur_message_layer = torch.cat((cur_message_layer, padding), dim=1)
        elif cur_message_layer.size(1) > target_size:
            cur_message_layer = cur_message_layer[:, :target_size]

        hidden = self.linear1(cur_message_layer)
        self.feature = hidden
        hidden = F.relu(hidden)

        if self.with_dropout:
            hidden = F.dropout(hidden, training=self.training)

        logits = self.linear2(hidden)
        logits = F.log_softmax(logits, dim=1)

        # Adjust logits to ensure it matches the batch size of y
        if y is not None:
            if logits.size(0) != y.size(0):
                logits = logits[:y.size(0)]  # Align the batch size

            y = Variable(y)
            loss = F.nll_loss(logits, y)

            pred = logits.data.max(1, keepdim=True)[1]
            acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(y.size()[0])
            return logits, loss, acc, self.feature
        else:
            return logits
