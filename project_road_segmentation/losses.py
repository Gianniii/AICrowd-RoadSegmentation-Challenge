"""This module contains a set of loss functions that we implemented
 ourselves. Alas, none of what is here was used for our best submission. But
 they were experimented with, so we thought it was worth to keep it to show what we did.
 """
import torch
import os
import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from torchvision.utils import save_image
# from topologylayer.nn import (
#   LevelSetLayer2D,
#   SumBarcodeLengths,
#   PartialSumBarcodeLengths,
#   TopKBarcodeLengths,
# )

class Jaccard(nn.Module):
    def __init__(self, threshold=0.5):
        """A pytorch module to compute the Jaccard distance between a prediction
        and its target. The ratio between the size of the area where they agree
        and the size of the total area they cover

        Args:
            threshold (float): The point after which a pixel is considered as
            classified as 1
        """
        super().__init__()
        self.threshold = threshold

    def forward(self, inputs, target):
        """The function that is called when you use this module

        Args:
            input (torch.Tensor, dtype=float): A prediction
            target (torch.Tensor, dtype=float): The ground truth

        Returns:
            float: The Jaccard distance between the input and target

        >>> torch.seed = 123
        >>> input = torch.tensor([[0.2, 0.8], [0.6, 0.7]])
        >>> target = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        >>> loss = Jaccard()
        >>> loss.forward(input, target)
        tensor(0.6667)
        """
        inputs = torch.sigmoid(inputs)
        # input = (input > self.threshold) # Tensor of bools

        # The target should already be 0.0 or 1.0 but we turn it into bools
        # anyway
        target = (target > self.threshold).float()

        inputs = inputs.view(-1)
        target = target.view(-1)

        intersection = (inputs * target).sum()
        union_with_overap = (inputs + target).sum()

        union = union_with_overap - intersection

        return 1 - intersection / union


class TopLoss(nn.Module):
    def __init__(self, size):
        """A torch module to use persistence diagrams in image segmentation.
        We use a random sampling of the input images because otherwise 
        training takes too much time.

        Args:
            size (int): The size of the chunks used for forward call.
        """
        super(TopLoss, self).__init__()
        self.size = size[0]
        self.pdfn = LevelSetLayer2D(size=size, sublevel=False)
        self.con_comps = TopKBarcodeLengths(dim=0, k=2)  # Compute connected components
        self.loops = TopKBarcodeLengths(dim=1, k=2)  # Compute "loops"

    def topo_loss2(self, inputs, targets, dim, k=3):
        """Computes the l2-bottleneck distance between two persistence diagrams.
        Only considering the top k points whose birth and death times are
        the most far apart.

        Args:
            inputs (torch 2d tensor): A tensor representing the prediction.
            targets (torch 2d tensor): A tensor representing the target.
            dim (0 or 1): Whether to compute the loss on the zero or one dimensional persistence diagram.
            k (int, optional): The `k` in top k. Defaults to 3.

        Returns:
            float: The loss
        """
        input_dgms = self.pdfn(inputs)
        target_dgms = self.pdfn(targets)

        # Get the k most salient points in the diagram from input and target images
        _, out_idx0 = input_dgms[0][0].diff(dim=1).abs().mul(np.power(2, 0.5)).squeeze().topk(k)
        _, tar_idx0 = target_dgms[0][0].diff(dim=1).abs().mul(np.power(2, 0.5)).squeeze().topk(k)

        # Make bipartite graph to find min weight matching between the points of input diagram and target diagram
        G = nx.Graph()
        left_nodes = []
        right_nodes = []
        weights = {}
        for i in out_idx0:
            for j in tar_idx0:
                diff_birth = (input_dgms[0][0][i][0] - target_dgms[0][0][j][0]).nan_to_num(
                    nan=0.0, posinf = 1e10, neginf = -1e10
                    )
                diff_death = (input_dgms[0][0][i][1] - target_dgms[0][0][j][1]).nan_to_num(
                    nan=0.0, posinf = 1e10, neginf = -1e10
                    )
                # print(diff_birth, diff_death)
                l = "l" + str(i.item())
                r = "r" + str(j.item())
                weights[l,r] = (diff_birth.pow(2) + diff_death.pow(2)).pow(0.5)
                # print(G, "before adding edges")
                left_nodes.append(l)
                right_nodes.append(r)
                # print(l, r)
                G.add_edge(l, r, weight = weights[l,r].item())
                # print(G, "after adding edges")
        # print(weights)
        matching = bipartite.matching.minimum_weight_full_matching(G, left_nodes, "weight")       

        # Now that we know to which target point each input point is associated we can calculate the cost
        # ...
        # For the points that we can match, the cost is how far they are from the target
        costs = []
        for key in matching.keys(): 
            if key.startswith('l'):
                val = matching[key]
                # print(weights[key, val])
                costs.append(weights[key, val])
        costs = torch.stack(costs)

        # For the other points, we approximate and say that they should be mapped to zero, therefore their cost is just their saliency
        not_k = input_dgms[0][0].shape[0] - k
        out_sal0, _ = input_dgms[0][0].diff(dim=1).abs().mul(np.power(2, 0.5)).squeeze().topk(not_k, largest = False, sorted = False)

        costs = torch.cat([costs, out_sal0])

        # Finally return the l2 norm of the costs
        result = costs.pow(2).sum().pow(0.5)
        return result

    def topo_loss3(self, inputs, targets):
        """Computes the prior and the loss with respect to that prior

        Args:
            inputs (torch 2d tensor): A tensor representing the prediction
            targets (torch 2d tensor): A tensor representing the target

        Returns:
            float: The loss
        """
        target_dgm = self.pdfn(targets)
        prior_dim0 = (target_dgm[0][0].diff(dim=1).squeeze().abs() != 0).sum()
        prior_dim1 = (target_dgm[0][1].diff(dim=1).squeeze().abs() != 0).sum()

        input_dgm  = self.pdfn(inputs)
        error_dim0 = PartialSumBarcodeLengths(dim=0, skip=prior_dim0)(input_dgm)

        error_dim1 = PartialSumBarcodeLengths(dim=1, skip=prior_dim1)(input_dgm)

        # print(prior_dim0, prior_dim1)
        # print(error_dim0)
        # print(error_dim0(input_dgm), error_dim1(input_dgm))

        loss = error_dim0 + error_dim1
        # print(loss)
        return loss


    def forward(self, batch_input, batch_target):
        """Performs forward pass: randomly chunk each image,
        compute the loss, return the average.

        Args:
            batch_input (torch 3d tensor): A batch of tensors representing the prediction.
            batch_target (torch 3d tensor): A batch of tensors representing the targets.

        Returns:
            torch 1d tensor: The loss for each prediction in the batch.
        """
        # batch_input = torch.sigmoid(batch_input)
        tuple_chunks = [random_chunks_and_pad(batch_input[i], batch_target[i], self.size, 2) for i in range(batch_input.shape[0])]
        batch_input_chunks = [c[0] for c in tuple_chunks]
        batch_target_chunks = [c[1] for c in tuple_chunks]
        costs = []
        for (input_chunks, target_chunks) in zip(batch_input_chunks, batch_target_chunks):
            sub_costs = torch.stack([
                    self.topo_loss3(input_chunks[i], target_chunks[i]) for i in range(input_chunks.shape[0])
                ])
            # print(sub_costs.mean())
            costs.append(sub_costs.mean())

        costs = torch.stack(costs)
        return torch.mean(costs.view(-1))

def random_chunks_and_pad(image, target, chunk_size, num_chunks, pad=1):
    """Turns a 2d square image tensor, and its target into a 3d tensor of randomly sampled 
    2d images with padded 1.0 contour.

    Args:
        image (2d torch tensor): The image that gets chunked and padded
        target (2d torch tensor): The target of the input
        chunk_size (int): How large should the returned chink be
        num_chunks (int): How many chunks to sample
        pad (int): The size of the padding

    Returns:
        chunks_image (3d torch tensor): The image chunks like in the discription
        chunks_target (3d torch tensor): The target chunks like in the discription
    """
    # TODO update docstring
    image = image.squeeze()
    target = target.squeeze()
    img_size = image.shape[0]

    chunks_image = []
    chunks_target = []
    while len(chunks_image) < num_chunks:
        i = math.floor(random.random() * (img_size - chunk_size))
        j = math.floor(random.random() * (img_size - chunk_size))
        chunk_img = image[i: i + chunk_size - 2 * pad, j: j + chunk_size - 2 * pad]
        chunk_img = F.pad(chunk_img, (pad, pad, pad, pad), "constant", 1.0)

        chunk_tar = target[i: i + chunk_size - 2 * pad, j: j + chunk_size - 2 * pad]
        chunk_tar = F.pad(chunk_tar, (pad, pad, pad, pad), "constant", 1.0)
        chunks_image.append(chunk_img)
        chunks_target.append(chunk_tar)
        
    chunks_image = torch.stack(chunks_image)
    chunks_target = torch.stack(chunks_target)
    return chunks_image, chunks_target
    

if __name__ == "__main__":
    import torch
    import doctest

    doctest.testmod()
