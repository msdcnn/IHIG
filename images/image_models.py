"""
    Class file that enlists models for extracting features from image
"""
import torch
from torch import nn
from pytorch_pretrained_vit import ViT
import torch.nn.functional as F
from arconv import ARConv
class ImageEncoder(nn.Module):
    def __init__(self, input_dim=768, inter_dim=500, output_dim=300):
        """
            Initializes the model to process bounding box features extracted by MaskRCNNExtractor
            Returns:
                None
        """
        super(ImageEncoder, self).__init__()

        self.input_dim = input_dim
        self.inter_dim = inter_dim
        self.output_dim = output_dim

        self.arconv = ARConv(inc=self.input_dim, outc=self.output_dim, padding=1, stride=1)
        self.relu = nn.GELU()
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, x):
        """
            Function to compute forward pass of the ImageEncoder
        Args:
            x (Tensor): Bounding box features extracted from Mask R-CNN. Tensor of shape (N,K,2048,7,7)
            where N is the batch size and K is the bboxes number, features map of shape 2048 X 7 x 7 for each object
            edge_index(tensor, dtype=torch.long): Graph connectivity in COO format. Tensor of shape (2,num_edges).
            Because in general, the image graph is fully connected.
            edge_attr(tensor, dtype=torch.float) : Edge feture with shape (N, num_edges, num_features)
        Returns:
            x (Tensor): Processed bounding box features. Tensor of shape (N,K,output_dim), K=49
            pv: (N,K) Tensor, the importance of each visual object of image
            """
        if x.dim() == 5:
            N, K, C, H, W = x.size()
            x = x.view(N * K, C, H, W)
        elif x.dim() == 3:
            N, K, C = x.size()
            x = x.view(N * K, C, 1, 1)
        else:
            raise ValueError("Unsupported input shape for ImageEncoder: {}".format(x.shape))

        x = self.arconv(x)  
        x = x.mean(dim=[2, 3]) 
        x = self.relu(x)
        x = self.norm(x)
        x = x.view(N, K, self.output_dim)
        return x
