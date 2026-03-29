import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import math
from utils import L2_norm, cosine_distance
from transformers import BertModel
from utils.data_utils import pad_tensor
from transformers import RobertaModel
from transformers import BertConfig, BertForPreTraining, RobertaForMaskedLM, RobertaModel, RobertaConfig, AlbertModel, AlbertConfig


class TextEncoder(nn.Module):
    r"""Initializes a NLP embedding block.
     :param input_size:
     :param nhead:
     :param dim_feedforward:
     :param dropout:
     :param activation:
     """

    def __init__(self, input_size=768, out_size = 300):
        super(TextEncoder, self).__init__()

        self.input_size = input_size
        self.out_size = out_size


        self.norm = nn.LayerNorm(self.out_size)
        self.linear = nn.Linear(self.out_size, 1)
        self.linear_ = nn.Linear(self.input_size, self.out_size)
        self.config = BertConfig.from_pretrained('./bert-base-uncased/')
        self.bert_model = BertModel.from_pretrained('./bert-base-uncased/')
            
    def get_config(self):
        return self.config

    def forward(self, t1, word_seq, key_padding_mask):
        """
        Function to compute forward pass of the TextEncoder
        Args:
            t1: (N,L,D) Padded Tensor. L is the length. D is dimension after BERT
            word_seq: (N, list of tuples) span of each word in the sentence, or None
            key_padding_mask: (N,L) Tensor. Mask for padding tokens.
        Returns:
            t1: (N,L1,D). The embedding of each word or np. D is dimension after BERT.
        """
        # (batch_size, sequence_length, hidden_size)
        t1 = self.bert_model(**t1)[0]

        if word_seq is None:
            t1 = t1[:, 1:-1, :]  
            t1 = t1.mean(dim=1, keepdim=True)  # (B, 1, H) 
            return self.norm(self.linear_(t1))  
        else:

            t1 = t1[:, 1:-1, :]  
            captions = []
            for i in range(t1.size(0)):
                captions.append(torch.stack([
                    torch.mean(t1[i][tup[0]:tup[1], :], dim=0) for tup in word_seq[i]
                ]))
            t1 = pad_sequence(captions, batch_first=True).cuda()
            t1 = self.norm(self.linear_(t1))
            return t1