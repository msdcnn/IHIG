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

    def __init__(self, input_size=768, out_size=300, knowledge_type=1, know_max_length=20):
        super(TextEncoder, self).__init__()

        self.input_size = input_size
        self.out_size = out_size

        self.norm = nn.LayerNorm(self.out_size)
        self.linear = nn.Linear(self.out_size, 1)
        self.linear_ = nn.Linear(self.input_size, self.out_size)
        self.knowledge_type = knowledge_type
        self.know_max_length = know_max_length
        self.bert_model = BertModel.from_pretrained('./bert-base-uncased/')
        self.roberta = RobertaModel.from_pretrained('./roberta-base', return_dict=True)
        for param in self.bert_model.parameters():
            param.requires_grad = False

        self.linear2 = nn.Linear(self.input_size, self.out_size)

        self.relu1 = nn.ReLU()

    def forward(self, t1, word_seq, key_padding_mask, encoded_know, know_word_spans, key_padding_mask_know, lam=1):
        """
        Function to compute forward pass of the ImageEncoder TextEncoder
        Args:
            t1: (N,L,D) Padded Tensor. L is the length. D is dimension after bert
            token_length: (N) list of length
            word_seq:(N,tensor) list of tensor
            key_padding_mask: (N,L1) Tensor. L1 is the np length. True means mask

        Returns:
            t1: (N,L1,D). The embedding of each word or np. D is dimension after bert.
            score: (N,L1,D). The importance of each word or np. For convenience, expand the tensor (N,L1，D) to compute
            the caption embedding.


        """

        encoded_know = self.bert_model(**encoded_know)[0]
        outputs = self.roberta(input_ids=input_ids, attention_mask=input_mask)
        encoded_know = encoded_know[:, 1:-1, :]

        if know_word_spans is not None:
            know_list = []
            # [X,L,H] X is the number of np and word
            for i in range(encoded_know.size(0)):
                know_list.append(
                    torch.stack([torch.mean(encoded_know[i][tup[0]:tup[1], :], dim=0) for tup in know_word_spans[i]]))
            # assert encoded_know.size(1)==20 or 5
            if self.knowledge_type != 1:
                # recover the word embedding to knowledge embedding
                know_list = [torch.cat(know_list[5*i: 5*i+5], dim=0) for i in range(int(len(know_list)/5))]

        if self.knowledge_type == 1:
            encoded_know = torch.stack(
                [pad_tensor(vec=know.cpu(), pad=self.know_max_length, dim=0) for know in know_list], dim=0)

        else:
            encoded_know = torch.stack(know_list, dim=0)
        encoded_know = self.norm(self.linear_(encoded_know.cuda()))
        if self.knowledge_type == 1:
            # score_know = self.linear(encoded_know).squeeze()
            score_know = self.linear(encoded_know).squeeze().masked_fill_(key_padding_mask_know, float("-Inf"))
        else:
            score_know = self.linear(encoded_know).squeeze().masked_fill_(key_padding_mask_know, float("-Inf"))

        score_know = nn.Softmax(dim=1)(score_know * lam)

        # (batch_size, sequence_length, hidden_size)
        t1 = self.bert_model(**t1)[0]
        # remove cls token and sep token
        t1 = t1[:, 1:-1, :]
        captions = []
        for i in range(t1.size(0)):
            # [X,L,H] X is the number of np and word
            captions.append(torch.stack([torch.mean(t1[i][tup[0]:tup[1], :], dim=0) for tup in word_seq[i]]))

        # (N,L,D)
        t1 = pad_sequence(captions, batch_first=True).cuda()
        t1 = self.norm(self.linear_(t1))
        # get each word importance
        # (N,L)
        score = self.linear(t1).squeeze().masked_fill_(key_padding_mask, float("-Inf"))
        score = nn.Softmax(dim=1)(score * lam).unsqueeze(2).repeat((1, 1, self.out_size))

        return t1, score, encoded_know, score_know


class TextEncoder_without_know(nn.Module):
    r"""Initializes a NLP embedding block.
     :param input_size:
     :param nhead:
     :param dim_feedforward:
     :param dropout:
     :param activation:
     """

    def __init__(self, input_size=768, out_size = 300):
        super(TextEncoder_without_know, self).__init__()

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