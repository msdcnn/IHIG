import json
import torch
import math
import random
import numpy as np

from transformers import AutoTokenizer


def read_json(path):
    """

    :param path: resolute path for json files using json.dump function to construct
    :return: dict
    """
    with open(path, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data


def write_json(path, data):
    with open(path, "w", encoding='utf-8') as f:
        json.dump(data, f)


def reads_json(path):
    with open(path, "r") as f:
        data = [json.loads(line) for line in f]
        return data


def pad_tensor(vec, pad, dim):
    """
        Pads a tensor with zeros according to arguments given

        Args:
            vec (Tensor): Tensor to pad
            pad (int): The total tensor size with pad
            dim (int): Dimension to pad

        Returns:
            padded_tensor (Tensor): A new tensor padded to 'pad' in dimension 'dim'

    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    # torch.zeros torch.float32
    padded_tensor = torch.cat([vec, torch.zeros(*pad_size)], dim=dim)
    return padded_tensor



class PadCollate_without_know:

    def __init__(self, img_dim=0, edge_dim=1, twitter_dim=2, desc_dim=3, dep_dim=4, label_dim=5, chunk_dim=6, use_np=False):
        self.img_dim = img_dim
        self.edge_dim = edge_dim
        self.twitter = twitter_dim
        self.desc = desc_dim
        self.dep = dep_dim
        self.label_dim = label_dim
        self.chunk = chunk_dim
        self.use_np = use_np
        self.tokenizer = AutoTokenizer.from_pretrained('/bert-base-uncased/')


    def pad_collate(self, batch):
        xs = torch.stack([t[self.img_dim] for t in batch])
        img_edge = [t[self.edge_dim] for t in batch]


        twitter = [t[self.twitter] for t in batch]
        desc = [t[self.desc] for t in batch]
        token_lens = [len(t) for t in twitter]

        encoded_twitter = self.tokenizer(twitter, is_split_into_words=True, return_tensors="pt",
                                         truncation=True, max_length=100, padding=True)
        encoded_desc = self.tokenizer(desc, is_split_into_words=False, return_tensors="pt",
                                      truncation=True, max_length=100, padding=True)  

        word_spans = []
        word_len = []
        for idx, tok_len in enumerate(token_lens):
            span = []
            for i in range(tok_len):
                s = encoded_twitter.word_to_tokens(idx, i)
                if s: span.append([s[0] - 1, s[1] - 1])
            word_spans.append(span)
            word_len.append(len(span))

        max_len1 = max(word_len)
        mask_batch1 = construct_mask_text(word_len, max_len1)

        deps1 = [t[self.dep] for t in batch]
        deps1_ = [[d for d in dep if d[0] < max_len1 and d[1] < max_len1] for dep in deps1]
        if self.use_np:
            org_chunk = [torch.tensor(t[self.chunk], dtype=torch.long) for t in batch]
        else:
            org_chunk = [torch.arange(i, dtype=torch.long) for i in word_len]

        labels = torch.tensor([t[self.label_dim] for t in batch], dtype=torch.long)
        edge_cap1, gnn_mask_1, np_mask_1 = construct_edge_text(deps=deps1_, max_length=max_len1, use_np=self.use_np,
                                                               chunk=org_chunk)


        temp_labels = [labels - 0, labels - 1]
        target_labels = []
        for i in range(2):
            cur = [j for j in range(temp_labels[0].size(0)) if temp_labels[i][j] == 0]
            target_labels.append(torch.LongTensor(cur))


        img_patch_lens = [len(img) for img in xs]
        max_len_img = max(img_patch_lens)
        mask_batch_img = construct_mask_text(img_patch_lens, max_len_img)

        return (xs, img_edge,
                encoded_twitter, encoded_desc,
                word_spans, word_len, mask_batch1, encoded_desc["attention_mask"],
                edge_cap1, gnn_mask_1, np_mask_1, labels, mask_batch_img, target_labels)

    def __call__(self, batch):
        return self.pad_collate(batch)


def construct_mask_text(seq_len, max_length):
    """

    Args:
        seq_len1(N): list of number of words in a caption without padding in a minibatch
        max_length: the dimension one of shape of embedding of captions of a batch

    Returns:
        mask(N,max_length): Boolean Tensor
    """
    # the realistic max length of sequence
    max_len = max(seq_len)
    if max_len <= max_length:
        mask = torch.stack(
            [torch.cat([torch.zeros(len, dtype=bool), torch.ones(max_length - len, dtype=bool)]) for len in seq_len])
    else:
        mask = torch.stack(
            [torch.cat([torch.zeros(len, dtype=bool),
                        torch.ones(max_length - len, dtype=bool)]) if len <= max_length else torch.zeros(max_length,
                                                                                                         dtype=bool) for
             len in seq_len])

    return mask



def construct_edge_text(deps, max_length, chunk=None, use_np=False):
    """

    Args:
        deps: list of dependencies of all captions in a minibatch
        chunk: use to confirm where
        max_length : the max length of word(np) length in a minibatch
        use_np:

    Returns:
        deps(N,2,num_edges): list of dependencies of all captions in a minibatch. with out self loop.
        gnn_mask(N): Tensor. If True, mask.
        np_mask(N,max_length+1): Tensor. If True, mask
    """
    dep_se = []
    gnn_mask = []
    np_mask = []
    if use_np:
        for i, dep in enumerate(deps):
            if len(dep) > 1 and len(chunk[i]) > 1:
                # dependency between word(np) and word(np)
                dep_np = [torch.tensor(dep, dtype=torch.long), torch.tensor(dep, dtype=torch.long)[:, [1, 0]]]
                dep_np = torch.cat(dep_np, dim=0).T.contiguous()
                gnn_mask.append(False)
                np_mask.append(True)
            else:
                dep_np = torch.tensor([])
                gnn_mask.append(True)
                np_mask.append(False)
            dep_se.append(dep_np)
    else:
        for i, dep in enumerate(deps):
            if len(dep) > 3 and len(chunk[i]) > 1:
                dep_np = [torch.tensor(dep, dtype=torch.long), torch.tensor(dep, dtype=torch.long)[:, [1, 0]]]
                gnn_mask.append(False)
                np_mask.append(True)
                dep_np = torch.cat(dep_np, dim=0).T.contiguous()
            else:
                dep_np = torch.tensor([])
                gnn_mask.append(True)
                np_mask.append(False)
            dep_se.append(dep_np.long())

    np_mask = torch.tensor(np_mask).unsqueeze(1)
    np_mask_ = [torch.tensor(
        [True] * max_length) if gnn_mask[i] else torch.tensor([True] * max_length).index_fill_(0, chunk_,
                                                                                               False).clone().detach()
                for i, chunk_ in enumerate(chunk)]
    np_mask_ = torch.stack(np_mask_)
    np_mask = torch.cat([np_mask_, np_mask], dim=1)
    gnn_mask = torch.tensor(gnn_mask)
    return dep_se, gnn_mask, np_mask


def construct_edge_know(deps):
    """

    Args:
        deps: list of dependencies of all captions in a minibatch
        chunk: use to confirm where
        max_length : the max length of word(np) length in a minibatch
        use_np:

    Returns:
        deps(N,2,num_edges): list of dependencies of all captions in a minibatch. with out self loop.
        gnn_mask(N): Tensor. If True, mask.
        np_mask(N,max_length+1): Tensor. If True, mask
    """
    dep_se = []
    gnn_mask = []
    for i, dep in enumerate(deps):
        if len(dep) > 1:
            dep_np = [torch.tensor(dep, dtype=torch.long), torch.tensor(dep, dtype=torch.long)[:, [1, 0]]]
            gnn_mask.append(False)
            dep_np = torch.cat(dep_np, dim=0).T.contiguous()
        else:
            dep_np = torch.tensor([])
            gnn_mask.append(True)
        dep_se.append(dep_np.long())
    gnn_mask = torch.tensor(gnn_mask)
    return dep_se, gnn_mask


def construct_edge_image(num_patches):
    """
    Args:
        num_patches: the patches of image (49)
    There are two kinds of construct method
    Returns:
        edge_image(2,num_edges): List. num_edges = num_boxes*num_boxes
    """

    edge_image = []
    p = math.sqrt(num_patches)
    for i in range(num_patches):
        for j in range(num_patches):
            if j == i:
                continue
            if math.fabs(i % p - j % p) <= 1 and math.fabs(i // p - j // p) <= 1:
                edge_image.append([i, j])
    edge_image = torch.tensor(edge_image, dtype=torch.long).T
    return edge_image


def construct_edge_attr(bboxes, imgs):
    """

    Args:
        bboxes(N,num_bboxes,4): the last dimension is (x1,y1) and (x2,y2) referring to start vertex and end vertex

    Returns:
        bboxes_festures(N, num_edges, num_edge_features) : List of Tensor. Size of Tensor is
        (num_edges, num_edge_features)

    """
    bboxes_festures = []
    for i, sample in enumerate(bboxes):
        s_img = imgs[i].size(0) * imgs[i].size(1)
        sample_feature = []
        for o in sample:
            bbox_feature = []
            wo = o[2] - o[0]
            ho = o[3] - o[1]
            so = wo * ho
            for s in sample:
                ws = s[2] - s[0]
                hs = s[3] - s[1]
                bbox_feature.append(
                    [(o[0] - s[0]) / ws, (o[1] - s[1]) / hs, math.log(wo / ws), math.log(ho / hs), so / s_img])
            sample_feature.append(torch.tensor(bbox_feature))
        # (N, bbox_number,bbox_number,5)
        bboxes_festures.append(torch.cat(sample_feature))
    return torch.tensor(bboxes_festures)


def seed_everything(seed: int):
    r"""Sets the seed for generating random numbers in PyTorch, numpy and
    Python.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)