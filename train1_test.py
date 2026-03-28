import argparse
import time

from tqdm import tqdm
import os
from utils.logging.tf_logger import Logger
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
import torch.nn.functional as F
from model import KEHModel_without_know
from utils.data_utils import construct_edge_image
from utils.dataset0 import BaseSet  
from utils.compute_scores import get_metrics, get_four_metrics
from utils.data_utils import PadCollate, PadCollate_without_know
import json
import re
from utils.data_utils import seed_everything


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-m', '--mode', type=str, default='train',
                    help="mode, {'" + "train" + "', '" + "eval" + "'}")
parser.add_argument('-p', '--path', type=str, default='saved_model324_1 path',
                    help="path, relative path to save model}")  
parser.add_argument('-s', '--save', type=str, default='saved model',
                    help="path, path to saved model}")
parser.add_argument('-o', '--para', type=str, default='parameter0.json',
                    help="path, path to json file keeping parameter}")  
args = parser.parse_args()
with open(args.para) as f:
    parameter = json.load(f)
annotation_files = parameter["annotation_files"]
img_files = parameter["DATA_DIR"]
use_np = parameter["use_np"]

model = KEHModel_without_know(txt_input_dim=parameter["txt_input_dim"], txt_out_size=parameter["txt_out_size"],
                              img_input_dim=parameter["img_input_dim"],
                              img_inter_dim=parameter["img_inter_dim"],
                              img_out_dim=parameter["img_out_dim"], cro_layers=parameter["cro_layers"],
                              cro_heads=parameter["cro_heads"], cro_drop=parameter["cro_drop"],
                              txt_gat_layer=parameter["txt_gat_layer"], txt_gat_drop=parameter["txt_gat_drop"],
                              txt_gat_head=parameter["txt_gat_head"],
                              txt_self_loops=parameter["txt_self_loops"], img_gat_layer=parameter["img_gat_layer"],
                              img_gat_drop=parameter["img_gat_drop"],
                              img_gat_head=parameter["img_gat_head"], img_self_loops=parameter["img_self_loops"],
                              img_edge_dim=parameter["img_edge_dim"],
                              img_patch=parameter["img_patch"], lam=parameter["lambda"],
                              type_bmco=parameter["type_bmco"], visualization=parameter["visualization"])
print("Image Encoder", sum(p.numel() for p in model.img_encoder.parameters() if p.requires_grad))
print("Text Encoder", sum(p.numel() for p in model.txt_encoder.parameters() if p.requires_grad))
print("Total Params", sum(p.numel() for p in model.parameters() if p.requires_grad))

model.to(device=device)
optimizer = optim.Adam(params=model.parameters(), lr=parameter["lr"], betas=(0.9, 0.999), eps=1e-8,
                       weight_decay=parameter["weight_decay"],
                       amsgrad=True)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=parameter["patience"], verbose=True)

cross_entropy_loss = CrossEntropyLoss()

logger = Logger(model_name=parameter["model_name"], data_name='twitter',
                log_path=os.path.join(parameter["TARGET_DIR"], args.path,
                                      'tf_logs', parameter["model_name"]))
img_edge_index = construct_edge_image(parameter["img_patch"])


def construct_edge_image_region(image):
    """
    Args:
        num_patches: the patches of image (49)
    There are two kinds of construct method
    Returns:
        edge_image(2,num_edges): List. num_edges = num_boxes*num_boxes
    """
    image_region = image.size(1)
    image_batch = image.size(0)
    all_edge = []
    for k in range(image_batch):
        edge_image = []
        for i in range(image_region):
            for j in range(image_region):
                if j == i:
                    continue
                if F.cosine_similarity(image[k, i, :], image[k, j, :], dim=-1) > 0.6:
                    edge_image.append([i, j])
        edge_image = torch.tensor(edge_image, dtype=torch.long).T
        all_edge.append(edge_image)
    return all_edge


def innovative_dynamic_fusion_batch(y_cl, database_feature, database_label, k=20, base_sigma=0.5):
    """
    Equivalent to sample-wise KNN, but implemented with batch vectorization:
    - Distance: L2 (torch.cdist)
    - Top-k selection: torch.topk(largest=False) replaces sort + slicing
    - Sigma: Adaptively determined based on the mean/variance of the full-database distances for each sample
    - Gaussian kernel-weighted voting
"""

    dist = torch.cdist(y_cl, database_feature)

    # top-k (B, k)
    topk_dist, topk_idx = torch.topk(dist, k=k, largest=False, sorted=False)
    topk_labels = database_label[topk_idx].float()

    mean_dist = dist.mean(dim=1)
    std_dist = dist.std(dim=1)
    density = 1.0 / (std_dist + 1e-8)
    sigma = base_sigma * (1 + torch.sigmoid((mean_dist - 0.5) * density))
    sigma = sigma.clamp(0.5, 1.5)

    weights = torch.exp(-(topk_dist ** 2) / (2 * (sigma.unsqueeze(1) ** 2)))
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-12)

    knn_pred = (weights * topk_labels).sum(dim=1)
    final_label = (knn_pred >= 0.5).long().tolist()
    return final_label



def train_model(epoch, train_loader):
    """
        Performs one training epoch and updates the weight of the current model
        Args:
            train_loader:
            optimizer:
            epoch(int): Current epoch number
        Returns:
            None
    """
    train_loss = 0.0
    total = 0.0
    model.train()
    predict = []
    temp = 0.07
    real_label = []

    for batch_idx, batch in enumerate(tqdm(train_loader)):
        (img_batch, img_edge, embed_t, embed_d, org_seq, org_word_len, mask_batch1, desc_mask,
         edge_cap1, gnn_mask_1, np_mask_1, labels, key_padding_mask_img, target_labels) = batch

        embed_t = {k: v.to(device) for k, v in embed_t.items()}
        embed_d = {k: v.to(device) for k, v in embed_d.items()}

        batch_size = len(img_batch)
        for i in range(len(target_labels)):
            target_labels[i] = target_labels[i].to(device)

        with torch.set_grad_enabled(True):
            y, y_cl = model(
                imgs=img_batch.to(device),
                orig_texts=embed_t,
                desc_texts=embed_d,
                mask_batch=mask_batch1.to(device),
                desc_mask=desc_mask.to(device),
                img_edge_index=img_edge,
                orig_word_seq=org_seq,
                desc_word_seq=None,
                txt_edge_index=edge_cap1,
                gnn_mask=gnn_mask_1.to(device),
                np_mask=np_mask_1.to(device),
                img_edge_attr=None,
                key_padding_mask_img=key_padding_mask_img
            )

            y_cl = F.normalize(y_cl, dim=-1)
            l_pos_neg_self = torch.einsum('nc,ck->nk', [y_cl, y_cl.T]) / temp
            l_pos_neg_self = torch.log_softmax(l_pos_neg_self, dim=-1).view(-1)
            cl_self_labels = target_labels[labels[0]]
            for idx in range(1, y_cl.size(0)):
                cl_self_labels = torch.cat((cl_self_labels, target_labels[labels[idx]] + idx * labels.size(0)), 0)
            cl_self_loss = -torch.gather(l_pos_neg_self, dim=0, index=cl_self_labels).sum() / cl_self_labels.size(0)

            loss = cross_entropy_loss(y, labels.to(device)) + cl_self_loss

            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()  # clear gradients for this training step

        predict += get_metrics(y.cpu())
        real_label += labels.cpu().tolist()
        total += batch_size
        torch.cuda.empty_cache()
        del img_batch, embed_t, embed_d

    # Calculate loss and accuracy for current epoch
    logger.log(mode="train", scalar_value=train_loss / len(train_loader), epoch=epoch, scalar_name='loss')
    acc, recall, precision, f1 = get_four_metrics(real_label, predict)
    logger.log(mode="train", scalar_value=acc, epoch=epoch, scalar_name='accuracy')

    print(' Train Epoch: {} Loss: {:.4f} Acc: {:.4f} Rec: {:.4f} Pre: {:.4f} F1: {:.4f}'.format(epoch, train_loss / len(
        train_loader), acc, recall, precision, f1))


def eval_validation_loss(val_loader):
    """
        Computes validation loss on the saved model, useful to resume training for an already saved model
    """
    val_loss = 0.
    predict = []
    real_label = []
    model.eval()
    with torch.no_grad():

        for batch_idx, (img_batch, img_edge, embed_t, embed_d, org_seq, org_word_len, mask_batch1, desc_mask,
                        edge_cap1, gnn_mask_1, np_mask_1, labels, key_padding_mask_img, target_labels) in enumerate(
            tqdm(val_loader)):
            embed_t = {k: v.to(device) for k, v in embed_t.items()}
            embed_d = {k: v.to(device) for k, v in embed_d.items()}
            img_edge_index = construct_edge_image_region(img_batch)
            y, _ = model(    imgs=img_batch.cuda(),
                            orig_texts=embed_t,
                            desc_texts=embed_d,
                            mask_batch=mask_batch1.cuda(),
                            desc_mask=desc_mask.cuda(),
                            img_edge_index=img_edge_index,
                            orig_word_seq=org_seq,
                            desc_word_seq=None,
                            txt_edge_index=edge_cap1,
                            gnn_mask=gnn_mask_1.cuda(),
                            np_mask=np_mask_1.cuda(),
                             img_edge_attr=None,
                             key_padding_mask_img=key_padding_mask_img)
            loss = cross_entropy_loss(y, labels.cuda())
            val_loss += float(loss.clone().detach().item())
            predict = predict + get_metrics(y.cpu())

            real_label = real_label + labels.cpu().numpy().tolist()
            # torch.cuda.empty_cache()
            del img_batch, embed_t, embed_d
        torch.cuda.empty_cache()  
        acc, recall, precision, f1 = get_four_metrics(real_label, predict)
        print(' Val Avg loss: {:.4f} Acc: {:.4f} Rec: {:.4f} Pre: {:.4f} F1: {:.4f}'.format(val_loss / len(val_loader),
                                                                                            acc, recall,
                                                                                            precision, f1))
    return val_loss


def evaluate_model(epoch, val_loader, train_loader):
    """
        Performs one validation epoch and computes loss and accuracy on the validation set
        Args:
            model:
            epoch (int): Current epoch number
        Returns:
            val_loss (float): Average loss on the validation set
    """
    val_loss = 0.
    predict = []
    real_label = []
    model.eval()
    with torch.no_grad():

        database_feature = []
        database_label = []
        for batch_idx, (img_batch, img_edge, embed_t, embed_d, org_seq, org_word_len, mask_batch1, desc_mask,
                        edge_cap1, gnn_mask_1, np_mask_1, labels, key_padding_mask_img, target_labels) in enumerate(tqdm(train_loader)):

            if batch_idx == 16:
                break
            embed_t = {k: v.to(device) for k, v in embed_t.items()}
            embed_d = {k: v.to(device) for k, v in embed_d.items()}
            for i in range(len(target_labels)):
                target_labels[i] = target_labels[i].cuda()
            y, y_cl = model(imgs=img_batch.cuda(),
                            orig_texts=embed_t,
                            desc_texts=embed_d,
                            mask_batch=mask_batch1.cuda(),
                            desc_mask=desc_mask.cuda(),
                            img_edge_index=img_edge,
                            orig_word_seq=org_seq,
                            desc_word_seq=None,
                            txt_edge_index=edge_cap1,
                            gnn_mask=gnn_mask_1.cuda(),
                            np_mask=np_mask_1.cuda(),
                            img_edge_attr=None,
                            key_padding_mask_img=key_padding_mask_img)
            database_feature.append(y_cl)
            database_label.append(labels)

        database_feature = torch.cat(database_feature, dim=0).cuda()
        database_label = torch.cat(database_label, dim=0).cuda()
        t = time.time()
        for batch_idx, (img_batch, img_edge, embed_t, embed_d, org_seq, org_word_len, mask_batch1, desc_mask,
                        edge_cap1, gnn_mask_1, np_mask_1, labels, key_padding_mask_img, target_labels) in enumerate(
            tqdm(val_loader)):
            embed_t = {k: v.to(device) for k, v in embed_t.items()}
            embed_d = {k: v.to(device) for k, v in embed_d.items()}
            y, y_cl = model(imgs=img_batch.cuda(),
                            orig_texts=embed_t,
                            desc_texts=embed_d,
                            mask_batch=mask_batch1.cuda(),
                            desc_mask=desc_mask.cuda(),
                            img_edge_index=img_edge,
                            orig_word_seq=org_seq,
                            desc_word_seq=None,
                            txt_edge_index=edge_cap1,
                            gnn_mask=gnn_mask_1.cuda(),
                            np_mask=np_mask_1.cuda(),
                            img_edge_attr=None,
                            key_padding_mask_img=key_padding_mask_img)
            final_label = innovative_dynamic_fusion_batch(y_cl, database_feature, database_label, k=20,
                                                          base_sigma=0.5)


            loss = cross_entropy_loss(y, labels.cuda())
            val_loss += float(loss.clone().detach().item())

            predict = predict + final_label
            real_label = real_label + labels.cpu().numpy().tolist()
            del img_batch, embed_t, embed_d
        print(f'coast:{(time.time() - t) / batch_idx:.4f}s')

        torch.cuda.empty_cache()  
        acc, recall, precision, f1 = get_four_metrics(real_label, predict)
        logger.log(mode="val", scalar_value=val_loss / len(val_loader), epoch=epoch, scalar_name='loss')
        logger.log(mode="val", scalar_value=acc, epoch=epoch, scalar_name='accuracy')
        print(' Val Epoch: {} Avg loss: {:.4f} Acc: {:.4f}  Rec: {:.4f} Pre: {:.4f} F1: {:.4f}'.format(epoch, val_loss / len(val_loader), acc, recall,
                                                                                                precision, f1))
    return val_loss





def evaluate_model_test(epoch, test_loader, train_loader):
    """
        Performs one validation epoch and computes loss and accuracy on the validation set
        Args:
            epoch (int): Current epoch number
            test_loader:
        Returns:
            val_loss (float): Average loss on the validation set
    """
    test_loss = 0.
    predict = []
    real_label = []
    model.eval()

    with torch.no_grad():
        database_feature = []
        database_label = []
        for batch_idx, (img_batch, img_edge, embed_t, embed_d, org_seq, org_word_len, mask_batch1, desc_mask,
                        edge_cap1, gnn_mask_1, np_mask_1, labels, key_padding_mask_img, target_labels) in enumerate(tqdm(train_loader)):
            embed_t = {k: v.to(device) for k, v in embed_t.items()}
            embed_d = {k: v.to(device) for k, v in embed_d.items()}
            if batch_idx == 16:
                break
            for i in range(len(target_labels)):
                target_labels[i] = target_labels[i].cuda()
            y, y_cl = model(imgs=img_batch.cuda(),
                            orig_texts=embed_t,
                            desc_texts=embed_d,
                            mask_batch=mask_batch1.cuda(),
                            desc_mask=desc_mask.cuda(),
                            img_edge_index=img_edge,
                            orig_word_seq=org_seq,
                            desc_word_seq=None,
                            txt_edge_index=edge_cap1,
                            gnn_mask=gnn_mask_1.cuda(),
                            np_mask=np_mask_1.cuda(),
                            img_edge_attr=None,
                            key_padding_mask_img=key_padding_mask_img)

            database_feature.append(y_cl)
            database_label.append(labels)
        database_feature = torch.cat(database_feature, dim=0).cuda()
        database_label = torch.cat(database_label, dim=0).cuda()
        for batch_idx, (img_batch, img_edge, embed_t, embed_d, org_seq, org_word_len, mask_batch1, desc_mask,
                        edge_cap1, gnn_mask_1, np_mask_1, labels, key_padding_mask_img, target_labels) in enumerate(
            tqdm(test_loader)):
            embed_t = {k: v.to(device) for k, v in embed_t.items()}
            embed_d = {k: v.to(device) for k, v in embed_d.items()}

            y, y_cl = model(imgs=img_batch.cuda(),
                            orig_texts=embed_t,
                            desc_texts=embed_d,
                            mask_batch=mask_batch1.cuda(),
                            desc_mask=desc_mask.cuda(),
                            img_edge_index=img_edge,
                            orig_word_seq=org_seq,
                            desc_word_seq=None,
                            txt_edge_index=edge_cap1,
                            gnn_mask=gnn_mask_1.cuda(),
                            np_mask=np_mask_1.cuda(),
                            img_edge_attr=None,
                            key_padding_mask_img=key_padding_mask_img)

            final_label = innovative_dynamic_fusion_batch(y_cl, database_feature, database_label, k=20,
                                                          base_sigma=0.5)

            loss = cross_entropy_loss(y, labels.cuda())
            test_loss += float(loss.clone().detach().item())

            predict = predict + final_label
            real_label = real_label + labels.cpu().numpy().tolist()
            del img_batch, embed_t, embed_d

        torch.cuda.empty_cache()  
    acc, recall, precision, f1 = get_four_metrics(real_label, predict)

    logger.log(mode="test", scalar_value=test_loss / len(test_loader), epoch=epoch, scalar_name='loss')
    logger.log(mode="test", scalar_value=acc, epoch=epoch, scalar_name='accuracy')
    print(' Test Epoch: {} Avg loss: {:.4f} Acc: {:.4f}  Rec: {:.4f} Pre: {:.4f} F1: {:.4f}'.format(epoch,
                                                                                                    test_loss / len(
                                                                                                        test_loader),
                                                                                                    acc, recall,
                                                                                                    precision, f1))
    return test_loss


def test_match_accuracy(val_loader):
    """
    Args:
        Once the model is trained, it is used to evaluate the how accurately the captions align with the objects in the image
    """
    try:
        print("Loading Saved Model")
        checkpoint = torch.load(args.save)
        model.load_state_dict(checkpoint)
        print("Saved Model successfully loaded")
        val_loss = 0.
        predict = []
        real_label = []
        pv_list = []
        a_list = []
        model.eval()
        with torch.no_grad():
            for batch_idx, (img_batch, img_edge, embed_t, embed_d, org_seq, org_word_len, mask_batch1, desc_mask,
                            edge_cap1, gnn_mask_1, np_mask_1, labels, key_padding_mask_img, target_labels) in enumerate(
                tqdm(val_loader)):
                embed_t = {k: v.to(device) for k, v in embed_t.items()}
                embed_d = {k: v.to(device) for k, v in embed_d.items()}

                with torch.no_grad():
                    y, a, pv = model( imgs=img_batch.cuda(),
                                    orig_texts=embed_t,
                                    desc_texts=embed_d,
                                    mask_batch=mask_batch1.cuda(),
                                    desc_mask=desc_mask.cuda(),
                                    img_edge_index=img_edge,
                                    orig_word_seq=org_seq,
                                    desc_word_seq=None,
                                    txt_edge_index=edge_cap1,
                                    gnn_mask=gnn_mask_1.cuda(),
                                    np_mask=np_mask_1.cuda(),
                                      img_edge_attr=None,
                                      key_padding_mask_img=key_padding_mask_img)

                    loss = cross_entropy_loss(y, labels.cuda())
                    val_loss += float(loss.clone().detach().item())

                predict = predict + get_metrics(y.cpu())
                real_label = real_label + labels.cpu().numpy().tolist()
                pv_list.append(pv.cpu().clone().detach())
                a_list.append(a.cpu().clone().detach())
                torch.cuda.empty_cache()
                del img_batch, embed_t, embed_d
            acc, recall, precision, f1 = get_four_metrics(real_label, predict)
            save_result = {"real_label": real_label, 'predict_label': predict, "pv_list": pv_list,
                           " a_list": a_list}
            torch.save(save_result, "with_out_knowledge")

        print(
            "Avg loss: {:.4f} Acc: {:.4f}  Rec: {:.4f} Pre: {:.4f} F1: {:.4f}".format(val_loss, acc, recall, precision,
                                                                                      f1))
    except Exception as e:
        print(e)
        exit()


def main():

    if args.mode == 'train':
        annotation_train = os.path.join(annotation_files, "traindep.json")
        annotation_val = os.path.join(annotation_files, "valdep.json")
        annotation_test = os.path.join(annotation_files, "testdep.json")

        img_train = os.path.join(img_files, "train_box.pt")  #
        img_val = os.path.join(img_files, "val_box.pt")
        img_test = os.path.join(img_files, "test_box.pt")
        img_edge_train = os.path.join(img_files, "train_edge.pt")  #
        img_edge_val = os.path.join(img_files, "val_edge.pt")
        img_edge_test = os.path.join(img_files, "test_edge.pt")
        train_dataset = BaseSet(type="train", max_length=parameter["max_length"], text_path=annotation_train,
                                use_np=use_np, img_path=img_train, edge_path=img_edge_train,
                                 desc_path="image_descriptions.jsonl"
                                )
        val_dataset = BaseSet(type="val", max_length=parameter["max_length"], text_path=annotation_val, use_np=use_np,
                              img_path=img_val, edge_path=img_edge_val,
                              desc_path="image_descriptions.jsonl"  
                              )
        test_dataset = BaseSet(type="test", max_length=parameter["max_length"], text_path=annotation_test,
                               use_np=use_np, img_path=img_test, edge_path=img_edge_test,
                               desc_path="image_descriptions.jsonl"  
                               )

        train_loader = DataLoader(dataset=train_dataset, batch_size=parameter["batch_size"], num_workers=4,
                                  shuffle=True,
                                  collate_fn=PadCollate_without_know())

        print("training dataset has been loaded successful!")
        val_loader = DataLoader(dataset=val_dataset, batch_size=parameter["batch_size"], num_workers=4,
                                shuffle=True,
                                collate_fn=PadCollate_without_know())
        print("validation dataset has been loaded successful!")
        test_loader = DataLoader(dataset=test_dataset, batch_size=parameter["batch_size"], num_workers=4,
                                 shuffle=True,
                                 collate_fn=PadCollate_without_know())
        print("test dataset has been loaded successful!")

        start_epoch = 0
        patience = 8

        if args.path is not None and not os.path.exists(args.path):
            os.mkdir(args.path)
        try:
            print("Loading Saved Model")
            checkpoint = torch.load(args.save)
            model.load_state_dict(checkpoint)
            start_epoch = int(re.search("-\d+", args.save).group()[1:]) + 1
            print("Saved Model successfully loaded")
            # Only effect special layers like dropout layer
            model.eval()
            best_loss = eval_validation_loss(val_loader=val_loader)
        except:
            print("Failed, No Saved Model")
            best_loss = np.Inf
        early_stop = False
        counter = 0
        for epoch in range(start_epoch + 1, parameter["epochs"] + 1):
            # Training epoch
            train_model(epoch=epoch, train_loader=train_loader)
            # Validation epoch
            avg_val_loss = evaluate_model(epoch, val_loader=val_loader, train_loader=train_loader)
            avg_test_loss = evaluate_model_test(epoch, test_loader=test_loader, train_loader=train_loader)

            scheduler.step(avg_val_loss)
            if avg_val_loss <= best_loss:
                counter = 0
                best_loss = avg_val_loss
                # torch.save(model.state_dict(), os.path.join(args.path, parameter["model_name"] + '-' + str(epoch) + '.pt'))
                print("Best model saved/updated..")
                torch.cuda.empty_cache()
            else:
                counter += 1
                if counter >= patience:
                    early_stop = True
            # If early stopping flag is true, then stop the training
            torch.save(model.state_dict(), os.path.join(args.path, parameter["model_name"] + '-' + str(epoch) + '.pt'))
            if early_stop:
                print("Early stopping")
                break

    elif args.mode == 'eval':
        # args.save
        annotation_test = os.path.join(annotation_files, "testdep.json")
        img_test = os.path.join(img_files, "test_B32.pt")

        test_dataset = BaseSet(type="test", max_length=parameter["max_length"], text_path=annotation_test,
                               use_np=use_np,
                               img_path=img_test)
        test_loader = DataLoader(dataset=test_dataset, batch_size=parameter["batch_size"], shuffle=False,
                                 collate_fn=PadCollate(use_np=use_np, max_know_len=parameter["know_max_length"]))

        print("validation dataset has been loaded successful!")
        test_match_accuracy(val_loader=test_loader)

    else:
        print("Mode of SSGN is error!")


if __name__ == "__main__":
    main()
    # seed_everything(42)