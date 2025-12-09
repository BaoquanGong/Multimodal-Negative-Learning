#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_pretrained_bert import BertAdam

from src.data.helpers import get_data_loaders
from src.models import get_model
from src.utils.logger import create_logger
from src.utils.utils import *
import wandb
import random
import time

def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=128)
    parser.add_argument("--bert_model", type=str, default="/root/MNL/bert-base-uncased")#, choices=["bert-base-uncased", "bert-large-uncased"])
    parser.add_argument("--data_path", type=str, default="/path/to/data_dir/")
    parser.add_argument("--drop_img_percent", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed_sz", type=int, default=300)
    parser.add_argument("--freeze_img", type=int, default=0)
    parser.add_argument("--freeze_txt", type=int, default=0)
    parser.add_argument("--glove_path", type=str, default="./datasets/glove_embeds/glove.840B.300d.txt")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=24)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--model", type=str, default="bow", choices=["latefusion"])
    parser.add_argument("--n_workers", type=int, default=12)
    parser.add_argument("--name", type=str, default="nameless")
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--savedir", type=str, default="/path/to/save_dir/")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--task", type=str, default="MVSA_Single")
    parser.add_argument("--task_type", type=str, default="classification", choices=["classification"])
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight_classes", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--noise_type", type=str, default="Gaussian")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--weight1", type=float, default=0.7)


def get_criterion(args):
    criterion = nn.CrossEntropyLoss()
    return criterion

def get_optimizer(model, args):

    param_optimizer = list(model.named_parameters())
    decay=[]
    name = [n for n, p in param_optimizer if any(nd in n for nd in decay)]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in decay)], "weight_decay": 0.0, },
    ]
    optimizer = optim.Adam(optimizer_grouped_parameters, lr=args.lr)
    return optimizer


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, factor=args.lr_factor
    )


def model_eval(i_epoch, data, model, args, criterion,optimizer,store_preds=False):
    with torch.no_grad():
        losses, preds, tgts,tcp_consists =[], [], [], []
        img_preds,txt_preds=[],[]
        for batch in data:
            loss,out,tgt,txt_logits,img_logits = model_forward(i_epoch, model, args, criterion,optimizer, batch,mode='eval')
            losses.append(loss.item())
            pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()
            txt_pred = torch.nn.functional.softmax(txt_logits, dim=1).argmax(dim=1).cpu().detach().numpy()
            img_pred = torch.nn.functional.softmax(img_logits, dim=1).argmax(dim=1).cpu().detach().numpy()

            preds.append(pred)
            txt_preds.append(txt_pred)
            img_preds.append(img_pred)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)

    metrics = {"loss": np.mean(losses)}
   
    tgts = [l for sl in tgts for l in sl]
    preds = [l for sl in preds for l in sl]
    txt_preds = [l for sl in txt_preds for l in sl]
    img_preds = [l for sl in img_preds for l in sl]

    metrics["acc"] = accuracy_score(tgts, preds)
    metrics["txt_acc"] = accuracy_score(tgts, txt_preds)
    metrics["img_acc"] = accuracy_score(tgts, img_preds)
    return metrics
def model_forward(i_epoch, model, args, criterion,optimizer, batch ,mode='eval'):
    
    txt, segment, mask, img, tgt,idx = batch

    txt, img = txt.cuda(), img.cuda()
    mask, segment = mask.cuda(), segment.cuda()
    tgt = tgt.cuda()

    out, txt_logits, img_logits = model(txt, mask,segment,img)
    label = F.one_hot(tgt, num_classes=args.n_classes) 
    if mode=='train':
        prob_out = F.softmax(out, dim=1) 
        prob_out_masked = prob_out.clone()
        prob_out_masked[torch.arange(prob_out.size(0)), tgt] = -1

        _, j_idx = prob_out_masked.max(dim=1)  # j_idx: [B]

        txt_tgt_logits = txt_logits[torch.arange(txt_logits.size(0)), tgt]     # [B]
        txt_j_logits = txt_logits[torch.arange(txt_logits.size(0)), j_idx]     # [B]
        xi_t = txt_tgt_logits - txt_j_logits                                 # [B] 

        
        img_tgt_logits = img_logits[torch.arange(img_logits.size(0)), tgt]     # [B]
        img_j_logits = img_logits[torch.arange(img_logits.size(0)), j_idx]     # [B]
        xi_v = img_tgt_logits - img_j_logits                                 # [B]
        
        prob_txt = F.softmax(txt_logits, dim=1)  # [B, C]
        prob_v = F.softmax(img_logits, dim=1)

        label_probs_txt = prob_txt[torch.arange(prob_txt.size(0)), tgt]
        label_probs_v = prob_v[torch.arange(prob_v.size(0)), tgt]
        
        txt_clf_loss = nn.CrossEntropyLoss()(txt_logits, tgt)
        img_clf_loss = nn.CrossEntropyLoss()(img_logits, tgt)
        clf_loss = txt_clf_loss + img_clf_loss + nn.CrossEntropyLoss()(out, tgt)
        
        loss = torch.mean(clf_loss) 
      
        valid_mask = (xi_t > xi_v) & (label_probs_txt > label_probs_v )
        if valid_mask.any():
            valid_mask = valid_mask.float().unsqueeze(1)
            boosting_loss = -  (label * img_logits.log_softmax(1)* valid_mask ) \
                            - (args.weight1) *(txt_logits.detach().softmax(1) * img_logits.log_softmax(1) * valid_mask) \
                            + (args.weight1) *(label * txt_logits.detach().softmax(1) * img_logits.log_softmax(1) * valid_mask)
            loss += torch.mean(boosting_loss)
            
        valid_mask = (xi_t <= xi_v) & (label_probs_txt <= label_probs_v )
        if valid_mask.any():
            valid_mask = valid_mask.float().unsqueeze(1)
            boosting_loss = -  (label * txt_logits.log_softmax(1)* valid_mask) \
                             - (args.weight1) *(img_logits.detach().softmax(1) * txt_logits.log_softmax(1) * valid_mask) \
                             + (args.weight1) *(label * img_logits.detach().softmax(1) * txt_logits.log_softmax(1) * valid_mask)
            loss += torch.mean(boosting_loss)

                
        return loss, out, tgt
    else:
        txt_clf_loss = nn.CrossEntropyLoss()(txt_logits, tgt)
        img_clf_loss = nn.CrossEntropyLoss()(img_logits, tgt)
        clf_loss = txt_clf_loss + img_clf_loss + nn.CrossEntropyLoss()(out, tgt)
        loss = torch.mean(clf_loss)
        return loss, out, tgt,txt_logits, img_logits,


def train(args):
    task = args.task
    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, f"{args.name}")
    os.makedirs(args.savedir, exist_ok=True)
    train_loader, val_loader, test_loaders = get_data_loaders(args)
    model = get_model(args)
    criterion = get_criterion(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    logger = create_logger("%s/logfile.log" % args.savedir, args)
    model.cuda()
    torch.save(args, os.path.join(args.savedir, "args.pt"))
    start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf
    if os.path.exists(os.path.join(args.savedir, "checkpoint.pt")):
        checkpoint = torch.load(os.path.join(args.savedir, "checkpoint.pt"))
        start_epoch = checkpoint["epoch"]
        n_no_improve = checkpoint["n_no_improve"]
        best_metric = checkpoint["best_metric"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
    logger.info("Training..")


    for i_epoch in range(start_epoch, args.max_epochs):
        train_losses = []
        tcp_consists = []

        model.train()
        optimizer.zero_grad()

        for batch in tqdm(train_loader, total=len(train_loader)):
            
            loss, _, _= model_forward(i_epoch, model, args, criterion,optimizer, batch,mode='train')

            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        metrics = model_eval(i_epoch, val_loader, model, args, criterion,optimizer)

        logger.info("Train Loss: {:.4f}".format(np.mean(train_losses)))
        log_metrics("Val", metrics, args, logger)

        tuning_metric = (metrics["acc"])
        scheduler.step(tuning_metric)
        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
        else:
            n_no_improve += 1

        save_checkpoint(
            {
                "epoch": i_epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "n_no_improve": n_no_improve,
                "best_metric": best_metric,
            },
            is_improvement,
            args.savedir,
        )
        if n_no_improve >= args.patience:
            logger.info("No improvement. Breaking out of loop.")
            break

    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    model.eval()

    for test_name, test_loader in test_loaders.items():
        test_metrics = model_eval(
            np.inf, test_loader, model, args, criterion, optimizer,store_preds=True
        )
        log_metrics(f"Test - {test_name}", test_metrics, args, logger)
def cli_main():
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    train(args)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    cli_main()
