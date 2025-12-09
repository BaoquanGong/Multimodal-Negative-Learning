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
    parser.add_argument("--bert_model", type=str, default="/root/PDF/bert-base-uncased")#, choices=["bert-base-uncased", "bert-large-uncased"])
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
    return nn.CrossEntropyLoss()

def model_eval(data_loader, model, args, criterion):
    model.eval()
    preds, tgts = [], []
    losses = []

    with torch.no_grad():
        for batch in data_loader:
            txt, segment, mask, img, tgt, idx = batch
            txt, img = txt.cuda(), img.cuda()
            mask, segment = mask.cuda(), segment.cuda()
            tgt = tgt.cuda()

            out, txt_logits, img_logits = model(txt, mask, segment, img)
            loss = nn.CrossEntropyLoss()(out, tgt) + \
                   nn.CrossEntropyLoss()(txt_logits, tgt) + \
                   nn.CrossEntropyLoss()(img_logits, tgt)
            losses.append(loss.item())

            pred = F.softmax(out, dim=1).argmax(dim=1).cpu().numpy()
            tgt_np = tgt.cpu().numpy()
            preds.append(pred)
            tgts.append(tgt_np)

    preds = np.concatenate(preds)
    tgts = np.concatenate(tgts)

    metrics = {
        "loss": np.mean(losses),
        "acc": accuracy_score(tgts, preds),
        "f1": f1_score(tgts, preds, average="macro")
    }
    return metrics

def test(args):
    set_seed(args.seed)
    logger = create_logger("%s/test_log.log" % args.savedir, args)
    
    # Load test data
    _, _, test_loaders = get_data_loaders(args)

    # Load model
    model = get_model(args)
    model.cuda()
    load_checkpoint(model, f"{args.savedir}/model_best.pt")
    model.eval()

    # Evaluate each test set
    for test_name, test_loader in test_loaders.items():
        metrics = model_eval(test_loader, model, args, get_criterion(args))
        log_metrics(f"Test - {test_name}", metrics, args, logger)

def cli_main():
    parser = argparse.ArgumentParser(description="Test Model")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    test(args)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    cli_main()
