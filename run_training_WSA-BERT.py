# -*-coding:utf-8-*- 
from __future__ import absolute_import, division, print_function

import argparse
import time
import numpy as np
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.nn import MSELoss

from transformers import get_linear_schedule_with_warmup
from transformers.optimization import AdamW

from models import BertForSequenceClassification

from utils import set_random_seed, get_device, get_parameter_number, write_txt, interval_time
from data_loader import set_up_data_loader


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["mosi", "mosei"], default="mosi")
parser.add_argument("--multimodal_encoder", type=str, choices=["MAG-BERT", "WA-BERT", "WSA-BERT"], default="WSA-BERT")
parser.add_argument("--max_words_length", type=int, default=50)
parser.add_argument("--train_batch_size", type=int, default=48)
parser.add_argument("--valid_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--num_epoch", type=int, default=40)
parser.add_argument("--beta_shift", type=float, default=1.0)
parser.add_argument("--dropout_prob", type=float, default=0.5)
parser.add_argument("--dropout2", type=float, default=0.5)
parser.add_argument("--dropout3", type=float, default=0.5)
parser.add_argument("--latent_dim", type=int, default=128)
parser.add_argument("--hidden_size_pred", type=int, default=64)
parser.add_argument("--backbone_model", type=str, default="bert-base-uncased")
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=2022)

args = parser.parse_args()


class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob, dropout2, dropout3, latent_dim, hidden_size_pred):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob
        self.dropout2 = dropout2
        self.dropout3 = dropout3
        self.latent_dim = latent_dim
        self.hidden_size_pred = hidden_size_pred


def prep_for_training(num_train_optimization_steps):
    multimodal_config = MultimodalConfig(
        beta_shift=args.beta_shift, dropout_prob=args.dropout_prob, dropout2=args.dropout2, dropout3=args.dropout3,
        latent_dim=args.latent_dim, hidden_size_pred=args.hidden_size_pred
    )


    model = BertForSequenceClassification.from_pretrained(
        args.backbone_model, multimodal_config=multimodal_config, model=args.multimodal_encoder, num_labels=1
    )

    model.to(get_device())

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_train_optimization_steps,
        num_training_steps=args.warmup_proportion * num_train_optimization_steps,
    )
    return model, optimizer, scheduler


def train_epoch(model, train_dataloader, optimizer, scheduler):
    model.train()
    train_loss = 0
    train_steps = 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(get_device()) for t in batch)
        input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)
        logits = model(
            input_ids,
            visual,
            acoustic,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
        )
        criterion = MSELoss()
        loss = criterion(logits.view(-1), label_ids.view(-1))

        loss.backward()

        train_loss += loss.item()
        train_steps += 1

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return train_loss / train_steps


def eval_epoch(model, valid_dataloader):
    model.eval()
    valid_loss = 0
    valid_steps = 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(valid_dataloader, desc="Iteration")):
            batch = tuple(t.to(get_device()) for t in batch)

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            logits = model(
                input_ids,
                visual,
                acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
            )

            criterion = MSELoss()
            loss = criterion(logits.view(-1), label_ids.view(-1))

            valid_loss += loss.item()
            valid_steps += 1

    return valid_loss / valid_steps


def test_epoch(model, test_dataloader):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(get_device()) for t in batch)

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            logits = model(
                input_ids,
                visual,
                acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
            )

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()

            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()

            preds.extend(logits)
            labels.extend(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)

    return preds, labels


def test_score_model(model, test_dataloader, use_zero=False):

    preds, y_test = test_epoch(model, test_dataloader)
    non_zeros = np.array(
        [i for i, e in enumerate(y_test) if e != 0 or use_zero]
    )

    preds = preds[non_zeros]
    y_test = y_test[non_zeros]

    mae = np.mean(np.absolute(preds - y_test))
    corr = np.corrcoef(preds, y_test)[0][1]

    preds = preds >= 0
    y_test = y_test >= 0

    f_score = f1_score(y_test, preds, average="weighted")
    acc = accuracy_score(y_test, preds)

    return acc, f_score, mae, corr


def train(
    model,
    train_dataloader,
    valid_dataloader,
    test_dataloader,
    optimizer,
    scheduler,
):

    max_valid_loss = 999
    for epoch_i in range(int(args.num_epoch)):
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler)
        valid_loss = eval_epoch(model, test_dataloader)

        print("epoch:{}, train_loss:{}, valid_loss:{}".format(epoch_i, train_loss, valid_loss))

        if valid_loss < max_valid_loss:
            max_valid_loss = valid_loss
            print('Saving model ...')
            if args.multimodal_encoder == "WA-BERT":
                torch.save(model, 'saved_models/WA-BERT_{}.pt'.format(args.dataset))
            elif args.multimodal_encoder == "MAG-BERT":
                torch.save(model, 'saved_models/MAG-BERT_{}.pt'.format(args.dataset))
            else:
                torch.save(model, 'saved_models/WSA-BERT_{}.pt'.format(args.dataset))


    if args.multimodal_encoder == "WA-BERT":
        model = torch.load('saved_models/WA-BERT_{}.pt'.format(args.dataset))
    elif args.multimodal_encoder == "MAG-BERT":
        model = torch.load('saved_models/MAG-BERT_{}.pt'.format(args.dataset))
    else:
        model = torch.load('saved_models/WSA-BERT_{}.pt'.format(args.dataset))

    test_acc, test_f_score, test_mae, test_corr = test_score_model(model, test_dataloader)
    print("\033[1;31mAccuracy: {}, F1_score: {}, MAE: {}, Corr: {}\033[0m".format(test_acc, test_f_score, test_mae, test_corr))

    saved_results_path = "saved_results/results.txt"
    results = ["Experimental Results of WSA-BERT (adding FC layer):\n",
               "Hyperparameters: seed {}, dataset {}, num_epoch {}, batch_size {}, "
               "learning_rate {}\n".format(
                   args.seed,
                   args.dataset,
                   args.num_epoch,
                   args.train_batch_size,
                   args.learning_rate,
               ),
               "Acc2: {:5.4f}\n".format(test_acc), "F1 score: {:5.4f}\n".format(test_f_score),
               "MAE: {:5.3f}\n".format(test_mae),
               "Corr: {:5.3f}\n".format(test_corr), "-" * 100 + "\n"]
    write_txt(saved_results_path, results)


def main():
    set_random_seed(args.seed)
    train_data_loader, valid_data_loader, test_data_loader, num_train_optimization_steps = set_up_data_loader(
        dataset=args.dataset,
        max_words_length=args.max_words_length,
        train_batch_size=args.train_batch_size,
        valid_batch_size=args.valid_batch_size,
        test_batch_size=args.test_batch_size,
        num_epoch=args.num_epoch,
    )

    model, optimizer, scheduler = prep_for_training(num_train_optimization_steps)
    print("\033[1;35mTotal parameters: {}, Trainable parameters: {}\033[0m".format(*get_parameter_number(model)))

    train(
        model,
        train_data_loader,
        valid_data_loader,
        test_data_loader,
        optimizer,
        scheduler,
    )


if __name__ == "__main__":
    start = time.time()
    main()
    print("Time Usage: {} minutes {} seconds".format(*interval_time(start, time.time())))
    print("=" * 200)