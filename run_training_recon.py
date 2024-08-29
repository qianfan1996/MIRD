# -*-coding:utf-8-*- 
from __future__ import absolute_import, division, print_function

import argparse
import time
import numpy as np
from tqdm import tqdm
from visdom import Visdom

from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.nn import MSELoss, CrossEntropyLoss

from transformers import get_linear_schedule_with_warmup
from transformers.optimization import AdamW

from models import WSABERTRecon

from utils import set_random_seed, get_device, get_parameter_number, write_txt, interval_time
from data_loader import set_up_data_loader


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["mosi", "mosei"], default="mosi")
parser.add_argument("--multimodal_encoder", type=str, choices=["MAG-BERT", "WA-BERT", "WSA-BERT"], default="WSA-BERT")
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--seed", type=int, default=2022)
parser.add_argument("--num_epoch", type=int, default=40)
parser.add_argument("--backbone_model", type=str, default="bert-base-uncased")

parser.add_argument("--train_batch_size", type=int, default=48)
parser.add_argument("--valid_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--warmup_proportion", type=float, default=0.1)

parser.add_argument("--max_words_length", type=int, default=50)
parser.add_argument("--beta_shift", type=float, default=1.0)
parser.add_argument("--dropout_prob", type=float, default=0.5)
parser.add_argument("--orig_d_a", type=int, default=33)
parser.add_argument("--orig_d_v", type=int, default=709)

parser.add_argument("--zy_size", type=int, default=64)
parser.add_argument("--zl_size", type=int, default=64)
parser.add_argument("--za_size", type=int, default=64)
parser.add_argument("--zv_size", type=int, default=64)
parser.add_argument("--zy_to_y_dropout", type=float, default=0.5)

parser.add_argument("--language_encoder_use_lstm", action='store_true')
parser.add_argument("--output_dim_l", type=int, default=30522)
parser.add_argument("--hidden_size_l", type=int, default=768)
parser.add_argument("--dropout_l", type=float, default=0.5)

parser.add_argument("--hidden_size_multi", type=int, default=768)
parser.add_argument("--dropout_multi", type=float, default=0.5)

parser.add_argument("--label_dim", type=int, default=1)

parser.add_argument("--lambda1", type=float, default=0.1)
parser.add_argument("--lambda2", type=float, default=10)
parser.add_argument("--lambda3", type=float, default=100)

args = parser.parse_args()


class MultimodalConfig(object):
    def __init__(self, beta_shift, dropout_prob, orig_d_a, orig_d_v, zy_size, zl_size, za_size, zv_size, zy_to_y_dropout,
                 language_encoder_use_lstm, output_dim_l, hidden_size_l, dropout_l, hidden_size_multi, dropout_multi, label_dim
    ):
        self.beta_shift = beta_shift
        self.dropout_prob = dropout_prob
        self.orig_d_a, self.orig_d_v = orig_d_a, orig_d_v
        self.zy_size = zy_size
        self.zl_size = zl_size
        self.za_size = za_size
        self.zv_size = zv_size
        self.zy_to_y_dropout = zy_to_y_dropout

        self.language_encoder_use_lstm = language_encoder_use_lstm
        self.output_dim_l = output_dim_l
        self.hidden_size_l = hidden_size_l
        self.dropout_l = dropout_l

        self.hidden_size_multi = hidden_size_multi
        self.dropout_multi = dropout_multi

        self.label_dim = label_dim


def prep_for_training(num_train_optimization_steps):
    multimodal_config = MultimodalConfig(
        beta_shift=args.beta_shift, dropout_prob=args.dropout_prob, orig_d_a=args.orig_d_a, orig_d_v=args.orig_d_v,
        zy_size=args.zy_size, zl_size=args.zl_size, za_size=args.za_size, zv_size=args.zv_size, zy_to_y_dropout=args.zy_to_y_dropout,
        language_encoder_use_lstm=args.language_encoder_use_lstm, output_dim_l=args.output_dim_l, hidden_size_l=args.hidden_size_l,
        dropout_l = args.dropout_l, hidden_size_multi = args.hidden_size_multi,
        dropout_multi = args.dropout_multi, label_dim=args.label_dim
    )

    model = WSABERTRecon.from_pretrained(
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
    # viz = Visdom()
    # viz.line([[0., 0., 0., 0.]], [0], win='train',
    #          opts=dict(title='train loss', legend=["prediction loss", "l_recon loss", "a_recon loss", "v_recon loss"]))
    model.train()
    train_loss = 0
    train_steps = 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(get_device()) for t in batch)
        input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)
        zl, za, zv, zy, x_l_hat, x_a_hat, x_v_hat, y_hat = model(
            input_ids,
            visual,
            acoustic,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
        )

        mse_loss = MSELoss()
        ce_loss = CrossEntropyLoss()

        input_ids = input_ids.view(-1)
        output_dim = x_l_hat.size(-1)
        x_l_hat = x_l_hat.contiguous().view(-1, output_dim)

        pred_loss = mse_loss(y_hat.view(-1), label_ids.view(-1))
        recon_loss = args.lambda1 * ce_loss(x_l_hat, input_ids) + args.lambda2 * mse_loss(x_a_hat, acoustic) + \
                     args.lambda3 * mse_loss(x_v_hat, visual)

        # print(ce_loss(x_l_hat, input_ids), mse_loss(x_a_hat, acoustic), mse_loss(x_v_hat, visual))
        # viz.line([[pred_loss.item(), args.lambda1 * ce_loss(x_l_hat, input_ids).item(),
        #            args.lambda2 * mse_loss(x_a_hat, acoustic).item(),
        #            args.lambda3 * mse_loss(x_v_hat, visual).item()]],
        #          [step], win="train", update="append")

        loss = pred_loss + recon_loss

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
            zl, za, zv, zy, x_l_hat, x_a_hat, x_v_hat, y_hat = model(
                input_ids,
                visual,
                acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
            )

            mse_loss = MSELoss()
            # ce_loss = CrossEntropyLoss()

            # input_ids = input_ids.view(-1)
            # output_dim = x_l_hat.size(-1)
            # x_l_hat = x_l_hat.contiguous().view(-1, output_dim)

            pred_loss = mse_loss(y_hat.view(-1), label_ids.view(-1))
            # recon_loss = args.lambda1 * ce_loss(x_l_hat, input_ids) + args.lambda2 * mse_loss(x_a_hat, acoustic) + \
                         # args.lambda3 * mse_loss(x_v_hat, visual)
            # loss = pred_loss + recon_loss

            valid_loss += pred_loss.item()
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
            zl, za, zv, zy, x_l_hat, x_a_hat, x_v_hat, y_hat = model(
                input_ids,
                visual,
                acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
            )

            y_hat = y_hat.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()

            y_hat = np.squeeze(y_hat).tolist()
            label_ids = np.squeeze(label_ids).tolist()

            preds.extend(y_hat)
            labels.extend(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)

    return preds, labels


def get_test_feat(model, test_dataloader):
    model.eval()
    language_feat = []
    visual_feat = []
    audio_feat = []
    multi_feat = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(get_device()) for t in batch)

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)

            zl, za, zv, zy, x_l_hat, x_a_hat, x_v_hat, y_hat = model(
                input_ids,
                visual,
                acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
            )

            zl = zl.detach().cpu().numpy()
            za = za.detach().cpu().numpy()
            zv = zv.detach().cpu().numpy()
            zy = zy.detach().cpu().numpy()

            language_feat.append(zl)
            visual_feat.append(zv)
            audio_feat.append(za)
            multi_feat.append(zy)

        language_feat = np.concatenate(language_feat, axis=0)
        visual_feat = np.concatenate(visual_feat, axis=0)
        audio_feat = np.concatenate(audio_feat, axis=0)
        multi_feat = np.concatenate(multi_feat, axis=0)

        print(language_feat.shape, visual_feat.shape, audio_feat.shape, multi_feat.shape)

        np.save('saved_feat/language_feat.npy', language_feat)
        np.save('saved_feat/visual_feat.npy', visual_feat)
        np.save('saved_feat/audio_feat.npy', audio_feat)
        np.save('saved_feat/multi_feat.npy', multi_feat)


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

    max_valid_loss = 999999
    for epoch_i in range(int(args.num_epoch)):
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler)
        valid_loss = eval_epoch(model, test_dataloader)

        print("epoch:{}, train_loss:{}, valid_loss:{}".format(epoch_i, train_loss, valid_loss))

        if valid_loss < max_valid_loss:
            max_valid_loss = valid_loss
            print('Saving model ...')
            if args.multimodal_encoder == "WA-BERT":
                torch.save(model, 'saved_models/WA-BERT_Recon_{}.pt'.format(args.dataset))
            elif args.multimodal_encoder == "MAG-BERT":
                torch.save(model, 'saved_models/MAG-BERT_Recon_{}.pt'.format(args.dataset))
            else:
                torch.save(model, 'saved_models/WSA-BERT_Recon_{}.pt'.format(args.dataset))


    if args.multimodal_encoder == "WA-BERT":
        model = torch.load('saved_models/WA-BERT_Recon_{}.pt'.format(args.dataset))
    elif args.multimodal_encoder == "MAG-BERT":
        model = torch.load('saved_models/MAG-BERT_Recon_{}.pt'.format(args.dataset))
    else:
        model = torch.load('saved_models/WSA-BERT_Recon_{}.pt'.format(args.dataset))

    test_acc, test_f_score, test_mae, test_corr = test_score_model(model, test_dataloader)
    print("\033[1;31mAccuracy: {}, F1_score: {}, MAE: {}, Corr: {}\033[0m".format(test_acc, test_f_score, test_mae, test_corr))

    saved_results_path = "saved_results/results.txt"
    results = ["Experimental Results of WSA-BERT (adding FC layer) with Modality Reconstruction:\n",
               "Hyperparameters: seed {}, dataset {}, num_epoch {}, batch_size {}, "
               "learning_rate {}, lambda1 {}, lambda2 {}, lambda3 {}\n".format(
                   args.seed,
                   args.dataset,
                   args.num_epoch,
                   args.train_batch_size,
                   args.learning_rate,
                   args.lambda1,
                   args.lambda2,
                   args.lambda3
               ),
               "Acc2: {:5.4f}\n".format(test_acc), "F1 score: {:5.4f}\n".format(test_f_score),
               "MAE: {:5.3f}\n".format(test_mae),
               "Corr: {:5.3f}\n".format(test_corr), "-" * 100 + "\n"
    ]
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
    model = torch.load('saved_models/WSA-BERT_Recon_mosi.pt')
    train_data_loader, valid_data_loader, test_data_loader, num_train_optimization_steps = set_up_data_loader(
        dataset="mosi",
        max_words_length=50,
        train_batch_size=48,
        valid_batch_size=128,
        test_batch_size=128,
        num_epoch=100,
    )
    get_test_feat(model, test_data_loader)
    print("Have saved middle latent representations as npy file.")