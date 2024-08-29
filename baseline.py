# -*-coding:utf-8-*-
import numpy as np
import time
import argparse
from tqdm import trange
from sklearn.metrics import accuracy_score, f1_score
from visdom import Visdom

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models import MFM
from data_loader import load_mosi_mosei_glove_pkl, MyDataset
from utils import set_random_seed, get_parameter_number, interval_time, write_txt, load_json

start = time.time()


parser = argparse.ArgumentParser(description='Multimodal Sentiment Analysis')

# Tasks
parser.add_argument('--vonly', action='store_true',
                    help='use the crossmodal fusion into v (default: False)')
parser.add_argument('--aonly', action='store_true',
                    help='use the crossmodal fusion into a (default: False)')
parser.add_argument('--lonly', action='store_true',
                    help='use the crossmodal fusion into l (default: False)')
parser.add_argument('--dataset', type=str, default='mosi',
                    help='dataset to use (default: mosi)')
parser.add_argument('--data_path', type=str, default='data',
                    help='path for storing the dataset')

# Dropouts
parser.add_argument('--attn_dropout', type=float, default=0.1,
                    help='attention dropout')
parser.add_argument('--attn_dropout_a', type=float, default=0.0,
                    help='attention dropout (for audio)')
parser.add_argument('--attn_dropout_v', type=float, default=0.0,
                    help='attention dropout (for visual)')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.25,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.0,
                    help='output layer dropout')

# Architecture
parser.add_argument('--num_layers', type=int, default=5,
                    help='number of layers in the network (default: 5)')
parser.add_argument('--num_heads', type=int, default=5,
                    help='number of heads for the transformer network (default: 5)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')

# Tuning
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size (default: 32)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--learning_rate', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--use_lr_schedule', action='store_true',
                    help='if use learning_rate schedule (default: false)')
parser.add_argument('--num_epoch', type=int, default=100,
                    help='number of epochs (default: 40)')
parser.add_argument('--when', type=int, default=20,
                    help='when to decay learning rate using ReduceLROnPlateau (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='number of chunks per batch (default: 1)')

# Logistics
parser.add_argument('--log_interval', type=int, default=30,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=666,
                    help='random seed')

args = parser.parse_args()

current_setting = [args.seed, args.dataset, args.num_epoch, args.batch_size, args.learning_rate, args.clip, args.num_layers, args.num_heads]

set_random_seed(args.seed)

valid_partial_mode = args.lonly + args.vonly + args.aonly # default is 0
if valid_partial_mode == 0:
    args.lonly = args.vonly = args.aonly = True

hyp_params = args
hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = 300, 33, 709
hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = 50, 400, 55
hyp_params.layers = args.num_layers
hyp_params.when = args.when
hyp_params.batch_chunk = args.batch_chunk
hyp_params.output_dim = 1


if args.dataset == "mosi":
    train_text, train_audio, train_video, train_label = load_mosi_mosei_glove_pkl("data/CMU-MOSI/mosi_glove_normalized_.pkl", "train")
    valid_text, valid_audio, valid_video, valid_label = load_mosi_mosei_glove_pkl("data/CMU-MOSI/mosi_glove_normalized_.pkl", "valid")
    test_text, test_audio, test_video, test_label = load_mosi_mosei_glove_pkl("data/CMU-MOSI/mosi_glove_normalized_.pkl", "test")
else:
    train_text, train_audio, train_video, train_label = load_mosi_mosei_glove_pkl("data/CMU-MOSEI/mosei_glove_normalized_.pkl", "train")
    valid_text, valid_audio, valid_video, valid_label = load_mosi_mosei_glove_pkl("data/CMU-MOSEI/mosei_glove_normalized_.pkl", "valid")
    test_text, test_audio, test_video, test_label = load_mosi_mosei_glove_pkl("data/CMU-MOSEI/mosei_glove_normalized_.pkl", "test")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = MyDataset(train_text, train_audio, train_video, train_label, device)
valid_data = MyDataset(valid_text, valid_audio, valid_video, valid_label, device)
test_data = MyDataset(test_text, test_audio, test_video, test_label, device)

train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)

    return torch.exp(-kernel_input) # (x_size, y_size)


def loss_MMD(zy):
    zy_real_gauss = torch.randn(zy.size()) # no need to be the same size

    # if args.cuda:
    zy_real_gauss = zy_real_gauss.cuda()
    zy_real_kernel = compute_kernel(zy_real_gauss, zy_real_gauss)
    zy_fake_kernel = compute_kernel(zy, zy)
    zy_kernel = compute_kernel(zy_real_gauss, zy)
    zy_mmd = zy_real_kernel.mean() + zy_fake_kernel.mean() - 2.0*zy_kernel.mean()

    return zy_mmd

config = load_json('configs/mosi_baseline.json')

model = MFM(hyp_params, config)

print("\033[1;35mTotal parameters: {}, Trainable parameters: {}\033[0m".format(*get_parameter_number(model)))

# optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

if args.use_lr_schedule:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    current_setting.append('ReduceLROnPlateau')
else:
    current_setting.append('None')

current_setting.append(str(config))

l2_loss = torch.nn.MSELoss()
l1_loss = torch.nn.L1Loss()

model = model.to(device)
l2_loss = l2_loss.to(device)
l1_loss = l1_loss.to(device)

viz = Visdom()
viz.line([[0., 0.]], [0], win='baseline_train', opts=dict(title='train&valid loss', legend=["train loss", "valid loss"]))

def train_epoch(model, iterator, optimizer, criterion1, criterion2, MMDLoss, config, clip_value):
    model.train()
    epoch_loss = 0
    for batch in iterator:
        text, audio, vision, label = batch
        label = label.squeeze(1)

        optimizer.zero_grad()

        zl, za, zv, zy, x_l_hat, x_a_hat, x_v_hat, y_hat = model(text, audio, vision)

        output = y_hat.squeeze(1)

        mmd_loss = config['lambda_mmd'] * (MMDLoss(zl) + MMDLoss(za) + MMDLoss(zv) + MMDLoss(zy))
        gen_loss = config['lambda_xl'] * criterion2(x_l_hat, text) + config['lambda_xa'] * criterion2(x_a_hat, audio) + \
                   config['lambda_xv'] * criterion2(x_v_hat, vision)
        prediction_loss = criterion1(output, label)
        # if criterion2(x_v_hat, vision).item() > 1000000:
        #     print(x_v_hat)
        #     print(vision)
        #     assert False
        # print(criterion2(x_v_hat, vision))
        loss = mmd_loss + gen_loss + prediction_loss

        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def valid_epoch(model, iterator, criterion1, criterion2, MMDLoss, config):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in iterator:
            text, audio, vision, label = batch
            label = label.squeeze(1)
            zl, za, zv, zy, x_l_hat, x_a_hat, x_v_hat, y_hat = model(text, audio, vision)
            output = y_hat.squeeze(1)

            mmd_loss = config['lambda_mmd'] * (MMDLoss(zl) + MMDLoss(za) + MMDLoss(zv) + MMDLoss(zy))
            gen_loss = config['lambda_xl'] * criterion2(x_l_hat, text) + config['lambda_xa'] * criterion2(x_a_hat, audio) + \
                       config['lambda_xv'] * criterion2(x_v_hat, vision)
            prediction_loss = criterion1(output, label)
            loss = mmd_loss + gen_loss + prediction_loss
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def test_epoch(model, iterator):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in iterator:
            text, audio, vision, label = batch
            label = label.squeeze(1)

            zl, za, zv, zy, x_l_hat, x_a_hat, x_v_hat, y_hat = model(text, audio, vision)
            outputs = y_hat.squeeze(1)

            logits = outputs.detach().cpu().numpy()
            label_ids = label.detach().cpu().numpy()

            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()

            preds.extend(logits)
            labels.extend(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)

    return preds, labels

def test_score(model, iterator, use_zero=False):

    preds, y_test = test_epoch(model, iterator)
    non_zeros = np.array([i for i, e in enumerate(y_test) if e != 0 or use_zero])

    preds = preds[non_zeros]
    y_test = y_test[non_zeros]

    mae = np.round(np.mean(np.absolute(preds - y_test)), decimals=3)
    corr = np.round(np.corrcoef(preds, y_test)[0][1], decimals=3)

    preds = preds >= 0
    y_test = y_test >= 0

    f_score = np.round(f1_score(y_test, preds, average="weighted"), decimals=4)
    acc = np.round(accuracy_score(y_test, preds), decimals=4)

    return acc, f_score, mae, corr

max_valid_loss = 999999

for epoch in trange(args.num_epoch):
    start_time = time.time()
    train_loss = train_epoch(model, train_loader, optimizer, l1_loss, l2_loss, loss_MMD, config, args.clip)
    valid_loss = valid_epoch(model, valid_loader, l1_loss, l2_loss, loss_MMD, config)
    viz.line([[train_loss, valid_loss]], [epoch], win="baseline_train", update="append")
    end_time = time.time()
    epoch_mins, epoch_secs = interval_time(start_time, end_time)
    print("Epoch: {} | Train Loss: {} | Validation Loss: {} | Time: {} min {} sec".format(epoch + 1, train_loss, valid_loss, epoch_mins, epoch_secs))

    if valid_loss < max_valid_loss:
        max_valid_loss = valid_loss
        print('Saving the model ...')
        torch.save(model, 'saved_models/MOSI/baseline_model.pth')
        print("Saved the model to saved_models/MOSI/baseline_model.pth !")

    if args.use_lr_schedule:
        scheduler.step(valid_loss)

model = torch.load('saved_models/MOSI/baseline_model.pth')

test_acc, test_f_score, test_mae, test_corr = test_score(model, test_loader)
print("\033[1;31mAccuracy: {}, F1_score: {}, MAE: {}, Corr: {}\033[0m".format(test_acc, test_f_score, test_mae, test_corr))

print("Time Usage: {} minutes {} seconds".format(*interval_time(start, time.time())))

saved_results_path = "saved_results/results.txt"
results = ["Baseline using MulT and generative model:\n",
           "Hyperparameters: seed {}, dataset {}, num_epoch {}, batch_size {}, learning_rate {}, clip {}, "
           "num_layers {}, num_heads {}, lr_schedule {},\nmodel_config {}\n".format(*current_setting),
           "Acc2: {:5.4f}\n".format(test_acc), "F1 score: {:5.4f}\n".format(test_f_score), "MAE: {:5.3f}\n".format(test_mae),
           "Corr: {:5.3f}\n".format(test_corr), "-"*100+"\n"]
write_txt(saved_results_path, results)

print("="*150)