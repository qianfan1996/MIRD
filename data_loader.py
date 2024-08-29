# -*-coding:utf-8-*-
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader
from transformers import BertTokenizer
from utils import get_device


# load CMU-MOSI or CMU-MOSEI dataset with raw text, 33-dim audio and 709-dim visual feature
def load_pkl(file_path, mode='train'):
    with open(file_path, 'rb') as file:
        info = pickle.load(file)
        raw_text = list(info[mode]['raw_text'])
        audio = info[mode]['audio']
        video = info[mode]['video']
        label = info[mode]['labels']
    return raw_text, audio, video, label


def get_dataset(examples, max_words_length, tokenizer):
    words, acoustic, visual, label = examples

    dic = tokenizer(words, padding='max_length', truncation=True, max_length=max_words_length, return_tensors='pt')
    input_ids = dic['input_ids']
    input_mask = dic['attention_mask']
    segment_ids = dic['token_type_ids']
    visual = torch.tensor(visual, dtype=torch.float)
    acoustic = torch.tensor(acoustic, dtype=torch.float)
    label = torch.tensor(np.array([[label]]), dtype=torch.float).transpose(0, 2)

    dataset = TensorDataset(
        input_ids,
        visual,
        acoustic,
        input_mask,
        segment_ids,
        label,
    )

    return dataset


def set_up_data_loader(dataset="mosi", max_words_length=50, train_batch_size=48, valid_batch_size=128, test_batch_size=128, num_epoch=100):
    train_data = load_pkl("data/{}/{}.pkl".format(dataset, dataset), 'train')
    valid_data = load_pkl("data/{}/{}.pkl".format(dataset, dataset), 'valid')
    test_data = load_pkl("data/{}/{}.pkl".format(dataset, dataset), 'test')

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = get_dataset(train_data, max_words_length, tokenizer)
    valid_dataset = get_dataset(valid_data, max_words_length, tokenizer)
    test_dataset = get_dataset(test_data, max_words_length, tokenizer)

    num_train_optimization_steps = (
            int(
                len(train_dataset) / train_batch_size
            ) * num_epoch
    )  # number of training iterations

    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True
    )

    valid_dataloader = DataLoader(
        valid_dataset, batch_size=valid_batch_size, shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=True,
    )

    return (
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        num_train_optimization_steps,
    )


# load EmoVoxCeleb dataset with raw text
def load_emovoxceleb_pkl(file_path):
    with open(file_path, 'rb') as file:
        info = pickle.load(file)
        raw_text = list(info['raw_text'])
        audio = info['audio']
        video = info['video']
    return raw_text, audio, video


class TrainDataset(Dataset):
    def __init__(self, labeled_data, unlabeled_data, device, split_rate=103):
        self.input_ids_l, self.input_mask_l, self.segment_ids_l, self.audio_l, self.vision_l, self.label = labeled_data
        self.input_ids_u, self.input_mask_u, self.segment_ids_u, self.audio_u, self.vision_u = unlabeled_data

        self.device = device
        self.split_rate = split_rate

        len_labeled = len(self.input_ids_l)
        len_unlabeled = int(split_rate * len_labeled)
        self.len_labeled = len_labeled

        self.input_ids_u = self.input_ids_u[0:len_unlabeled]
        self.input_mask_u = self.input_mask_u[0:len_unlabeled]
        self.segment_ids_u = self.segment_ids_u[0:len_unlabeled]
        self.audio_u = self.audio_u[0:len_unlabeled]
        self.vision_u = self.vision_u[0:len_unlabeled]

    def __getitem__(self, index):
        input_ids_l = self.input_ids_l[index].to(self.device)
        input_mask_l = self.input_mask_l[index].to(self.device)
        segment_ids_l = self.segment_ids_l[index].to(self.device)
        audio_l = torch.FloatTensor(self.audio_l[index]).to(self.device)
        vision_l = torch.FloatTensor(self.vision_l[index]).to(self.device)
        label = torch.FloatTensor(self.label[index]).to(self.device)
        labeled_data = (input_ids_l, input_mask_l, segment_ids_l, audio_l, vision_l, label)

        input_ids_u, input_mask_u, segment_ids_u, audio_u, vision_u = [], [], [], [], []
        for idx in range(self.split_rate):
            input_ids_u.append(self.input_ids_u[index + idx*self.len_labeled])
            input_mask_u.append(self.input_mask_u[index + idx*self.len_labeled])
            segment_ids_u.append(self.segment_ids_u[index + idx*self.len_labeled])
            audio_u.append(self.audio_u[index + idx*self.len_labeled])
            vision_u.append(self.vision_u[index + idx*self.len_labeled])

        input_ids_u = torch.stack(input_ids_u, dim=0).to(self.device)
        input_mask_u = torch.stack(input_mask_u, dim=0).to(self.device)
        segment_ids_u = torch.stack(segment_ids_u, dim=0).to(self.device)
        audio_u = torch.FloatTensor(torch.stack(audio_u, dim=0)).to(self.device)
        vision_u = torch.FloatTensor(torch.stack(vision_u, dim=0)).to(self.device)
        unlabeled_data = (input_ids_u, input_mask_u, segment_ids_u, audio_u, vision_u)

        return labeled_data, unlabeled_data

    def __len__(self):
        return int(self.len_labeled)


class ValidTestDataset(Dataset):
    def __init__(self, labeled_data, device):
        self.input_ids, self.input_mask, self.segment_ids, self.audio, self.vision, self.label = labeled_data
        self.device = device
        self.len_labeled = len(self.input_ids)

    def __getitem__(self, index):
        input_ids = self.input_ids[index].to(self.device)
        input_mask = self.input_mask[index].to(self.device)
        segment_ids = self.segment_ids[index].to(self.device)
        audio = torch.FloatTensor(self.audio[index]).to(self.device)
        vision = torch.FloatTensor(self.vision[index]).to(self.device)
        label = torch.FloatTensor(self.label[index]).to(self.device)
        return input_ids, input_mask, segment_ids, audio, vision, label

    def __len__(self):
        return int(self.len_labeled)



def data_loader_sl(dataset="mosi", max_words_length=50, train_batch_size=48, valid_batch_size=128, test_batch_size=128, num_epoch=100, split_rate=103):
    train_text, train_acoustic, train_visual, train_label = load_pkl("data/{}/{}.pkl".format(dataset, dataset), 'train')
    valid_text, valid_acoustic, valid_visual, valid_label = load_pkl("data/{}/{}.pkl".format(dataset, dataset), 'valid')
    test_text, test_acoustic, test_visual, test_label = load_pkl("data/{}/{}.pkl".format(dataset, dataset), 'test')

    text, acoustic, visual = load_emovoxceleb_pkl("data/emovoxceleb/emovoxceleb.pkl")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_dic = tokenizer(train_text, padding='max_length', truncation=True, max_length=max_words_length, return_tensors='pt')
    train_input_ids = train_dic['input_ids']
    train_input_mask = train_dic['attention_mask']
    train_segment_ids = train_dic['token_type_ids']
    train_visual = torch.tensor(train_visual, dtype=torch.float)
    train_acoustic = torch.tensor(train_acoustic, dtype=torch.float)
    train_label = torch.tensor(np.array([[train_label]]), dtype=torch.float).transpose(0, 2)

    valid_dic = tokenizer(valid_text, padding='max_length', truncation=True, max_length=max_words_length, return_tensors='pt')
    valid_input_ids = valid_dic['input_ids']
    valid_input_mask = valid_dic['attention_mask']
    valid_segment_ids = valid_dic['token_type_ids']
    valid_visual = torch.tensor(valid_visual, dtype=torch.float)
    valid_acoustic = torch.tensor(valid_acoustic, dtype=torch.float)
    valid_label = torch.tensor(np.array([[valid_label]]), dtype=torch.float).transpose(0, 2)

    test_dic = tokenizer(test_text, padding='max_length', truncation=True, max_length=max_words_length, return_tensors='pt')
    test_input_ids = test_dic['input_ids']
    test_input_mask = test_dic['attention_mask']
    test_segment_ids = test_dic['token_type_ids']
    test_visual = torch.tensor(test_visual, dtype=torch.float)
    test_acoustic = torch.tensor(test_acoustic, dtype=torch.float)
    test_label = torch.tensor(np.array([[test_label]]), dtype=torch.float).transpose(0, 2)

    dic = tokenizer(text, padding='max_length', truncation=True, max_length=max_words_length, return_tensors='pt')
    input_ids = dic['input_ids']
    input_mask = dic['attention_mask']
    segment_ids = dic['token_type_ids']
    visual = torch.tensor(visual, dtype=torch.float)
    acoustic = torch.tensor(acoustic, dtype=torch.float)

    train_data_labeled = (train_input_ids, train_input_mask, train_segment_ids, train_acoustic, train_visual, train_label)
    valid_data_labeled = (valid_input_ids, valid_input_mask, valid_segment_ids, valid_acoustic, valid_visual, valid_label)
    test_data_labeled = (test_input_ids, test_input_mask, test_segment_ids, test_acoustic, test_visual, test_label)
    train_data_unlabeled = (input_ids, input_mask, segment_ids, acoustic, visual)

    train_dataset = TrainDataset(train_data_labeled, train_data_unlabeled, get_device(), split_rate)
    valid_dataset = ValidTestDataset(valid_data_labeled, get_device())
    test_dataset = ValidTestDataset(test_data_labeled, get_device())

    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=valid_batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=True
    )

    num_train_optimization_steps = (
            int(
                len(train_dataset) / train_batch_size
            ) * num_epoch
    )  # number of training iterations

    return (
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        num_train_optimization_steps,
    )



if __name__ == "__main__":
    data_path = "./data/mosi/mosi.pkl"
    text, audio, vision, label = load_pkl(data_path)
    print(type(text), type(audio), type(vision), type(label))
    print(len(text), audio.shape, vision.shape, len(label))

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dic = tokenizer(text, padding='max_length', truncation=True, max_length=50, return_tensors='pt')
    input_ids = dic['input_ids']
    print(input_ids.size())
    # =================================================================