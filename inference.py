import torch 
from torch import nn 
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
from typing import Union, List, Tuple, Dict, Any
from tqdm import tqdm
import numpy as np

from transformers import BertTokenizer, BertModel

import os
import argparse

class nluDataset(Dataset):
    def __init__(self, data: List):
        self.data_items = data 
    
    def __getitem__(self, index) -> Any:
        return self.data_items[index]

    def __len__(self):
        return len(self.data_items)


def prepare_train_val_test(dir: str="./", val_ratio: float=0.05) \
    -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str, int]], List[Tuple[str, str, int]]]:
    label2id = {"entailment": 1, "contradiction": 2, "neutral": 0};
    train_val = []
    with open(os.path.join(dir, "train.tsv"), "r", encoding="utf8") as f:
        for line in tqdm(f.readlines()[1:]):
            parts = line.strip().split('\t')
            assert (len(parts) == 4)
            train_val.append((parts[1], parts[2], label2id[parts[3]]))
    test = []
    with open(os.path.join(dir, "test.tsv"), "r", encoding="utf8") as f:
        for line in tqdm(f.readlines()[1:]):
            parts = line.strip().split('\t')
            assert(len(parts) == 3)
            test.append((parts[1], parts[2], -1))
    num_train = int(len(train_val) * (1.0 - val_ratio))
    train = train_val[:num_train]
    val = train_val[num_train:]
    return train, val, test 


class LinearProbeLayer(nn.Module):
    def __init__(self, hidden_dim: int=768, num_heads=12, k_class=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.k_class = k_class
        # self.attn_nxn = nn.MultiheadAttention(
        #     embed_dim=hidden_dim,
        #     num_heads=num_heads,
        #     dropout=0.1,
        #     batch_first=True
        # )
        self.layernorm = nn.LayerNorm((hidden_dim,))
        self.attn_nx1 = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_dim, k_class)
        self.softmax = nn.Softmax(dim=-1)
        self.act = nn.SiLU()
        
    def forward(self, x, lengths):
        shape = x.shape # B x S x E
        # attn_nxn_mask = torch.ones((shape[0] * self.num_heads, max(lengths), max(lengths))).bool()
        # for i, length in enumerate(lengths):
        #     attn_nxn_mask[i * self.num_heads : i * self.num_heads + self.num_heads, :length, :length] = False
        # x, _ = self.attn_nxn(x, x, x)
        # x = self.layernorm(x)
        # x = self.act(x)
        attn_nx1_mask = torch.ones((shape[0] * self.num_heads, 1, max(lengths))).bool()
        for i, length in enumerate(lengths):
            attn_nx1_mask[i * self.num_heads : i * self.num_heads + self.num_heads, :, :length] = False 
        x, _ = self.attn_nx1(x[:, :1, :], x, x)
        x = self.layernorm(x)
        x = self.act(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x


class WarmUpScheduler(LambdaLR):
    def __init__(self, optimizer, warmup, beta):
        self.optimizer = optimizer
        self.warmup = warmup
        self.beta = beta
        def lr_lambda(epoch):
            return min(1., (epoch + 1) / self.warmup) * np.power(self.beta, epoch + 1 - warmup)
        super().__init__(optimizer=optimizer, lr_lambda=lr_lambda)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--exp_id", type=int, default=0)
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--expid", type=str, required=True)
    parser.add_argument("--step", type=str, default='last')
    args = parser.parse_args()
    trainset, valset, testset = prepare_train_val_test()
    print(f"train: {len(trainset)}, val: {len(valset)}, test: {len(testset)}")
    # print(trainset[:10])
    device = torch.device("cuda:0")
    pretrained_model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
    bertmodel = BertModel.from_pretrained(pretrained_model_name)
    bertmodel.load_state_dict(torch.load(f"bertmodel-{args.expid}-{args.step}.pth")["model"])
    bertmodel = bertmodel.cuda(device)
    
    tokenizer: BertTokenizer 
    bertmodel: BertModel

    data_item = trainset[0]
    print(data_item[0], '\n', data_item[1])
    encoded_input = tokenizer(text=[trainset[0][0], trainset[1][0]], text_pair=[trainset[0][1], trainset[1][1]], return_tensors="pt", padding=True)
    print(encoded_input)
    for key in encoded_input.keys():
        encoded_input[key] = encoded_input[key].to(device)
    output = bertmodel(**encoded_input)
    h = output.last_hidden_state
    print(h.shape)
    
    batchsize = 48

    train_loader = DataLoader(nluDataset(trainset), batch_size=batchsize, shuffle=False) # Not Shuffle Here
    val_loader = DataLoader(nluDataset(valset), batch_size=batchsize, shuffle=False)
    test_loader = DataLoader(nluDataset(testset), batch_size=batchsize, shuffle=False)

    def cycle(loader):
        while True:
            for item in loader:
                yield item 
    
    # train_loader = cycle(train_loader)
    # data_piece = next(train_loader)

    def prepare_for_batch(data_piece):
        encoded = tokenizer(
            text=list(data_piece[0]),
            text_pair=list(data_piece[1]),
            return_tensors="pt",
            padding=True
        )
        lengths = encoded["attention_mask"].sum(dim=1).tolist()
        return encoded, lengths, data_piece[2]

    def half_data_piece(data_piece):
        text, text_pair, labels = list(data_piece[0]), list(data_piece[1]), data_piece[2].tolist()
        nums = len(text)
        # print(nums)
        half_nums = nums // 2
        assert (half_nums >= 1)
        return (tuple(text[:half_nums]), tuple(text_pair[:half_nums]), torch.tensor(labels[:half_nums])), (tuple(text[half_nums:]), tuple(text_pair[half_nums:]), torch.tensor(labels[half_nums:]))

    
    # batch = prepare_for_batch(data_piece)
    # print(batch)

    linear_probe = LinearProbeLayer()
    linear_probe.load_state_dict(torch.load(f"linear_probe_model-{args.expid}-{args.step}.pth")["model"])
    linear_probe = linear_probe.to(device)
    # optimizer = Adam(linear_probe.parameters(), lr=1e-3)
    # optimizer = Adam([
    #     {'params' : linear_probe.parameters(), 'lr': 1e-3},
    #     {'params' : bertmodel.parameters(), 'lr': 1e-5}
    # ])
    # # scheduler = ExponentialLR(optimizer, gamma=0.99997)
    # scheduler = WarmUpScheduler(optimizer=optimizer, warmup=2500, beta=0.99997)
    
    max_iters = 100000
    record_interval = 1000
    valid_interval = 1000
    save_interval = 1000
    # save_name = f"./linear_probe_model-{args.exp_id}"
    # save_name2 = f"./bertmodel-{args.exp_id}"
    iter_step = 0
    loss_record = []
    acc_record = []
    max_batch_len = 8192
    data_pieces_to_fetch = []
    top_5_steps = []
    count = 0
    id2label = {1:"entailment", 2:"contradiction", 0:"neutral"}
    valid_acc = []
    valid_loss = []
    for datapiece in tqdm(val_loader):
        batch, lengths, labels = prepare_for_batch(datapiece)
        labels = labels.to(device)
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        bertmodel.eval()
        with torch.no_grad():
            output = bertmodel(**batch)
        linear_probe.eval()
        with torch.no_grad():
            output = linear_probe.forward(output.last_hidden_state, lengths)
            output = torch.squeeze(output, dim=1)
            label_distribution = torch.zeros_like(output)
            for i in range(output.shape[0]):
                label_distribution[i][labels[i].item()] = 1.
            loss = -torch.mean(label_distribution * torch.log(output))
            valid_loss.append(loss.item())
            predict = torch.argmax(output, dim=1)
            acc = torch.sum(predict == labels) / labels.shape[0]
            valid_acc.append(acc.item())
    print(f"valid_loss_avg: {sum(valid_loss) / len(valid_loss) :.6f}, valid_acc_avg: {sum(valid_acc) / len(valid_acc) :.6f}")
    with open("output.csv", "w", encoding="utf8") as f:
        for datapiece in tqdm(test_loader):
            batch, lengths, _ = prepare_for_batch(datapiece)
            # print(batch, lengths)
            # labels = labels.to(device)
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            bertmodel.eval()
            with torch.no_grad():
                output = bertmodel(**batch)
            linear_probe.eval()
            with torch.no_grad():
                output = linear_probe.forward(output.last_hidden_state, lengths)
                output = torch.squeeze(output, dim=1)
                predict = torch.argmax(output, dim=1)
            predict = predict.tolist()
            for item in predict:
                count += 1
                f.write(f"{count},{id2label[item]}\n")

    train_acc = []
    with open(f"train_label-{args.id}.txt", "w", encoding="utf8") as f:
        item_idx = 0
        for datapiece in tqdm(train_loader):
            batch, lengths, labels = prepare_for_batch(datapiece)
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            bertmodel.eval()
            linear_probe.eval()
            with torch.no_grad():
                output = bertmodel(**batch)
                output = linear_probe.forward(output.last_hidden_state, lengths)
                output = torch.squeeze(output, dim=1)
                predict = torch.argmax(output, dim=1)
                acc = torch.sum(predict == labels) / labels.shape[0]
                train_acc.append(acc.item())
            output_list = output.tolist()
            for output_piece in output_list:
                item_idx += 1
                output_string = '\t'.join(output_piece)
                f.write(f"{item_idx}\t{output_string}\n")

    print(f"train_acc_avg: {sum(train_acc) / len(train_acc) :.6f}")

