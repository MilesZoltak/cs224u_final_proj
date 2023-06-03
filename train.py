import time
import data
import processing
from model import TextClassificationModel
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
import torch
from tqdm import tqdm

def train(dataloader, optimizer, model, criterion, epoch):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text) in tqdm(enumerate(dataloader), total=len(dataloader)):
        optimizer.zero_grad()
        predicted_label = model(text).squeeze()
        label = label.to(torch.float32)
        # breakpoint()
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(0) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                            total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader, model, criterion):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text) in tqdm(enumerate(dataloader), total=len(dataloader)):
            predicted_label = model(text).squeeze()
            label = label.to(torch.float32)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count
