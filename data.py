import torch
from torchtext.datasets import AG_NEWS
from torch.utils.data import DataLoader
from processing import Processing
from transformers import BertModel, BertTokenizer

class Data():
    def __init__(self, batch_size=8, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_iter, self.test_iter = self.get_AG_train_test()
        processing = Processing()
        # self.vocab = processing.build_vocab(self.train_iter, ["<unk>"])
        # self.dataloader = DataLoader(self.train_iter, batch_size=self.batch_size, shuffle=self.shuffle)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_AG_train_test(self):
        train_iter = AG_NEWS(split="train")
        test_iter = AG_NEWS(split="test")

        return train_iter, test_iter
    
    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")