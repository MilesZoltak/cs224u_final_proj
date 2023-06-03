from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from transformers import BertModel, BertTokenizer

class Processing():
    def __init__(self):
        # self.tokenizer = tokenizer
        self.tokenizer = get_tokenizer('basic_english')
        self.bert_tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-mini")

    def yield_tokens(self, data_iter):
        for _, text in data_iter:
            # yield self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
            breakpoint()
            yield self.tokenizer(text)
    
    def build_vocab(self, iter, specials):
        print("1")
        max_length = 100  # Set the maximum length for your text samples

        # Preprocess the data and filter out samples longer than the maximum length
        iter = self.yield_tokens(iter)
        print("2")
        vocab = build_vocab_from_iterator(iter, specials=specials)
        print("3")
        vocab.set_default_index(vocab["<unk>"])
        print("4")
        return vocab