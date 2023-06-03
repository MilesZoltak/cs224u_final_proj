from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
import torch
from model import TextClassificationModel
from data import Data
from torch.utils.data import DataLoader
from train import train, evaluate
import time
from processing import Processing

def main():
    print("Training on device: ", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # Hyperparameters
    EPOCHS = 10 # epoch
    LR = 5  # learning rate
    BATCH_SIZE = 64 # batch size for training
    BATCH_SIZE = 4 # batch size for training

    # data stuff!
    data = Data()
    train_iter, test_iter = data.get_AG_train_test()
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)
    num_train = int(len(train_dataset) * 0.95)
    split_train_, split_valid_ = \
        random_split(train_dataset, [num_train, len(train_dataset) - num_train])
    train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                                shuffle=True)
    valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                                shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                shuffle=True)

    # get the model
    num_class = len(set([label for (label, text) in train_iter]))
    model = TextClassificationModel(num_class)

    # create stuff we'll need for training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None


    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        # breakpoint()
        train(train_dataloader, optimizer, model, criterion, epoch)
        accu_val = evaluate(valid_dataloader, model, criterion)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
            'valid accuracy {:8.3f} '.format(epoch,
                                            time.time() - epoch_start_time,
                                            accu_val))
        print('-' * 59)

if __name__ == "__main__":
    main()