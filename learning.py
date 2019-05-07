import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn.utils.rnn import pack_padded_sequence
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import math
import numpy as np

class DatasetSemEval(Dataset):

    def __init__(self, x_train, x_tags_train, x_entities_train, y_train, lengths):
        self.x_train = x_train
        self.y_train = y_train
        self.x_tags_train = x_tags_train
        self.x_entities_train = x_entities_train
        self.lengths = lengths

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, index):
        data = self.x_train[index, 0:]
        data_tags = self.x_tags_train[index, 0:]
        data_entities = self.x_entities_train[index, 0:]
        target = self.y_train[index]
        lengths = self.lengths[index]
        return data, data_tags, data_entities, target, lengths


class History:

    def __init__(self):
        self.history = {
            'train_acc': [],
            'val_acc': [],
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }

    def save(self, train_acc, val_acc, train_loss, val_loss, lr):
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['lr'].append(lr)

    def display_accuracy(self):
        epoch = len(self.history['train_acc'])
        epochs = [x for x in range(1, epoch + 1)]
        plt.title('Training accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.plot(epochs, self.history['train_acc'], label='Train')
        plt.plot(epochs, self.history['val_acc'], label='Validation')
        plt.legend()
        plt.show()

    def display_loss(self):
        epoch = len(self.history['train_acc'])
        epochs = [x for x in range(1, epoch + 1)]
        plt.title('Training loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.plot(epochs, self.history['train_loss'], label='Train')
        plt.plot(epochs, self.history['val_loss'], label='Validation')
        plt.legend()
        plt.show()

    def display_lr(self):
        epoch = len(self.history['train_acc'])
        epochs = [x for x in range(1, epoch + 1)]
        plt.title('Learning rate')
        plt.xlabel('Epochs')
        plt.ylabel('Lr')
        plt.plot(epochs, self.history['lr'], label='Lr')
        plt.show()

    def display(self):
        epoch = len(self.history['train_acc'])
        epochs = [x for x in range(1, epoch + 1)]

        fig, axes = plt.subplots(3, 1)
        plt.tight_layout()

        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Accuracy')
        #         axes[0].set_ylabel('F1 score')
        axes[0].plot(epochs, self.history['train_acc'], label='Train')
        axes[0].plot(epochs, self.history['val_acc'], label='Validation')
        axes[0].legend()

        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Loss')
        axes[1].plot(epochs, self.history['train_loss'], label='Train')
        axes[1].plot(epochs, self.history['val_loss'], label='Validation')

        axes[2].set_xlabel('Epochs')
        axes[2].set_ylabel('Lr')
        axes[2].plot(epochs, self.history['lr'], label='Lr')

        plt.show()


def train(model, dataset, n_epoch, batch_size, learning_rate, use_gpu=False):
    history = History()

    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    train_loader, valid_loader = train_valid_loaders(dataset, batch_size=batch_size)

    for i in range(n_epoch):
        model.train()
        for j, batch in enumerate(train_loader):
            inputs_words, inputs_tags, inputs_entities, targets, lengths = batch

            # Ordonner par ordre decroissant
            le_zip = list(zip(inputs_words.tolist(), inputs_tags.tolist(), inputs_entities.tolist(), targets.tolist(),
                              lengths.tolist()))
            le_zip.sort(key=lambda x: x[4], reverse=True)
            inp, inp_tags, inp_entities, targs, lens = zip(*le_zip)
            inputs_words = torch.tensor(inp).cuda()
            inputs_tags = torch.tensor(inp_tags).cuda()
            inputs_entities = torch.tensor(inp_entities).cuda()
            targets = torch.tensor(targs, dtype=torch.long)
            lengths = torch.tensor(lens)
            inputs_words = pack_padded_sequence(inputs_words, lengths, batch_first=True)
            inputs = (inputs_words, inputs_tags, inputs_entities)

            # On envoit les données au GPU pour le traitement
            #       inputs = inputs.cuda()
            targets = targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward(retain_graph=True)
            optimizer.step()

        train_acc, train_loss = validate(model, train_loader, use_gpu)
        val_acc, val_loss = validate(model, valid_loader, use_gpu)
        #     train_f1 = validate(model, train_loader, use_gpu)
        #     val_f1 = validate(model, valid_loader, use_gpu)
        history.save(train_acc, val_acc, train_loss, val_loss, learning_rate)
        #     history.save(train_f1, val_f1, 0, 0, learning_rate)
        print('Epoch {} - Train acc: {:.2f} - Val acc: {:.2f} - Train loss: {:.2f} - Val loss: {:.2f}'.format(i + 1,
                                                                                                              train_acc,
                                                                                                              val_acc,
                                                                                                              train_loss,
                                                                                                              val_loss))
    #     print('Epoch {} - Train f1: {:.2f} - Val f1: {:.2f}'.format(i+1, train_f1, val_f1))

    return history


def validate(model, val_loader, use_gpu=False):
    true = []
    pred = []
    val_loss = []

    criterion = nn.NLLLoss()
    model.eval()

    for j, batch in enumerate(val_loader):
        inputs_words, inputs_tags, inputs_entities, targets, lengths = batch
        # Ordonner par ordre decroissant
        le_zip = list(zip(inputs_words.tolist(), inputs_tags.tolist(), inputs_entities.tolist(), targets.tolist(),
                          lengths.tolist()))
        le_zip.sort(key=lambda x: x[4], reverse=True)
        inp, inp_tags, inp_entities, targs, lens = zip(*le_zip)
        inputs_words = torch.tensor(inp).cuda()
        inputs_tags = torch.tensor(inp_tags).cuda()
        inputs_entities = torch.tensor(inp_entities).cuda()
        targets = torch.tensor(targs, dtype=torch.long)
        lengths = torch.tensor(lens)
        inputs_words = pack_padded_sequence(inputs_words, lengths, batch_first=True)
        inputs = (inputs_words, inputs_tags, inputs_entities)

        # On envoit les données au GPU pour le traitement
        #     inputs = inputs.cuda()
        targets = targets.cuda()

        output = model(inputs)

        predictions = output.max(dim=1)[1]

        val_loss.append(criterion(output, targets).item())
        true.extend(targets.data.cpu().numpy().tolist())
        pred.extend(predictions.data.cpu().numpy().tolist())

        #     return f1_score(true, pred, average="micro")
        return accuracy_score(true, pred) * 100, sum(val_loss) / len(val_loss)


def train_valid_loaders(dataset, batch_size, train_split=0.8, shuffle=True):
    num_data = len(dataset)
    indices = np.arange(num_data)

    if shuffle:
        np.random.shuffle(indices)

    split = math.floor(train_split * num_data)
    train_idx, valid_idx = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size, sampler=valid_sampler)

    return train_loader, valid_loader