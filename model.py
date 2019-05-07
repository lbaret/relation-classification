import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence
import torch.nn.functional as F

class BiRNN(nn.Module):

    def __init__(self, num_classes, num_layers, hidden_dim, word_embedding_dim, pos_embedding_dim, entity_dim,
                 batch_size):
        super(BiRNN, self).__init__()

        # Instanciation de notre couche RNN > LSTM dans notre cas
        self.lstm = nn.LSTM(word_embedding_dim, hidden_dim, num_layers, dropout=0.5, bidirectional=True,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dims = [word_embedding_dim, pos_embedding_dim, entity_dim]
        self.bs = batch_size
        self.hidden_dim = hidden_dim
        self.softmax = nn.LogSoftmax()
        self.conv = nn.Conv1d(in_channels=pos_embedding_dim, out_channels=hidden_dim, kernel_size=3, padding=1)

    def init_hidden(self):
        h0 = Variable(torch.zeros(2 * 2, self.bs, self.hidden_dim, device=torch.device("cuda")))
        c0 = Variable(torch.zeros(2 * 2, self.bs, self.hidden_dim, device=torch.device("cuda")))
        return h0, c0

    def forward(self, x):
        # ICI SPLIT LE X EN 3 -> Words / Tags / Entities
        words, tags, entities = x
        # Mécanisme attentionnel
        # Passage dans le réseau Words
        self.hidden = self.init_hidden()
        lstm_out, self.hidden = self.lstm(words, self.hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        lstm_out = F.tanh(lstm_out)
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2), return_indices=False)
        lstm_out = lstm_out.squeeze(2)

        # Passage dans le réseau Tags
        tags = torch.transpose(tags, 1, 2)
        cnn_out = self.conv(tags)
        cnn_out = F.max_pool1d(cnn_out, cnn_out.size(2), return_indices=False)
        cnn_out = cnn_out.squeeze(2)

        # Concaténation des tensors de sorties
        add_out = torch.add(lstm_out, value=0.7, cnn_out)

        # On detruit la dernier dimension
        lstm_feats = self.fc(add_out)
        output = self.softmax(lstm_feats)

        return output