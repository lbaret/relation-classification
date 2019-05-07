import loading as load
import torch
import torch.nn as nn
from model import BiRNN
from learning import train, DatasetSemEval

epochs = 250
lr = 0.01
# Prendre une batch size cohérente
batch_size = 64
NUM_CLASSES = 10
NUM_LAYERS = 2
HIDDEN_DIM = 512
WORDS_EMBEDDING_DIM = 64
TAGS_EMBEDDING_DIM = 16
ENTITY_DIM = 2

data_file = "/train_semeval_2010/TRAIN_FILE.TXT"
data_load = load.LoadData(data_file)
words, tags, entities, lengths, y_train = data_load.getData()

# Tensors
words_tensor = torch.LongTensor(words)
tags_tensor = torch.LongTensor(tags)
entities_tensor = torch.FloatTensor(entities)
target_tensor = torch.LongTensor(y_train)
lengths_tensor = torch.FloatTensor(lengths)

# Embeddings
words_vocab_size, tags_vocab_size = data_load.getVocabSizes()
words_embedding = nn.Embedding(words_vocab_size, WORDS_EMBEDDING_DIM)
tags_embedding = nn.Embedding(tags_vocab_size, TAGS_EMBEDDING_DIM)
words_embeds = words_embedding(words_tensor)
tags_embeds = tags_embedding(tags_tensor)

# Reshape pour concat
words_size = words_embeds.size()
tags_size = tags_embeds.size()

# Création du DataLoader
# train_tensor = torch.cat((words_embeds, tags_embeds, entities_tensor), dim=1)
dataset = DatasetSemEval(words_embeds, tags_embeds, entities_tensor, target_tensor, lengths_tensor)
# dataloader = DataLoader(dataset, batch_size=batch_size)

# Déclaration de notre modèle
model = BiRNN(NUM_CLASSES, NUM_LAYERS, HIDDEN_DIM, WORDS_EMBEDDING_DIM, TAGS_EMBEDDING_DIM, ENTITY_DIM, batch_size).cuda()
history = train(model, dataset, epochs, batch_size, lr, use_gpu=True)
history.display()

