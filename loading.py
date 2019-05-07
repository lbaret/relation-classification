import os
import re
import nltk

"""
    Avec ce dataset, nous prenons un vecteur indiquant a quelle indice sont les deux entités
"""
class LoadData:

    tokens = []
    pos = []
    entities = []
    tokens_embeddings = []
    pos_embeddings = []
    entities_tensor = None
    lengths = []
    max_len = None
    y_train = []
    words_vocab_size = None
    pos_vocab_size = None
    test = []


    def __init__(self, file_path):

        patterns = [(['<', 'e1', '>'], ['<', '/e1', '>']), (['<', 'e2', '>'], ['<', '/e2', '>'])]
        # Partie extraction du fichier avec loading des datas
        f = open(file_path)
        text = f.read()
        splitted = text.split('\n\n')
        del splitted[8000]
        for ind, part in enumerate(splitted):
            part = part.split('\n')
            part[0] = re.sub(r'^\d+\t', '', part[0])
            part[0] = part[0].replace('\"', '')
            part[1] = re.sub(r'\(.+\)', '', part[1])
            del part[2]
            splitted[ind] = part

        # Prétraitement des données
        relations = set([elem[1] for elem in splitted])
        relation_index = {}
        for ind, relation in enumerate(relations):
            relation_index[relation] = ind

        tag_index = self.createTagVocab()

        tokens = []
        all_words = []
        for data in splitted:
            words = nltk.word_tokenize(data[0])
            all_words += words
            tokens.append(words)
            self.y_train.append(self.convertToIndex(data[1], relation_index))

        word_index = self.createWordVocab(all_words)
        for ind, t in enumerate(tokens):
            # Recherche des entités
            find = [None, None, None, None]
            find[0] = self.findSublist(t, patterns[0][0])
            find[1] = self.findSublist(t, patterns[0][1])
            find[2] = self.findSublist(t, patterns[1][0])
            find[3] = self.findSublist(t, patterns[1][1])
            self.entities.append([find[0][0], find[1][0]])
            find = [j for i in find for j in i]
            # Tri sens inverse important pour conserver les index trouvés
            find.sort(reverse=True)
            for index in find:
                del t[index]

            pos_tags = [tag[1] for tag in nltk.pos_tag(t)]
            tags = self.convertToIndex(pos_tags, tag_index)
            self.pos.append(tags)

            # Vecteur mots index
            self.tokens.append(self.convertToIndex(t, word_index))
            self.lengths.append(len(t))

        # Il ne reste plus qu'à Padder les vecteurs et tout est pret !!!
        self.max_len = max(self.lengths)
        self.padLists(self.tokens, self.pos, self.max_len)

        self.test = tokens

    def convertToIndex(self, element, dict_index):
        if isinstance(element, str):
            return dict_index[element]
        else:
            the_list = element.copy()
            for i in range(len(the_list)):
                if the_list[i] in dict_index:
                    the_list[i] = dict_index[the_list[i]]
                else:
                    the_list[i] = dict_index['UNK']
            return the_list

    def createTagVocab(self):
        tagdict = nltk.data.load('help/tagsets/upenn_tagset.pickle')
        list_tag = list(tagdict)
        tag_index = {}
        for ind, tag in enumerate(list_tag):
            tag_index[tag] = ind + 1
        tag_index["UNK"] = 0
        self.pos_vocab_size = len(tag_index)
        return tag_index

    def createWordVocab(self, words):
        uniques = set(words)
        word_index = {}
        word_index['UNK'] = 0
        for ind, w in enumerate(uniques):
            word_index[w] = ind + 1
        self.words_vocab_size = len(word_index)
        return word_index

    def getVocabSizes(self):
        return self.words_vocab_size, self.pos_vocab_size

    def findSublist(self, the_list, sub_list):
        if len(sub_list) > 1:
            for i in range(0, len(the_list) - len(sub_list)):
                if the_list[i] == sub_list[0]:
                    for k in range(1, len(sub_list)):
                        if the_list[i + k] != sub_list[k]:
                            break
                        if k == len(sub_list) - 1:
                            if sub_list[k] == the_list[i + k]:
                                return list(range(i, i + k + 1))
        else:
            if sub_list[0] in the_list:
                return [the_list.index(sub_list[0])]
        return None

    def padLists(self, tokens_train, pos_train, max_len=300):
        # On met tous les mots a la meme dimension
        for i in range(len(tokens_train)):
            for j in range(max_len - len(tokens_train[i])):
                tokens_train[i].append(0)
                pos_train[i].append(0)

    def getData(self):
        return self.tokens, self.pos, self.entities, self.lengths, self.y_train

    # def __getitem__(self, key):
    #     if key == "index":
    #         return self.tokens, self.pos, self.entities, self.lengths, self.y_train
    #     elif key == "embeddings":
    #         return self.tokens_embeddings, self.pos_embeddings, self.lengths, self.y_train
    #     else:
    #         return None