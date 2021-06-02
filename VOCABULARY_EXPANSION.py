
import pickle
from time import time
from random import shuffle
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from SKIP_THOUGHTS_ARCH import SkipThoughtsNN

def LoadModelsAndUtils ( loc = 'MODELS' ) :
    global ST_MODEL, WORD_2_VEC_MODEL, WORD_TO_ID, ID_TO_WORD
    ST_MODEL = torch.load(loc+'/SKIP_THOUGHTS_NN')
    ST_MODEL.eval()
    
    limit = int(1e7) # 10 Million words
    WORD_2_VEC_MODEL = KeyedVectors.load_word2vec_format(loc+'/GoogleNews-vectors-negative300.bin.gz', binary = True, limit = limit)
    
    file = open(loc+'/UTILS/SKIP_THOUGHTS_NN_WORD2ID', 'rb')
    WORD_TO_ID = pickle.load(file)
    file.close()

    file = open(loc+'/UTILS/SKIP_THOUGHTS_NN_ID2WORD', 'rb')
    ID_TO_WORD = pickle.load(file)
    file.close()

def ExtractVocabularies ( ) :
    global W2V_VOCAB, ST_VOCAB, VE_VOCAB
    W2V_VOCAB = set(list(WORD_2_VEC_MODEL.vocab))
    ST_VOCAB = set(ID_TO_WORD)
    VE_VOCAB = list(ST_VOCAB.intersection(W2V_VOCAB))
    VE_VOCAB.sort()

def SaveVocabularies ( loc = 'MODELS' ) :
    file = open(loc+'/UTILS/VOCAB_EXPANSION_NN_ST_VOCAB', 'wb')
    pickle.dump(ST_VOCAB, file)
    file.close()

    file = open(loc+'/UTILS/VOCAB_EXPANSION_NN_W2V_VOCAB', 'wb')
    pickle.dump(W2V_VOCAB, file)
    file.close()

def GetSTModelWordEmbeddingSize ( ) :
    embedding_layer = list(ST_MODEL.modules())[1]
    embedding_layer_decription = str(embedding_layer)
    return eval(embedding_layer_decription.split(', ')[-1][:-1])

def GetSTModelWordEmbedding ( word ) :
    word_id = WORD_TO_ID[word]
    word_id = torch.tensor([word_id]).long()
    embedding_layer = list(ST_MODEL.modules())[1]
    return embedding_layer(word_id).detach()[0]

def GetW2VModelWordEmbedding ( word ) :
    options = [word, word[0].upper()+word[1:], word.upper()]
    for w in options :
        try :
            vec = WORD_2_VEC_MODEL[w].copy()
            vec = torch.tensor(vec)
            return vec
        except : continue

def PrepareVocabularyExpansionDataset ( ) :
    global DATASET
    inputs = [ GetW2VModelWordEmbedding(w) for w in VE_VOCAB ]
    outputs = [ GetSTModelWordEmbedding(w) for w in VE_VOCAB ]
    DATASET = list(zip(inputs, outputs))

def MakeBatches ( batch_size = 64 ) :
    shuffle(DATASET)
    batches = list()
    for start in range(0, len(DATASET), batch_size) :
        end = start + batch_size
        inputs, outputs = list(zip(*DATASET[start:end]))
        inputs, outputs = torch.stack(inputs), torch.stack(outputs)
        batches.append((inputs, outputs))
    return batches

class VocabularyExpansionNN ( nn.Module ) :
    def __init__ ( self , input_dim , output_dim ) :
        super(VocabularyExpansionNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dense = nn.Linear ( input_dim, output_dim, bias = True )

    def forward ( self , inputs ) :
        return self.dense(inputs)

def TrainModel ( total_epochs , learning_rate = 0.001 , batch_size = 64 ) :
    optimizer = torch.optim.Adam(VE_MODEL.parameters(), lr = learning_rate)
    loss_fn = nn.MSELoss()
    
    for epoch in range(total_epochs) :
        total_loss = 0.0
        batches = MakeBatches(batch_size)
        epoch_start_time = time()
        
        for step, (inputs, true) in enumerate(batches) :
            pred = VE_MODEL(inputs)
            loss = loss_fn(pred, true)
            
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = total_loss / len(batches)
        ti = time() - epoch_start_time
        print(' EPOCH {:3d} MEAN LOSS : {:.6f} | DUR : {:.4f}'.format(epoch+1, loss, ti))

    torch.save(VE_MODEL, 'MODELS/VOCABULARY_EXPANSION_NN')

if __name__ == '__main__' :
    LoadModelsAndUtils()
    ExtractVocabularies()
    SaveVocabularies()
    PrepareVocabularyExpansionDataset()

    INPUT_DIMENSION = 300
    OUTPUT_DIMENSION = GetSTModelWordEmbeddingSize()
    VE_MODEL = VocabularyExpansionNN(INPUT_DIMENSION, OUTPUT_DIMENSION)
    # VE_MODEL = torch.load('MODELS/VOCABULARY_EXPANSION_NN') # to resume training

    BATCH_SIZE = 64
    TOTAL_EPOCHS = 500
    LEARNING_RATE = 0.001

    VE_MODEL.train()
    TrainModel(TOTAL_EPOCHS, LEARNING_RATE, BATCH_SIZE)
