
import pickle
from time import time
from random import shuffle
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import KeyedVectors
from SKIP_THOUGHTS_ARCH import SkipThoughtsNN
import VOCABULARY_EXPANSION
from VOCABULARY_EXPANSION import VocabularyExpansionNN
from VOCABULARY_EXPANSION import GetSTModelWordEmbedding
from VOCABULARY_EXPANSION import GetW2VModelWordEmbedding

def LoadModelsAndUtils ( loc = 'MODELS' ) :
    global ST_MODEL, VE_MODEL, WORD_2_VEC_MODEL, WORD_TO_ID, ID_TO_WORD
    ST_MODEL = torch.load(loc+'/SKIP_THOUGHTS_NN')
    ST_MODEL.eval()
    VE_MODEL = torch.load(loc+'/VOCABULARY_EXPANSION_NN')
    VE_MODEL.eval()

    limit = int(1e7) # 10 Million words
    WORD_2_VEC_MODEL = KeyedVectors.load_word2vec_format(loc+'/GoogleNews-vectors-negative300.bin.gz', binary = True, limit = limit)

    file = open(loc+'/UTILS/SKIP_THOUGHTS_NN_WORD2ID', 'rb')
    WORD_TO_ID = pickle.load(file)
    file.close()

    file = open(loc+'/UTILS/SKIP_THOUGHTS_NN_ID2WORD', 'rb')
    ID_TO_WORD = pickle.load(file)
    file.close()

    global W2V_VOCAB, ST_VOCAB
    file = open(loc+'/UTILS/VOCAB_EXPANSION_NN_ST_VOCAB', 'rb')
    ST_VOCAB = pickle.load(file)
    file.close()

    file = open(loc+'/UTILS/VOCAB_EXPANSION_NN_W2V_VOCAB', 'rb')
    W2V_VOCAB = pickle.load(file)
    file.close()

    VOCABULARY_EXPANSION.ST_MODEL = ST_MODEL
    VOCABULARY_EXPANSION.WORD_2_VEC_MODEL = WORD_2_VEC_MODEL
    VOCABULARY_EXPANSION.WORD_TO_ID = WORD_TO_ID
    VOCABULARY_EXPANSION.ID_TO_WORD = ID_TO_WORD
    VOCABULARY_EXPANSION.ST_VOCAB = ST_VOCAB
    VOCABULARY_EXPANSION.W2V_VOCAB = W2V_VOCAB

def GetSTModelWordEmbeddingSize ( ) :
    embedding_layer = list(ST_MODEL.modules())[1]
    embedding_layer_decription = str(embedding_layer)
    return eval(embedding_layer_decription.split(', ')[-1][:-1])

def GetSTModelThoughtSize ( ) :
    encoder_layer = list(ST_MODEL.modules())[2]
    encoder_layer_decription = str(encoder_layer)
    return eval(encoder_layer_decription.split(', ')[0][4:]) * 2

def GetEmbedding ( word ) :
    if not word in ST_VOCAB and not word in W2V_VOCAB : return None
    if word in ST_VOCAB :
        return GetSTModelWordEmbedding(word)
    vec = GetW2VModelWordEmbedding(word)
    vec = vec.reshape(1, -1)
    return VE_MODEL(vec)[0]

def GetThought ( sent ) :
    sent = sent.lower()
    alphabet = set("abcdefghijklmnopqrstuvwxyz'")
    invalid = set(sent) - alphabet
    for char in invalid :
        sent = sent.replace(char, ' ')
    
    embedding = [ GetEmbedding(w) for w in sent.split(' ') if w != '' ]
    if None in embedding : return None
    embedding = torch.stack(embedding).reshape(1, -1, WORD_EMBED_SIZE)
    h0 = torch.zeros(2, 1, THOUGHT_SIZE // 2)
    encoder = list(ST_MODEL.modules())[2]
    return encoder(embedding, h0)[0][:, -1, :].reshape(-1)

def LoadSICKData ( filename = "DATA/SENTENCES_INVOLVING_COMPOSITIONAL_KNOWLEDGE.json" ) :
    global SICK_DATA
    SICK_DATA = None
    with open(filename, 'r') as openfile:
        SICK_DATA = json.load(openfile)

def PrepareDatasets ( data , training_data_ratio = 0.6 ) :
    global TRAIN_SICK_DATA, TEST_SICK_DATA
    inputs = list()
    outputs = list()
    
    l = round(len(data) * training_data_ratio)
    for sample in data[:l] :
        _, sent1, sent2, score = sample.values()
        thought1 = GetThought(sent1)
        if thought1 is None : continue
        thought2 = GetThought(sent2)
        if thought2 is None : continue
        prod = thought1 * thought2
        diff = torch.abs(thought1 - thought2)
        inp = torch.cat([prod, diff], 0)
        
        out = torch.zeros(5,)
        if score == 5 :
            out[4] = 1.0
        else :
            id1 = int(score) + 1
            id2 = id1 - 1
            prob1 = score - int(score)
            prob2 = 1 - prob1
            out[id1-1] = prob1
            out[id2-1] = prob2
        
        inputs.append(inp)
        outputs.append(out)
    
    TRAIN_SICK_DATA = list(zip(inputs, outputs))
    file = open('DATA/SICK_TRAIN_DATA', 'wb')
    pickle.dump(TRAIN_SICK_DATA, file)
    file.close()
    
    TEST_SICK_DATA = data[l:]
    file = open('DATA/SICK_TEST_DATA', 'wb')
    pickle.dump(TEST_SICK_DATA, file)
    file.close()

def LoadTrainDataset ( ) :
    global TRAIN_SICK_DATA
    file = open('DATA/SICK_TRAIN_DATA', 'rb')
    TRAIN_SICK_DATA = pickle.load(file)
    file.close()

def MakeBatches ( dataset , batch_size = 64 ) :
    shuffle(dataset)
    batches = list()
    for start in range(0, len(dataset), batch_size) :
        end = start + batch_size
        inputs, outputs = list(zip(*dataset[start:end]))
        inputs, outputs = torch.stack(inputs), torch.stack(outputs)
        batches.append((inputs, outputs))
    return batches

class SemanticRelatednessNN ( nn.Module ) :
    def __init__ ( self , input_dim , max_score , hidden_size = 512 ) :
        super(SemanticRelatednessNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = max_score
        self.hidden_size = hidden_size
        self.dense1 = nn.Linear ( input_dim, hidden_size )
        self.dense2 = nn.Linear ( hidden_size, max_score )
        
    def forward ( self , inputs ) :
        output1 = self.dense1(inputs)
        activated_output1 = torch.tanh(output1)
        output2 = self.dense2(activated_output1)
        outputs = F.softmax(output2, dim = 1)
        return outputs

def TrainModel ( total_epochs , learning_rate = 0.001 , batch_size = 64 ) :
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(SR_MODEL.parameters(), lr = learning_rate)
    for epoch in range(total_epochs) :
        total_loss = 0.0
        batches = MakeBatches(TRAIN_SICK_DATA, batch_size)
        epoch_start_time = time()
        
        for step, (batch, true) in enumerate(batches) :
            pred = SR_MODEL(batch)
            
            loss = loss_fn(pred, true)
            total_loss += loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = total_loss / len(batches)
        ti = time() - epoch_start_time
        print(' EPOCH {:3d} MEAN LOSS : {:.6f} | DUR : {:.4f}'.format(epoch+1, loss, ti))
    
    torch.save(SR_MODEL, 'MODELS/SEMANTIC_RELATEDNESS_NN')

if __name__ == '__main__' :
    LoadModelsAndUtils()
    WORD_EMBED_SIZE = GetSTModelWordEmbeddingSize()
    THOUGHT_SIZE = GetSTModelThoughtSize()
    LoadSICKData()
    # PrepareDatasets(SICK_DATA)
    # To save time, test and train datasets were already prepared and saved in respective files. Uncomment line 180
    # in case you want to repeat dataset construction. The existing ones will be over-written.
    LoadTrainDataset()

    INPUT_DIMENSION = 2 * THOUGHT_SIZE
    MAX_SCORE = 5
    SR_MODEL = SemanticRelatednessNN(INPUT_DIMENSION, MAX_SCORE)
    # SR_MODEL = torch.load('MODELS/SEMANTIC_RELATEDNESS_NN') # to resume training

    BATCH_SIZE = 64
    TOTAL_EPOCHS = 400
    LEARNING_RATE = 0.001

    SR_MODEL.train()
    TrainModel(TOTAL_EPOCHS, LEARNING_RATE, BATCH_SIZE)
