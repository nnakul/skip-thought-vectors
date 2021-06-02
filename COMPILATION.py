
import pickle
import torch
import numpy as np
from gensim.models import KeyedVectors
from SKIP_THOUGHTS_ARCH import SkipThoughtsNN
import VOCABULARY_EXPANSION
from VOCABULARY_EXPANSION import VocabularyExpansionNN
from VOCABULARY_EXPANSION import GetSTModelWordEmbedding
from VOCABULARY_EXPANSION import GetW2VModelWordEmbedding
import SEMANTIC_RELATEDNESS
from SEMANTIC_RELATEDNESS import SemanticRelatednessNN
from SEMANTIC_RELATEDNESS import GetEmbedding
from SEMANTIC_RELATEDNESS import GetThought
from SEMANTIC_RELATEDNESS import GetSTModelWordEmbeddingSize
from SEMANTIC_RELATEDNESS import GetSTModelThoughtSize

def LoadModelsAndUtils ( loc = 'MODELS' ) :
    global ST_MODEL, VE_MODEL, SR_MODEL, WORD_2_VEC_MODEL, WORD_TO_ID, ID_TO_WORD, ST_VOCAB, W2V_VOCAB
    ST_MODEL = torch.load(loc+'/SKIP_THOUGHTS_NN')
    ST_MODEL.eval()
    VE_MODEL = torch.load(loc+'/VOCABULARY_EXPANSION_NN')
    VE_MODEL.eval()
    SR_MODEL = torch.load(loc+'/SEMANTIC_RELATEDNESS_NN')
    SR_MODEL.eval()

    limit = int(1e7) # 10 Million words
    WORD_2_VEC_MODEL = KeyedVectors.load_word2vec_format(loc+'/GoogleNews-vectors-negative300.bin.gz', binary = True, limit = limit)
    
    file = open(loc+'/UTILS/SKIP_THOUGHTS_NN_WORD2ID', 'rb')
    WORD_TO_ID = pickle.load(file)
    file.close()

    file = open(loc+'/UTILS/SKIP_THOUGHTS_NN_ID2WORD', 'rb')
    ID_TO_WORD = pickle.load(file)
    file.close()

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

    SEMANTIC_RELATEDNESS.ST_MODEL = ST_MODEL
    SEMANTIC_RELATEDNESS.ST_VOCAB = ST_VOCAB
    SEMANTIC_RELATEDNESS.W2V_VOCAB = W2V_VOCAB
    SEMANTIC_RELATEDNESS.VE_MODEL = VE_MODEL

def LoadTestDataset ( ) :
    global TEST_SICK_DATA
    file = open('DATA/SICK_TEST_DATA', 'rb')
    TEST_SICK_DATA = pickle.load(file)
    file.close()

def GetSentencePairVector ( sent1 , sent2 ) :
    thought1 = GetThought(sent1)
    if thought1 is None : return None
    thought2 = GetThought(sent2)
    if thought2 is None : return None
    prod = thought1 * thought2
    diff = torch.abs(thought1 - thought2)
    vec = torch.cat([prod, diff], 0)
    return vec

def GetSimilarityRelationScore ( sent1 , sent2 ) :
    vec = GetSentencePairVector(sent1, sent2)
    if vec is None : return None
    inp = torch.stack([vec])
    score_dist = SR_MODEL(inp)[0]
    scores = torch.tensor([1., 2., 3., 4., 5.])
    return round(torch.dot(scores, score_dist).item(), 3)

def PredictScores ( dataset ) :
    true_scores = list()
    predicted_scores = list()
    for sample in TEST_SICK_DATA :
        _, sent1, sent2, true = sample.values()
        pred = GetSimilarityRelationScore(sent1, sent2)
        if pred is None : continue
        true_scores.append(true)
        predicted_scores.append(pred)
    true_scores = np.array(true_scores)
    predicted_scores = np.array(predicted_scores)
    return predicted_scores, true_scores

def GetMSE ( x , y ) :
    return ((x - y) ** 2).mean()

def GetPearsonsCoeff ( x , y ) :
    x_mean = x.mean()
    y_mean = y.mean()
    numer = (x - x_mean) * (y - y_mean)
    numer = numer.sum()
    denom = ((x - x_mean)**2).sum() * ((y - y_mean)**2).sum()
    denom = denom ** 0.5
    return numer / denom

def GetSpearmansCoeff ( x , y ) :
    rank_x = list(enumerate(x))
    rank_x.sort(key = lambda x : -x[1])
    rank_x = [first for first, second in rank_x]
    rank_x = list(enumerate(rank_x))
    rank_x.sort(key = lambda x : x[1])
    rank_x = [first + 1 for first, second in rank_x]
    x = np.array(rank_x)
    
    rank_y = list(enumerate(y))
    rank_y.sort(key = lambda y : -y[1])
    rank_y = [first for first, second in rank_y]
    rank_y = list(enumerate(rank_y))
    rank_y.sort(key = lambda y : y[1])
    rank_y = [first + 1 for first, second in rank_y]
    y = np.array(rank_y)
    
    n = x.shape[0]
    coeff = (x - y) ** 2
    coeff = coeff / (n * (n*n - 1))
    coeff = 6 * coeff.sum()
    return coeff

if __name__ == '__main__' :
    LoadModelsAndUtils()
    LoadTestDataset()
    SEMANTIC_RELATEDNESS.WORD_EMBED_SIZE = GetSTModelWordEmbeddingSize()
    SEMANTIC_RELATEDNESS.THOUGHT_SIZE = GetSTModelThoughtSize()
    predicted_scores, true_scores = PredictScores(TEST_SICK_DATA)

    print(' MEAN SQAURED ERROR :', GetMSE(predicted_scores, true_scores))
    print(' PEARSON\'S COEFFICIENT :', GetPearsonsCoeff(predicted_scores, true_scores))
    print(' SPEARMAN\'S COEFFICIENT :', GetSpearmansCoeff(predicted_scores, true_scores))
