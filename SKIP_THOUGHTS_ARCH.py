
import pickle
import numpy as np
from time import time
from random import shuffle
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F

def LoadText ( filename ) :
    global TEXT
    file = open(filename, 'r')
    TEXT = file.read()
    file.close()

def ProcessText ( limit = None, upper_limit = float('inf'), lower_limit = float('-inf') ) :
    UPPER_LIMIT = upper_limit
    LOWER_LIMIT = lower_limit
    
    global TEXT_TOKENS, ID_TO_WORD, WORD_TO_ID, VOCAB_SIZE
    all_sents = TEXT.split('\n')
    if not limit is None :
        all_sents = all_sents[:limit]
    
    TEXT_TOKENS = list()
    all_words = list()
    
    for sent in all_sents :
        tokens = sent.split(' ')
        if not ( LOWER_LIMIT <= len(tokens) <= UPPER_LIMIT ) :
            continue
        all_words.extend(tokens)
        TEXT_TOKENS.append(tokens)
    
    global FREQUENCY_DIST_TABLE
    FREQUENCY_DIST_TABLE = Counter(all_words)
    FREQUENCY_DIST_TABLE = dict(FREQUENCY_DIST_TABLE)
    FREQUENCY_DIST_TABLE[''] = 0
    
    FREQUENCY_DIST_TABLE = list(FREQUENCY_DIST_TABLE.items())
    FREQUENCY_DIST_TABLE.sort(key = lambda x : x[0])
    FREQUENCY_DIST_TABLE = dict(FREQUENCY_DIST_TABLE)
    
    ID_TO_WORD = list(FREQUENCY_DIST_TABLE.keys())
    VOCAB_SIZE = len(ID_TO_WORD)
    WORD_TO_ID = { word : idx for idx, word in enumerate(ID_TO_WORD) }

def FindMaxSequenceLength ( ) :
    global SEQ_LENGTH
    longest_seq = max(TEXT_TOKENS, key = lambda i: len(i))
    SEQ_LENGTH = len(longest_seq)

def EncodeTextTokens ( ) :
    FindMaxSequenceLength()
    global TEXT_TOKENS_IDXD
    TEXT_TOKENS_IDXD = list()
    for sent in TEXT_TOKENS :
        words_index = list()
        for word in sent :
            words_index.append(WORD_TO_ID[word])
        padding = SEQ_LENGTH - len(sent)
        words_index.extend([0] * padding)
        TEXT_TOKENS_IDXD.append(words_index)

def PrepareDataset ( ) :
    EncodeTextTokens()
    global DATASET
    inputs = list()
    outputs_prev = list()
    outputs_next = list()
    
    for i in range(1, len(TEXT_TOKENS_IDXD)-1) :
        inputs.append(TEXT_TOKENS_IDXD[i])
        outputs_prev.append(TEXT_TOKENS_IDXD[i-1])
        outputs_next.append(TEXT_TOKENS_IDXD[i+1])
        
    inputs.append(TEXT_TOKENS_IDXD[0])
    outputs_prev.append([0] * SEQ_LENGTH)
    outputs_next.append(TEXT_TOKENS_IDXD[1])
    
    inputs.append(TEXT_TOKENS_IDXD[-1])
    outputs_prev.append(TEXT_TOKENS_IDXD[-2])
    outputs_next.append([0] * SEQ_LENGTH)
    
    DATASET = list(zip(inputs, outputs_prev, outputs_next)) 

def MakeBatches ( batch_size ) :
    shuffle(DATASET)
    batches = list()
    for start in range(0, len(DATASET), batch_size) :
        end = start + batch_size
        batch, prev_labels, next_labels = list(zip(*DATASET[start:end]))
        prev_labels, next_labels = torch.Tensor(prev_labels), torch.Tensor(next_labels)
        prev_labels, next_labels = prev_labels.long(), next_labels.long()
        labels = (prev_labels, next_labels)
        batch = torch.Tensor(batch).long()
        batches.append((batch, labels))
    return batches

def InitializeNoiseDistribution ( ) :
    global NOISE_DISTRIBUTION
    word_freqs = np.array(list(FREQUENCY_DIST_TABLE.values()))
    unigram_dist = word_freqs / word_freqs.sum()
    NOISE_DISTRIBUTION = torch.from_numpy(unigram_dist**(0.75)/np.sum(unigram_dist**(0.75)))

class SkipThoughtsNN ( nn.Module ) :
    def __init__ ( self , thought_size , word_embed_size , hidden_size = 256 ) :
        super(SkipThoughtsNN, self).__init__()
        self.hidden_size = hidden_size
        self.word_embed_size = word_embed_size
        self.thought_size = thought_size
        
        self.word_embed = nn.Embedding ( VOCAB_SIZE, word_embed_size )
        self.gru_enc = nn.GRU ( input_size = word_embed_size, hidden_size = thought_size // 2,
                                batch_first = True, bidirectional = True )
        
        bias = False
        self.W_r_dp = nn.Linear ( word_embed_size, hidden_size, bias = bias )
        self.U_r_dp = nn.Linear ( hidden_size, hidden_size, bias = bias )
        self.C_r_dp = nn.Linear ( thought_size, hidden_size, bias = bias )
        self.W_z_dp = nn.Linear ( word_embed_size, hidden_size, bias = bias )
        self.U_z_dp = nn.Linear ( hidden_size, hidden_size, bias = bias )
        self.C_z_dp = nn.Linear ( thought_size, hidden_size, bias = bias )
        self.W_n_dp = nn.Linear ( word_embed_size, hidden_size, bias = bias )
        self.U_n_dp = nn.Linear ( hidden_size, hidden_size, bias = bias )
        self.C_n_dp = nn.Linear ( thought_size, hidden_size, bias = bias )
        
        self.W_r_dn = nn.Linear ( word_embed_size, hidden_size, bias = bias )
        self.U_r_dn = nn.Linear ( hidden_size, hidden_size, bias = bias )
        self.C_r_dn = nn.Linear ( thought_size, hidden_size, bias = bias )
        self.W_z_dn = nn.Linear ( word_embed_size, hidden_size, bias = bias )
        self.U_z_dn = nn.Linear ( hidden_size, hidden_size, bias = bias )
        self.C_z_dn = nn.Linear ( thought_size, hidden_size, bias = bias )
        self.W_n_dn = nn.Linear ( word_embed_size, hidden_size, bias = bias )
        self.U_n_dn = nn.Linear ( hidden_size, hidden_size, bias = bias )
        self.C_n_dn = nn.Linear ( thought_size, hidden_size, bias = bias )
        
        self.dec_out = nn.Linear ( hidden_size, word_embed_size, bias = bias )
        
    def forward ( self , inputs ) :
        orig_inputs = inputs
        inputs = self.word_embed(inputs)
        
        seq_length = inputs.shape[1]
        batch_size = inputs.shape[0]
        
        h0_enc = torch.zeros(2, inputs.size(0), self.thought_size // 2)
        thoughts, _ = self.gru_enc(inputs, h0_enc)
        masking_indices = [ np.argmax(orig_inputs[x] == 0) - 1 for x in range(batch_size) ]
        thoughts = torch.stack( [ thoughts[idd, k, :] for idd, k in enumerate(masking_indices) ] )
        
        prev_sent_words = list()
        next_sent_words = list()
        
        h_dec_prev = torch.zeros(batch_size, self.hidden_size)
        for t in range(seq_length) :
            inp = inputs[:, t, :]
            r = torch.sigmoid ( self.W_r_dp(inp) + self.U_r_dp(h_dec_prev) + self.C_r_dp(thoughts) )
            z = torch.sigmoid ( self.W_z_dp(inp) + self.U_z_dp(h_dec_prev) + self.C_z_dp(thoughts) )
            n = torch.tanh ( self.W_n_dp(inp) + self.U_n_dp(r*h_dec_prev) + self.C_n_dp(thoughts) )
            h_dec_prev = (1 - z)*h_dec_prev + z*n
            prev_sent_words.append( self.dec_out(h_dec_prev) )

        h_dec_next = torch.zeros(batch_size, self.hidden_size)
        for t in range(seq_length) :
            inp = inputs[:, t, :]
            r = torch.sigmoid ( self.W_r_dn(inp) + self.U_r_dn(h_dec_next) + self.C_r_dn(thoughts) )
            z = torch.sigmoid ( self.W_z_dn(inp) + self.U_z_dn(h_dec_next) + self.C_z_dn(thoughts) )
            n = torch.tanh ( self.W_n_dn(inp) + self.U_n_dn(r*h_dec_next) + self.C_n_dn(thoughts) )
            h_dec_next = (1 - z)*h_dec_next + z*n
            next_sent_words.append( self.dec_out(h_dec_next) )
            
        prev_words = torch.stack(prev_sent_words).reshape(batch_size, seq_length, -1)
        next_words = torch.stack(next_sent_words).reshape(batch_size, seq_length, -1)
        
        return prev_words, next_words
    
    def get_loss ( self , inputs , outputs ) :
        L, embed_size = inputs.shape
        
        outputs = self.word_embed(outputs)
        inputs = inputs.view(L, embed_size, 1)
        outputs = outputs.view(L, 1, embed_size)
        
        out_loss = torch.bmm(outputs, inputs).sigmoid().log()
        out_loss = out_loss.squeeze()

        noise_tensor = self.get_forward_noise(L, 5)
        noise_loss = torch.bmm(noise_tensor.neg(), inputs).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum(1)
        
        return -(out_loss + noise_loss).mean()
    
    def get_forward_noise ( self , batch_size , sample_count ) :
        if NOISE_DISTRIBUTION is None : InitializeNoiseDistribution()
        noise_words = torch.multinomial(NOISE_DISTRIBUTION, batch_size * sample_count, replacement=True)
        noise_vector = self.word_embed(noise_words).view(batch_size, sample_count, -1)        
        return noise_vector

def SaveProgressAndModel ( loss_progress , loc = 'MODELS' ) :
    file = open(loc+'/UTILS/SKIP_THOUGHTS_NN_LOSS_PROGRESS', 'rb')
    x = pickle.load(file)
    file.close()
    file = open(loc+'/UTILS/SKIP_THOUGHTS_NN_LOSS_PROGRESS', 'wb')
    pickle.dump(x + loss_progress, file)
    file.close()
    torch.save(ST_MODEL, loc+'/SKIP_THOUGHTS_NN')
    
def ClearProgress ( loc = 'MODELS' ) :
    file = open(loc+'/UTILS/SKIP_THOUGHTS_NN_LOSS_PROGRESS', 'wb')
    pickle.dump([], file)
    file.close()

def LoadModel ( loc = 'MODELS' ) :
    global ST_MODEL
    ST_MODEL = torch.load(loc+'/SKIP_THOUGHTS_NN')

def SaveLookupTables ( loc = 'MODELS' ) :
    file = open(loc+'/UTILS/SKIP_THOUGHTS_NN_WORD2ID', 'wb')
    pickle.dump(WORD_TO_ID, file)
    file.close()

    file = open(loc+'/UTILS/SKIP_THOUGHTS_NN_ID2WORD', 'wb')
    pickle.dump(ID_TO_WORD, file)
    file.close()

def Mask ( true , pred ) :
    bool_idx = (true != 0)
    true_masked = true[bool_idx]
    pred_masked = pred[bool_idx]
    return true_masked, pred_masked

def GetParametersCounts ( ) :
    total_params = 0
    for p in ST_MODEL.parameters() :
        total_params += torch.numel(p)
    return total_params

def TrainModel ( total_epochs , learning_rate = 0.01 , batch_size = 64 ) :
    global ST_MODEL
    optimizer = torch.optim.Adam(ST_MODEL.parameters(), lr = learning_rate)

    loss_progress = []
    for epoch in range(total_epochs) :
        print('\n EPOCH {} STARTED '.format(epoch+1))
        total_loss = 0.0
        batches = MakeBatches(batch_size)
        epoch_start_time = time()
        
        for step, (batch, labels) in enumerate(batches) :
            step_start_time = time()
            true_prev, true_next = labels
            
            prev_words, next_words = ST_MODEL(batch)
            
            true_prev, prev_words = Mask(true_prev, prev_words)
            true_next, next_words = Mask(true_next, next_words)
            
            loss_prev = ST_MODEL.get_loss(prev_words, true_prev)
            loss_next = ST_MODEL.get_loss(next_words, true_next)
            loss = loss_prev + loss_next
            
            total_loss += loss
            loss_progress.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ti = time() - step_start_time
            print('    STEP : {:3d} | LOSS : {:.6f} | DUR : {:.4f}'.format(step+1, loss, ti))

        loss = total_loss / len(batches)
        ti = time() - epoch_start_time
        print(' EPOCH\'S MEAN LOSS : {:.6f} | DUR : {:.4f}'.format(loss, ti))

    SaveProgressAndModel(loss_progress)

if __name__ == '__main__' :
    TEXT = None
    TEXT_TOKENS = None
    ID_TO_WORD = None
    WORD_TO_ID = None
    VOCAB_SIZE = None
    FREQUENCY_DIST_TABLE = None
    SEQ_LENGTH = None
    TEXT_TOKENS_IDXD = None
    DATASET = None
    NOISE_DISTRIBUTION = None

    LoadText('DATA/ST_TRAINING_CORPUS.txt')
    ProcessText(limit = 12000)
    PrepareDataset()
    InitializeNoiseDistribution()
    SaveLookupTables()

    THOUGHT_SIZE = 512
    WORD_EMBED_SIZE = 256
    BATCH_SIZE = 64
    TOTAL_EPOCHS = 80
    LEARNING_RATE = 0.01 # learning rate was discretely reduced from 0.01 to 0.0001 over 80 epochs

    # ClearProgress() # when a new model is being trained
    ST_MODEL = SkipThoughtsNN(THOUGHT_SIZE, WORD_EMBED_SIZE)
    # LoadModel() # to resume training

    ST_MODEL.train()
    TrainModel(TOTAL_EPOCHS, LEARNING_RATE, BATCH_SIZE)
