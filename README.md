## About
The project is inspired by the work of *Kiros et al.* published in the paper <a href = "https://arxiv.org/pdf/1506.06726.pdf"> *Skip-Thought Vectors* </a>. In this paper, the authors describe an approach for unsupervised learning of a generic, distributed sentence encoder. Using the continuity of text from books, an *encoder-decoder model* can be trained that tries to reconstruct the surrounding sentences of an encoded passage. Sentences that share semantic and syntactic properties are thus mapped to similar vector representations. The authors next introduce a simple *vocabulary expansion* method to encode words that were not seen as part of training, allowing us to expand our vocabulary to millions of words. The end result of the implementation will be an off-the-shelf encoder that can produce highly generic sentence representations that are robust and perform well in practice.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://user-images.githubusercontent.com/66432513/120635222-d7803780-c489-11eb-860f-eaefede9bf35.png" width = '830' height = '160'> 

## Skip Thoughts Architecture
The architecture of the model is decribed in detail in *section 2.1* (*Inducing skip-thought vectors*) of the paper. On a high level, the model consists of a *GRU encoder* and two *thought-biased GRU decoders*, one for decoding the previous sentence and the other for decoding the next sentence. The encoder takes as input a sequence of word embeddings representing the distinct words in the center sentence. The output of the encoder is a sequence of vectors, the last of which is the vector representation (*thought*) of the sentence.<br>

The decoders take the same input as the encoder. The *GRU* layers for the decoders are modified to account for the encoded vector representation of the center sentence. This is done by biasing the arguements of the *reset*, *update* and *new* gates in the layers by a scaled *thought* of the center sentence, as shown below. <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://user-images.githubusercontent.com/66432513/120626702-8c155b80-c480-11eb-9ca2-d2f8d6854b99.png" width = '700' height = '160'> 

The output of both the decoders is a sequence of vectors that represents the distinct words in the sentence adjacent to the center sentence (previous or next). As opposed to a dense layer followed by a plain softmax layer to classify these representations as words in the vocabulary, *negative-sampling* is used to get high-quality thoughts in a much shorter training time. *Negative sampling* was popularized by *Mikolov et al.* in their paper <a href="https://proceedings.neurips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf"> *Distributed Representations of Words and Phrases and their Compositionality* </a>, the proposed approach of which is implemented in <a href = "https://github.com/nnakul/word-embeddings"> this </a> project.

*SKIP_THOUGHTS_ARCH.py* source code implements *Skip Thoughts Model* architecture. The model was trained for *80 epochs*, that took almost *32 hours*. The learning rate was reduced discretely from *0.01* to *0.0001* (the bold orange line indicates the point when learning rate was reduced from 0.01 to 0.001).

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://user-images.githubusercontent.com/66432513/120628730-a7816600-c482-11eb-834c-e5774380c5b9.png" width = '450' height = '320'> 

The training text corpus consists of books of various genres, taken from the <a href = "https://www.gutenberg.org/"> *Gutenberg Project* </a>. Though the training corpus originally has about 96,000 sentences, only the first 12,000 are used to train the model so as to restrict the vocabulary to a relatively smaller size (almost *12,000*).

## Vocabulary Expansion
*Section 2.2* (*Vocabulary expansion*) of the paper describes how to expand our encoder’s vocabulary to words it has not seen during training. For this purpose, a shallow (only one layer) unregularized feed-forward neural network has to be trained on the vector representations of words common between the vocabularies of our *Skip Thoughts Model* and the *Google's Word2Vec Model*. The neural network basically creates a mapping from the 300-dimension *Word2Vec* embedding space to the 256-dimension embedding-space of our *Skip Thoughts* model, parameterized by a single weight matrix (dense layer). Like this, for a word that does not belong to our model's vocabulary, the vector representation of that word in the *Google's Word2Vec* embedding space can be mapped to our model's embedding space which can hence be further fed into the trained encoder to get the *thought* of the sentence of which it is a token.

*VOCABULARY_EXPANSION.py* source code implements this architecture. The original vocabulary size of our model was *11,901*. The vocabulary size was expanded to *10 Million*. The neural network for vocabulary expansion was trained on *11,192* samples (exactly *11,192* out of *11,901* words in our model's vocabulary were also present in the *Word2Vec*'s vocabulary). The *Vocabulary Expansion NN* was trained for *500 epochs* (took less than *3 minutes*) at a learning rate of *0.001*.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://user-images.githubusercontent.com/66432513/120634103-815ec480-c488-11eb-9ab7-7cef277da55f.png" width = '450' height = '320'>

During analysis and evaluation of the overall model, it was observed that the model appreciates the semantic meaning of the sentences containing the words that did not originally belong to our model's vocabulary, and their relationships with other similar sentences. Download *Google*'s pre-trained *Word2Vec* model from <a href="https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing"> here <a/> and save it in the *MODELS* folder.

## Semantic Relatedness
Though the encoder-decoder model, along with the expanded vocabulary, is capable of encoding pretty much any *English* sentence into a vector (as long as the words in the sentence fall in the *10 Million* words large vocabulary), it is impossible to comprehend the semantic relationship between two sentences just by merely looking at their thoughts. To be able to understand the correctness of the *thoughts* much more intuitively, it is imperitive to have another model that takes a pair of sentences as input and generates a score in a small range of say 1-5, that indicates how semantically similar the two sentences are. For this purpose, a feed-forward neural network (a *logistic regression classifier*) is designed, that consists of two hidden dense layers. The architecture of the model is explained in *section 3.2* (*Semantic relatedness*) of the paper, along with the scheme to represent a pair of sentences (concatenating component-wise product and absolute difference of their respective *thoughts*).

*SEMANTIC_RELATEDNESS.py* source code implements this architecture. 55% of the *SICK* (Sentences Involving Compositional Knowledge) dataset is used to train the *Semantic Relatedness NN*. The training set is already refined, encoded, packed into a pickle and saved in *DATA* folder. The model was trained for *400 epochs* (took only *6 minutes*) at a learning rate of *0.001*.
 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://user-images.githubusercontent.com/66432513/120641000-be2eb980-c490-11eb-857e-bae756f9171c.png" width = '450' height = '320'>
  
## Compilation
*COMPILATION.py* source code is written to bring all the three models (*Skip Thoughts encoder-decoder*, *Vocabulary Expansion NN* and *Semantic Relatedness NN*) together, for the purposes of analysis and evaluation.
 
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
 
## Evaluation
The performance of the *Skip Thoughts Model* with expanded vocabulary is checked on the remaining 45% of the *SICK* dataset (on which the *Semantic Relatedness NN* was not trained). Given two sentences, our goal is to produce a score of how semantically related these sentences are and hence compare the score generated by our model with the human generated score given in the *SICK* dataset. The evaluation metrics used are *Mean Squared Error*, *Pearson’s correlation coefficient r* and *Spearman’s correlation coefficient ρ*. The values of these evaluation metrics computed for this model are as follows.
 
    Mean Squared Error :   0.78585
    Pearson’s correlation coefficient :   0.55411
    Spearman’s correlation coefficient :  0.50161

The original model (by *Kiros et al.*) gave the following values of the evaluation metrics.
 
    Mean Squared Error :   0.26872
    Pearson’s correlation coefficient :   0.85846
    Spearman’s correlation coefficient :  0.79161

The difference in the performances can be explained by the relatively much smaller training dataset used in this case, to not exhaust the memory while training. This model was trained on *CPU* (due to unavailability of *GPU*) that slowed down the training process. Besides, the quality of dataset can also be responsible because the training corpus used in this project was self-made.
 
## Performance On Analogical Tasks
The performance of the *Skip Thoughts Model* with expanded vocabulary is checked on the basis of its ability to quantify the semantic relatedness between two sentences. Given two sentences, a score between 1 and 5 is generated that is indicative of the extent of relationship between the sentences. The callibre of the model is judged on the basis of how well it performs in the analogical reasoning tasks and how well it is able to find a common relational system between two situations, exemplars, or domains.

The model was able to successfully capture the semantics of most of the sentences, irrespective of whether the words in the sentences were present or not in the original vocabulary of the encoder.
 
 
 
 
The following is the comparison between the scores generated by the original model (by *Kiros et al.*) and this model for some pairs of sentences.

<img src="https://user-images.githubusercontent.com/66432513/120650492-78c3b980-c49b-11eb-962c-dc3a4215903f.png" width = '950' height = '380'> 

