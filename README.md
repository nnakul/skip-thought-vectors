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

During analysis and evaluation of the overall model, it was observed that the model appreciates the semantic meaning of the sentences containing the words that did not originally belong to our model's vocabulary, and their relationships with other similar sentences.

## Semantic Relatedness

## Compilation

## Evaluation

## Performance On Analogical Tasks

