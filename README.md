# Chinese-English Neural Machine Translator with Attention Mechanism
Built using Tensorflow 2.0 

## 1. Introduction  
This report illustrates a Neural Machine Translation (NMT) project, which translates English texts into Chinese ones. The general process of NMT involves data preprocessing, tokenization, embedding, train-test-split, the construction of an encoder as well as a decoder, model training, and model evaluation. The encoder and the decoder are both Recurrent Neural Networks (RNN). Particularly, a sequence-to-sequence (seq2seq) model with the attention mechanism is implemented to carry out NMT. Some preliminary evaluation results would be given at the end of this report.  
## 2. Dataset
The dataset used in this report is given by ManyThings.org, which contains many English-Chinese pairs (i.e. words or sentences), delimited by tab. Every pair is ended with the corresponding domain name and the material source. The dataset has 22,075 pairs in total. Some examples are given in the graph below. 

## 3. Preprocessing
We first converted Unicode to ASCII. Unlike english, Chinese sentences are composed of individual characters, a word can be one or several characters. We first added a <start> and an <end> token to the sentence for the model to know when to start and stop prediction, and then used jieba, a popular Chinese language tokenizer, to cut sentences into words, and finally used regular expression to add a space between a word and the punctuation following it and replace everything with space except (a-z, A-Z, ".", "?", "!", ",").

Here is an example:
	"我可以借这本书吗？" becomes <start> 我 可以 借 这 本书 吗   ？   <end>
	
Next, we created a dataset that has word pairs in the format: [ENGLISH, CHINESE] and added the length of each sentence. 

The original data set:
If a person has not had a chance to acquire his target language by the time he's an adult, he's unlikely to be able to reach native speaker level in that language.
如果一個人在成人前沒有機會習得目標語言，他對該語言的認識達到母語者程度的機會是相當小的。
CC-BY 2.0 (France) Attribution: tatoeba.org #1230633 (alec) & #1205914 (cienias)
After Preprocessing:
<start> if a person has not had a chance to acquire his target language by the time he s an adult , he s unlikely to be able to reach native speaker level in that language .  <end>
<start> 如果 一個 人 在 成人 前 沒 有 機會習 得 目標 語言   ，   他 對 該 語言 的 認識 達 到 母語者 程度 的 機會 是 相當 小 的   。   <end>

## 4. Tokenization
In general, Tokenization is the process of turning a meaningful piece of data, such as an account number, into a random string of characters called a token that has no meaningful value if breached. Tokens serve as reference to the original data, but cannot be used to guess those values. In our project, we tokenized each sentence, both in English and in Chinese. The lengths of the sentences vary from five words to more than 30 words. If we pad all the sentences to the length of the longest sentence, then most of the tensors are going to be sparse. To speed up performance, we choose a padding length (15 for Chinese, 17 for English) that covers 98 percent of all sentences.
Original sentence:
<start> i m all ears .  <end>
Tensor of the sentence:
[   2    5   40   57 1360    4    3    0    0    0    0    0    0    0    0    0    0]

Original sentence:
<start> 我 洗耳 恭聽   。   <end>
Tensor of the sentence:
[   2    5 6207 6208    4    3    0    0    0    0    0    0    0    0    0]
After tokenization, we found 6237 unique tokens in English and 14052 unique tokens in Chinese. 
The 10 most frequent tokens in Chinese are:
。|我|的|了|你|他|？|在|是|她|，
The 10 most frequent tokens in English are:
. | i | the | to | you | a | ? | is | he | t | tom

## 5. Creating the TensorFlow Dataset
We first split the English and Chinese tensors into training (90%) and testing sets (10%). We create a tensorflow dataset object from the training set with a batch size of 64. Each sample of dataset has the dimension of (64, (17, 15)) The reason why a tensorflow dataset object is chosen over other data formats is that it provides easy integration with tensorflow operations. We also set a variety of neural network parameters. 

## 6. Model
### 6.1 Introduction
The traditional translation systems divide sentences into chunks and then translate sentences phase by phase. However, the traditional method can cause disfluency and make sentences unlike those translated by humans. Therefore, We used a Sequence-to-sequence (seq2seq) model in this project to translate English sentences into Chinese sentences to address these issues. In our case, a sequence of words (or numbers) generates another sequence of words. Instead of generating a single output, seq2seq generates an output across every time step. The most basic form of the seq2seq model is the encoder-decoder structure, with the encoder converting words to number codes and a decoder converting number codes to words in the other languages. Both the encoder and the decoder are composed of multiple layers, the most important of which is the GRU. To solve these problems, other improved versions of the encoder-decoder structures appeared. We adopt the additive attention model proposed by Bahdanau. The attention layer trains the weights put on each element of the encoder sequence, aiming to capture long-range dependencies in languages. 
### 6.2 Structure  
We customize the model using TensorFlow’s subclassing method. The encoder and decoder models inherit from TensorFlow’s ‘model’ class and the Attention layer inherits from the ‘layer’ class. 
#### A. Encoder
The encoder layer is composed of an embedding layer, a dropout layer, and a GRU layer. The embedding layer takes the raw input 2D tensor ‘x’ with the shape of (batch size, padded tensor length) and outputs a 3D tensor with the shape of (batch size, padded tensor length, embedding dimension). In our case, the embedding dimension is 128. In essence, the embedding layer maps each token (originally represented by a single scalar), to a dense vector of length 128. The weights for the embedding are randomly initialized and are gradually adjusted via backpropagation during training. Once trained, the learned word embeddings will roughly encode similarities between words (as they were learned for the specific problem your model is trained on). It is also worth noting that we set the dropout rate of the dropout layer to 0.1 as a regularizing mechanism, but it is only used during training and not during evaluation. The GRU layer of the encoder returns the full sequence of outputs as well as the last hidden state.

#### B. Attention
The attention layer is composed of three dense layers, the first two layers with 1024 neurons each. The final hidden state of the encoder is expanded to a 3D tensor by adding a time axis (the second dimension). The original shape of (64, 1024) transforms into (64, 1, 1024). The purpose is to match the dimension of the encoder’s outputs, which is (64, 17, 1024). The final hidden state after transformation is then passed through the dense layer and creates an output layer of 1024 neurons. The encoder’s outputs at every time step (in total 17 time steps) is also passed through a dense layer, also outputting 1024 neurons. The two outputs are then added and passed through tanh. Finally, the result is passed through a dense layer with an output of one neuron, generating an attention score for every hidden state in the encoder. The dimension of the attention scores are (batch size, sequence length, 1), which in our case equals (64, 17, 1). The attention scores are then squished through a Softmax function and converted to attention weights, which have the same dimension. We then apply the weights to the encoder’s outputs to get the weighted sum of all the encoder’s outputs, also known as the context vector.

#### C. Decoder
The decoder incorporates the attention layer as part of its structure. For its ‘call’ method, it takes in three things as input : (1) the encoder’s hidden state at its last time step, (2) the full sequence of the encoder’s outputs, (3) Chinese (target language)’s tensors, although for initialization it is the 3D (after dimension expansion) tensor for the token ‘<start>’. The first two inputs are used for the attention layered introduced above. The decoder’s architecture allows input (3) to pass through an embedding layer and a dropout layer, and then concatenate with the context vector. Then the result goes through a GRU layer and through a dense layer with the number of neurons equal to the vocabulary size. The idea is that the token with the highest probability becomes the final decoder output.
7. Model Training
There are two scenarios during training. The first scenario is to use teacher forcing, which means that we feed the target as the next input. For each batch, the third input (the tensors) gets updated 15 times (length/number of tokens every Chinese sentence has), each time corresponding to each token. The second scenario is not using teacher forcing, which means that we use its own predictions as the next input. We also do this for 15 times. Regardless of which scenario, each time, we adjust the weights of the encoder, the decoder, and the attention layer, through backpropagation. The loss for the batch is the sum of all losses of the 15 predictions compared with the 15 actual target values.

We set the teacher-forcing ratio to 0.8 and use a random uniform distribution between 0 and 1 to decide which scenario to follow. If the randomly-generated number falls below 0.8, then we use teacher forcing. This means that approximately 80% of the times, we will be using teacher forcing. The advantages of teacher-forcing include faster convergence. Without teacher-forcing, the hidden states of the model will be updated by a sequence of wrong predictions, as the initial weights are random. Errors can accumulate which increases the model’s difficulty of learning. There are disadvantages as well. Teacher-forcing might generate a discrepancy between training and inference, thereby leading to poor model performance and instability. Therefore, we use a mixture of both methods in training.
## 8. Evaluation
We test the model’s translation in this part. The evaluation process is similar to training without teacher forcing. Each input that is used for evaluation is preprocessed, then turned to a tensor. At each time step, the decoder outputs a 1-D tensor of the Chinese vocabulary size and chooses the position of the largest one. The position--the predicted ID -- is then used to query the Chinese dictionary according to the original tokenizer and outputs a token. The token output is once again fed into the decoder as the input. In the meantime, we record the attention weights during each time step, and use the weights for plotting. The ‘evaluation’ function returns two things: attention plots and translated sentences. The ‘<end>’ marker breaks the prediction out of the loop, so the neural network knows to always stop the prediction on ‘<end>’. We also used BLEU score, a commonly-used metric to assess the quality of translations, to evaluate every sentence of the testing set (more than 200 sentences in total). We then calculate the mean of those scores, which is around 52.3%. However, BLEU score is limited in interpreting the quality of translation. There are many correct ways to translate a sentence, but we only have one reference translation per sentence. Intelligibility or grammatical correctness are not taken into account.
9. Findings and Conclusion
We tested many sentences and plotted the attention heatmaps. We found that the sentences translated simple sentences with satisfactory accuracy. There are two main problems. First, the tokenization of the Chinese sentences can be inaccurate. There are instances where a phrase should have been cut into two parts, but jieba failed to do so and lumped them together instead. This resulted in some imprecise translations. Another problem is that the training set is limited and lacks many essential vocabulary. The fact that the translation relies on querying the training dictionary makes this a serious issue. In the future, to further improve our result, we need to use a bigger training set.

## References 

https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

https://www.tensorflow.org/tutorials/text/nmt_with_attention

https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3
