
# Sentiment Analysis of Amazon reviews with Recurrent Neural Networks (RNN)/Long Short-Term Memory networks (LSTM)
## Motivation
- One of the most important elements for a business is to constanly be aware of the customer reaction and interaction with the company's product. It is vital for these organizations to know exactly (or asses with high probability) what consumers or clients think of recently released and established products or services. Companies are also interested in evaluating customer response to recent initiatives as well as customer service offerings.
- Sentiment analysis is one way to accomplish this necessary task. Sentiment analysis is a field of Natural Language Processing (NLP) that builds models with the aim to identify and classify attributes of the expression contained in the text. We are interested in identifying polarity in the Amazon review text.
- Sentiment Analysis can help to automatically transform the unstructured information into structured data of public opinions about products, services, brands, politics or any other topic that people can express opinions about. This data can be very useful for:
    - commercial applications such as marketing analysis
    - public relations
    - product reviews and feedback
    - customer service.

## Goal
- In this project a Neural Network (Deep Learning model) based on Recurrent Neural Networks aims to classify human emotion (sentiment) in an Amazon review. In other words, we are interested in the sentiment that an Amazon review conveys which could be either possitive or negative.

## Why LSTMs over plain RNN?
- Long Short-Term Memory networks (LSTMs) are a special kind of RNN, capable of learning long-term dependencies. LSTMs do not have a fundamentally different architecture from RNNs, but they incorporate additional components.
- The key to LSTMs is the cell state c(t). A cell state is an additional way to store memory, beyonf using the hidden state h(t). However, c(t) makes it possible that LSTMs can work with much longer sequences as opposed to vanilla RNNs.
- Furthermore, LSTMs have the ability to remove or add information to the cell state, carefully regulated by structures called gates. Gates are a way to optionally let information through. An LSTM has three of these gates, to protect and control the cell state.
    - Forget Gate
    - Input Gate
    - Output Gate
- The figure below shows the architechture of a LSTM network
![LSTM Architecture](https://github.com/mtorrico/Sentiment-Analysis-NeuralNetwork-LSTM/blob/master/lstm_diag.JPG)
- Toy example for LSTM:
    - Consider the following review for a product:
    - I am certainty not happy with the product.
    - If we were to look at each word independently then based on certainty and happy, review will be classified as positive. However, LSTM helps us with word dependencies and by keeping track of the order of the words in the sentece. In this way, LSTM model is able to learn that not comes after certainly and before happy, hence classifying the review as negative.
    

## Tools
  - Python 3.6.8
  - Keras 2.2.4
  - TensorFlow 1.12.0


## Data
- Dataset contains product reviews and metadata from Amazon, including 142.8 million reviews spanning May 1996 - July 2014. These reviews, made available by [Julian McAuley, UCSD](http://jmcauley.ucsd.edu/data/amazon/), are raw qualitative (text) and quantitative (rating) evaluations of products by users

## Preprocessing
- Before we can use the reviews as inputs for the recurrent neural network it is required to do some preprocessing on the data. Our main purpose here is to shrink the observation space. Following preprocessing techniques are applied:
    - Remove review repeats
    - Uniform Spelling of Words by lowercasing all words in reviews
    - Removing Special Characters ( . , ! ? ‘ etc) 
- Create label for classification. Review dataset is made of user reviews and ratings:
    - the reviewText column (string) contains the raw text of the review.
    - the overall column (double) contains the rating given by the user, in {1.0, 2.0, 3.0, 4.0, 5.0}
        - Set 0 for the reviews having an overall of 1.0 (neg class),
        - Set 1 for the reviews having an overall of 5.0 (pos class).
  
## Word Embeddings
- The words have been replaced by integers that indicate the ordered frequency of each word in the dataset. The sentences in each review are therefore comprised of a sequence of integers.
- Word Embeddings are a distributed representation for text that is perhaps one of the key breakthroughs for the impressive performance of deep learning methods on challenging NLP problems.
- Each word is mapped to one specific vector and the vector values are learned by the neural network.
- Toy example since usually we expect higher dimensions:
"life" = [6.3 2.5 8.7 3.4 1.5]

## How to Reduce Overfitting with Regularization in a Neural Network?
- Hyperparameter tuning is applied in Keras
- Dropout regularization is applied to reduce overfitting
- Dropout is a technique where randomly selected neurons are ignored during training and they are dropped out at random. In the fordward pass, the contribution of these nurons is also removed and any weight updates are not applied to the neuron on the backward pass. The main idea behind the regularization, is that as neurons are randomly dropped out of the network during training, then other neurons have to step in and handle the representation required to make predictions for the missing neurons which results in multiple independent internal representations being learned by the network. The impact of this is very significant in the optimization of the network, as the network becomes less sensitive to the specific weights of neurons. This in turn results in a network that is capable of better generalization which will make it less likely to overfit the training data.
- Method to optimize LSTM model:
    - Tune Dropout hyperparameter until we achieve the degree regularization we consider appropriate
    - We can evaluate accuracy scores for train and test data sets as a function of the number of data points (#reviews). This plot will help us make a sane judment of wheter we have enough data points or we have to many as well as if LSTM model is overfitting and Dropout rate has to be tuned further. 
    - Dropout can be applied between layers using the Dropout Keras layer. We can do this by adding Dropout layers between the Embedding and LSTM layers.
    - We consider 20 million reviews and 100 memory units
![Hyperparameter Tuning of LSTM model](https://github.com/mtorrico/Sentiment-Analysis-NeuralNetwork-LSTM/blob/master/lstm_opt.JPG)
    
    







