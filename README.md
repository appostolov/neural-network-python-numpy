# Email Address Validation Neural Network

## Overview
Feed forward neural network for guessing email address validity. The structure of the network is embedding layer, then hidden layer half the size of the embedding, then another hidden half the size fo the previous, and single neuron at the end of the network. The embedding layer transfroms the email characters into vector, then merge all the vectors into flat array. The hidden layers use ReLU activation, while the output uses Sigmoid. For training and testing we use 20000 AI generated, labeled data from dataset.csv in 80/20 proportion. The maximum accuracy achieved is 94%

## Build
1. Clone the repo locally using `git clone https://github.com/appostolov/neural-network-python-numpy.git`
2. Open Git Bash in the repo's root
3. Install dependencies using `pip install -r requirements.txt`
4. Run `python main.py`

This will train, test and save you network into `nn.json`

## Settings
* `skeleton.py` is a helper for neural network structure creation
    * Change `char_vector_size` to increase or lower the character vector size. This will make the input layer bigger.
    * `email_max_length` is the max email address length. 50 is used for training purposes. 254 is the real world standard. Changing this will affect the input layer size.
    * If you want to adjust hidden layers size change `first_hidden_size`. This is the size of the first hidden layer. The second hidden layer is half that size.
* `network.py` holds the class of the actual network
    * Change `learn_rate` to adjust the learning speed of the network. This is one of the most sensitive settings.
    * `min_neuron_val` is the minimum ReLU output to prevent dead neurons

## Goals
This project was created to exersize python and understanding neural network concepts like embedding and backpropagation.
Not using popular ML libraries is intentional.

