# cv_image_captioning

Simple image captioning system to automatically describe the content of an image using a Sequence to Sequence model

Note:

The pretrained folder contains a different version that could be directly implemented.
Check "the "pretrained/README.md #usage" section for steps to implement this solution.
which comes from this project: https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning

## Model
The model implemented is based on Show and Tell: A Neural Image Caption Generator.

LSTM model combined with a CNN image embedder and word embeddings.

## Training
At every time step, we look at the current caption word as input and combine it with the hidden state of the LSTM cell to produce an output.

This output is then passed to the fully connected layer that produces a distribution that represents the most likely next word.

## Hyperparameters
Learning Rate: 0.01

batch_size = 64          # batch size

vocab_threshold = 5      # minimum word count threshold

vocab_from_file = True   # if True, load existing vocab file

embed_size = 200         # dimensionality of image and word embeddings

hidden_size = 512        # number of features in hidden state of the RNN decoder

num_epochs = 3           # number of training epochs

save_every = 1           # determines frequency of saving model weights

print_every = 100        # determines window for printing average loss

log_file = 'training_log.txt'       # name of file with saved training loss and perplexity

## Results

Checkout the folder results:

https://github.com/laurent4la/cv_image_captioning/blob/master/results/trainin_2020_07_16.txt
