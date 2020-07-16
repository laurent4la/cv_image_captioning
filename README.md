# cv_image_captioning

Simple image captioning system using a Sequence to Sequence model

LSTM model combined with a CNN image embedder and word embeddings.

## training

At every time step, we look at the current caption word as input and combine it with the hidden state of the LSTM cell to produce an output. This output is then passed to the fully connected layer that produces a distribution that represents the most likely next word.
