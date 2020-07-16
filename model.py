import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """
        :param vocab_size: size of vocabulary
        """
        super(DecoderRNN, self).__init__()

        # an Embedding module containing vocab_size tensors of size embed_size
        # The input to the module is a list of indices,
        # and the output is the corresponding word embeddings.
        self.embedding = nn.Embedding(
                num_embeddings = vocab_size,
                embedding_dim = embed_size)

        # input_size – The number of expected features in the input x
        # hidden_size – The number of features in the hidden state h
        # num_layers – Number of recurrent layers
        self.lstm = nn.LSTM(
                input_size = embed_size,
                hidden_size = hidden_size,
                num_layers = num_layers,
                batch_first = True)

        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        """
        Forward propagation.
        features - embedded image feature vector
        captions - caption tensors (torch.Tensor(caption).long() see in data_loader __getitem__)
        """

        # transforming a sequence of words (caption) into a sequence of numerical values
        # a vector of "embed_size" numbers where each number maps to a specific word in our vocabulary
        embedded = self.embedding(captions)

        # linking image features and captions to create an input of vector of size embed_size
        # dim - dimension over which the tensors are concatenated
        input = torch.cat((features.unsqueeze(1), embedded), dim = 1)

        # lstm_outputs - tensor containing the output features (h_t) from the last layer of the LSTM, for each t.
        # input - needs to be of size embed_size
        lstm_outputs, _ = self.lstm(input)

        out = self.linear(lstm_outputs)

        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output_sentence = []
        for i in range(max_len):
            lstm_outputs, states = self.lstm(inputs, states)
            lstm_outputs = lstm_outputs.squeeze(1)
            out = self.linear(lstm_outputs)
            last_pick = out.max(1)[1]
            output_sentence.append(last_pick.item())
            inputs = self.embedding(last_pick).unsqueeze(1)

        return output_sentence
