## Implement Continuous Bag of Words model in Tensorflow 1.x. Train and test the model using text8 dataset(http://mattmahoney.net/dc/textdata.html)
Created on 1-28-2020.
- Network achitecture concatenates the outputs of the embedding layer for context words, goes through a connected hidden layer, feeds to Noise Contrastive Estimation(NCE) loss.
- Add dropout layer to the output of hidden layer
- Add tensorflow summary for tensorboard support. Monitor the norm of gradients from embedding layer and NCE layer.
- Build functions to evaluate the word2vec model, such as word similarity, word-addition analogy, prediction of paragraph.