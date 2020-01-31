# Nikola.ai
A thought experiment for therabeats. Service 1.

This consists of a Python File, an .ipynb Ipython Notebook file and a website folder.

The documentation in the notebook regarding Transformer is from the official TensorFlow website.

This repository contains the demo model and not the actual representation. 
We are doing the demo version with the smaller datasets due to constraints like Bandwidth on Cloud Platform, depleted funds for usage of Cloud Platforms like AWS, limited storage space. The actual dataset to be used would be a 40GB dataset from Reddit trained on an Nvidia 1080-Ti using Neural Machine Translator seq2seq implementation.

## Transformers :

The use of artificial neural networks to create chatbots is increasingly popular nowadays, however, teaching a computer to have natural conversations is very difficult and often requires large and complicated language models.
Preprocessing the Cornell Movie-Dialogs Corpus using TensorFlow Datasets and creating an input pipeline using tf.data
Implementing MultiHeadAttention with Model subclassing
Implementing a Transformer with Functional API


Transformer, proposed in the paper Attention is All You Need, is a neural network architecture solely based on self-attention mechanism and is very parallelizable.A Transformer model handles variable-sized input using stacks of self-attention layers instead of RNNs or CNNs. This general architecture has a number of advantages:
* It makes no assumptions about the temporal/spatial relationships across the data. This is ideal for processing a set of objects.
* Layer outputs can be calculated in parallel, instead of a series like an RNN.
* Distant items can affect each other’s output without passing through many recurrent steps, or convolution layers.
* It can learn long-range dependencies.

#### The disadvantage of this architecture:

* For a time-series, the output for a time-step is calculated from the entire history instead of only the inputs and current hidden-state. This may be less efficient.

* If the input does have a temporal/spatial relationship, like text, some positional encoding must be added or the model will effectively see a bag of words.

WebSite ScreenShots : Coming soon ( By 1st February 1pm )

References : 
* Use the Adam optimizer with a custom learning rate scheduler according to the formula in the [paper](https://arxiv.org/abs/1706.03762).
* Documentation help, Documentation in the JupyterNotebook help and code structure inspired from https://www.tensorflow.org/tutorials/text/transformer


