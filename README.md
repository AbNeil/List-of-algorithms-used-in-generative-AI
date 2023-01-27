## Generative AI models categories

#### Text-to-image
Input is a text prompt
output is an image.

i.e Stable diffusion, imagen, midjourney

####  Text-to-3D models
The models that have been described in the previous section deal with the mapping of text prompts to 2D images. However, for some industries like gaming,
it is necessary to generate 3D images. 

Dreamfusion : DreamFusion is a text-to-3D model developed by Google that uses a pretrained 2D text-to-image diffusion model to perform textto-3D synthesis. In particular, Dreamfusion replaces previous CLIP techniques with a loss derived from distillation of a 2D diffusion model. Concretely, the diffusion model can be used as a loss within a generic continuous optimization problem to generate samples. Critically, sampling in parameter space is much harder than in pixels as we want to create 3D models that look like good images when rendered from random angles. To solve the issue, this model uses a differentiable generator. Magic3D is another text to 3D model made by NVIDIA.

#### Image-to-Text

VisualGPT made by OpenAI, VisualGPT leverages knowledge from GPT-2. In order to bridge the semantic gap between different modalities, a novel encoder-decoder attention mechanism is designed with an unsaturated rectified gating function. Critically, the biggest advantage of this model is that it does not need for as much data as other image-to-text models. In particular, improving data efficiency in image captioning networks would enable quick data curation, description of rare objects, and applications in specialized domains.

#### Text-to-Video
 
[Phenaki](https://phenaki.video/), developed by Google Research, can generate realistic videos from a sequence of textual prompts. It can be accessed via its API on GitHub, and is the first model that can generate videos from open domain time variable prompts. It achieves this by jointly training on a large image-text pairs dataset and a smaller number of video-text examples, resulting in generalization beyond what is available in video datasets.

Additionally two open source demo models [CogVideo[(https://github.com/THUDM/CogVideo) by a groups of cs students a model by [Antonia Antonova](https://antonia.space/text-to-video-generation) and  have presented their own innovative methods of generating video from text
 
#### Video-to-text
[Google Muse](https://muse.ai/)

Soundify, developed by Runway, is a system that matches sound effects to video for professional video editing. It uses quality sound effects libraries and a neural network with zero-shot image classification capabilities (CLIP) to classify, synchronize, and mix sound effects with a video. The video is split based on color histogram distances to reduce distinct sound emitters, and intervals are identified by comparing the effects label with each frame and pinpointing consecutive matches above a threshold. Effects are then split into one-second chunks and stitched via crossfades.

#### Text-to-Audio

AudioLM : by Google for high-quality audio generation with long-term consistency. In particular, AudioLM maps the input audio into a sequence of discrete tokens and casts audio generation as language modeling task in this representation space. By training on large corpora of raw audio waveforms, AudioLM learns to generate natural and coherent continuations given short prompts. The approach can be extended beyond speech by generating coherent piano music continuations, despite being trained without any symbolic representation of music. Audio signals involve multiple scales of abstractions. When it comes to audio synthesis, multiple scales make achieving high audio quality while displaying consistency very challenging. This is achieved by this model by combining recent advances in neural audio compression, self-supervised representation learning and language modelling

Jukebox :by OpenAI this model tries to solve it by means of a hierarchical VQ-VAE architecture to compress audio into a latent space, with a loss function designed to retain the most amount of information.
 
#### Audio-to-text
Shazam and google search using Discrete FFT fast fourier transforms

#### Audio to Midi

[Spotify Basic Pitch](https://basicpitch.spotify.com/) 


## Artificial Neural Networks used in Generative AI

#### CNNs

* Convolutional layer
* Pooling operations
    * L2-norm pooling 
    * max pooling
    * sum pooling
* Fully connected layer
* Activation function
* Weight initialization
* Backpropagation
* Stochastic gradient descent (SGD)
* Adam optimization
* RMSprop optimization
* Dropout regularization

#### RNNs

* Recurrent layer
* Long Short-Term Memory LSTM layer
* GRU layer
* Peephole connections
* Activation function
* Weight initialization
* Backpropagation through time (BPTT)
* Truncated BPTT
* Stochastic gradient descent (SGD)
* Adam optimization
* RMSprop optimization
* Dropout regularization

#### GANs

* Generator network
* Discriminator network
* Loss function
* Optimization algorithm (e.g., SGD, Adam)
* Minimax game

#### VAEs

* Encoder network
* Decoder network
* Loss function
* Optimization algorithm (e.g., SGD, Adam)
* Latent space

#### U-Nets

* Encoder network
* Decoder network
* Skip connections
* Loss function
* Optimization algorithm (e.g., SGD, Adam)
* Up-sampling layer

## Technologies used with neural networks

#### Supervised learning

* Linear regression
* Logistic regression
* Support vector machines (SVMs)
* Decision trees
* Random forests
* Neural networks
* Deep learning
* Naive Bayes
* K-nearest neighbors (KNN)
* AdaBoost
* Gradient boosting
* Extreme gradient boosting (XGBoost)
* LogitBoost
* Multi-layer perceptron (MLP)
* Perceptron
* Linear discriminant analysis (LDA)
* Quadratic discriminant analysis (QDA)
* Partial least squares (PLS)
* Least absolute shrinkage and selection operator (LASSO)
* Elastic net
* Ridge regression
* Support vector regression (SVR)

#### Unsupervised learning

* K-means clustering
* Hierarchical clustering
* Self-organizing maps (SOMs)
* Expectation-maximization (EM)
* Principal component analysis (PCA)
* Singular value decomposition (SVD)
* Factor analysis
* Independent component analysis (ICA)
* Autoencoders
* Deep belief networks (DBNs)
* Hidden Markov models (HMMs)
* Markov random fields (MRFs)
* Restricted Boltzmann machines (RBMs)
* Deep Boltzmann machines (DBMs)
* Non-negative matrix factorization (NMF)

#### Reinforcement learning

* Q-learning
* SARSA
* DDPG
* A3C

#### Transfer learning

* Fine-tuning
* Feature extraction

# Detailed with formulas, equations or processes

#### CNNs

- **Convolutional layer:** A convolutional layer applies a convolution operation to the input data to extract features. The convolution is defined by a kernel or filter that is applied to the input data in a sliding window fashion. The output of the convolution is called a feature map. The formula for the convolution operation is:

![convolution formula]( https://user-images.githubusercontent.com/88499318/209689885-727c31f2-07d3-4f6a-819a-b3185d3919c3.png)

where *f* is the input data (e.g., an image), *g* is the kernel or filter, and *h* is the output or feature map.

- **Pooling layer:** A pooling layer down-samples the input data by applying a pooling (The term "pooling" comes from the idea of taking a set of values and "pooling" them together into a single value) operation, such as max pooling or average pooling. The pooling operation is applied in a sliding window fashion and reduces the dimensionality of the data. The formula for max pooling is:

    * L2-norm pooling  is a type of pooling operation that computes the L2-norm of the input values within a given window. The formula for L2-norm pooling is:

L2-norm pooling formula

where f is the input data, g is the pooling function (e.g., L2-norm), and h is the output of the pooling operation.
    * Max pooling a common type of pooling operation that selects the maximum value from a set of input values within a given window. For example, consider a max pooling operation that takes a 2x2 window of values from a matrix and outputs the maximum value in that window. The formula for this operation can be written as:

> Output = max(Input[i:i+2, j:j+2])

where Output is the resulting value after the max pooling operation, and Input is the matrix of input values. i and j are the indices of the elements in the matrix, and the colon operator (:) is used to specify a range of values.
    * Sum pooling is another type of pooling operation that simply sums the input values within a given window. The formula for sum pooling is:

sum pooling formula

where f is the input data, g is the pooling function (e.g., sum), and h is the output of the pooling operation.
    * Average pooling a similar operation that computes the average of the input values within a given window. The formula for average pooling can be written as:

> Output = sum(Input[i:i+2, j:j+2]) / 4

where Output is the resulting value after the average pooling operation, and Input is the matrix of input values. The sum function calculates the sum of all values in the input window, and the division by 4 is used to compute the average (since the window size is 2x2 in this example).

    * Expectation Pooling
    
    *![image](https://user-images.githubusercontent.com/88499318/209716583-846e53d8-1421-4018-b99d-10ed52695df2.png)

 (A) The overall design of Expectation Pooling. (B) Equations describing how ePooling computes. (C) The output of ePooling is (up to a constant difference) exactly the log-likelihood of the corresponding PWM.


- **Fully connected layer:** A fully connected layer is a traditional neural network layer that connects all the neurons in the previous layer to all the neurons in the current layer. The fully connected layer performs a linear combination of the input data followed by an activation function. The formula for a fully connected layer is:

(![image](https://user-images.githubusercontent.com/88499318/209717287-4ebcc757-de9d-4bcc-8a5c-c97aabacfc50.png)

Each individual function consists of a neuron (or a perceptron). In fully connected layers, the neuron applies a linear transformation to the input vector through a weights matrix. A non-linear transformation is then applied to the product through a non-linear activation function f and taking the dot product between the weights matrix W and the input vector x. The bias term (W0) can be added inside the non-linear function.
The generator aims to minimize the objective, and the discriminator aims to maximize the objective.


#### RNNs

- **Recurrent layer:** A recurrent layer is a type of neural network layer that has feedback connections, allowing it to process sequential data. The recurrent layer performs a linear combination of the input data and the hidden state followed by an activation function. The formula for a recurrent layer is:

Recurrent formula varies

where *W* is the weight matrix, *x* is the input data, *h* is the hidden state, *b* is the bias vector, and *y* is the output of the recurrent layer.

- **LSTM layer:** An LSTM (long short-term memory) layer is a type of recurrent neural network that uses gating mechanisms to control the flow of information through the network. The LSTM layer performs a linear combination of the input data, the hidden state, and the cell state, followed by three activation functions. The formulas for an LSTM layer are too complex to include here, but you can find more information about LSTM layers in the literature.

- **GRU layer:** A GRU (gated recurrent unit) layer is a type of recurrent neural network that uses gating mechanisms to control the flow of information through the network. The GRU layer performs a linear combination of the input data and the hidden state, followed by two activation functions. The formulas for a GRU layer are too complex to include here, but you can find more information about GRU layers in the literature.

- **Peephole connections:** Peephole connections are additional connections in an LSTM layer that allow the gates to incorporate information from the cell state.

- **Activation function:** An activation function is a non-linear function applied to the output of a neural network layer to introduce non-linearity into the model. Common activation functions include sigmoid, tanh, ReLU, and softmax. The template formula for a sigmoid activation function is:

![image](https://user-images.githubusercontent.com/88499318/209719197-0ec67f2b-c010-47c6-bbd5-4f6bde341c77.png)

Activation functions are important in neural networks because they allow the model to learn non-linear relationships in the data. Different activation functions can have different properties and may be more or less suitable for different types of problems.

* Weight initialization: Weight initialization is the process of setting the initial values for the weights in a neural network. The choice of weight initialization can have an impact on the performance of the network and it is important to choose an initialization method that is suitable for the problem at hand. Common methods for weight initialization include random initialization, Glorot initialization, and He initialization.

* Backpropagation: Backpropagation is an algorithm for training neural networks that involves calculating the gradients of the loss function with respect to the model parameters and using these gradients to update the parameters in the direction that reduces the loss. The process of backpropagation involves propagating the error from the output layer back through the network, calculating the gradients at each layer, and updating the weights accordingly.

# Optimization algorithms 

* Gradient descent

![image](https://user-images.githubusercontent.com/88499318/209719765-46f818e6-d2e0-4047-8b84-eab4380c956a.png)

gₜ,ⱼ the derivative of the loss function with respect to the j-th weight during time step t. In the second equation, 𝜃ₜ,ⱼ is the j-th weight of the network at time step t, while the second term of the right-hand side is the partial derivative multiplied by a constant alpha. This alpha is called the learning rate and is just a scalar by which we multiply the derivative in order to make the step a little smaller (if we take too big a step, we risk jumping over minima and thus not improving our model at all). Choosing a good learning rate is important, but it’s usually done empirically (you can generally start with something like 0.001 or 0.0001 and tweak from there). Taking all this into account, we see that the formula is essentially the following: To adjust the weight, we take the old one and subtract from it the loss function’s derivative in respect to it multiplied by the learning rate to make sure we’re actually improving the model, not just jumping around.

* Stochastic gradient descent (SGD): Stochastic gradient descent (SGD) is an optimization algorithm for training neural networks that involves updating the model parameters based on the gradients of the loss function. In SGD, the gradients are calculated using a small batch of data (also called a mini-batch) rather than the entire dataset. The model parameters are updated using the following formula:

![image](https://user-images.githubusercontent.com/88499318/209719354-1b4f69a8-f859-444e-902d-d0a94765f831.png)

*  Adam optimization: Adam (Adaptive Moment Estimation) is a gradient-based optimization algorithm for training neural networks that combines the ideas of SGD and momentum optimization. Adam uses an exponentially decaying average of the gradients and the squared gradients to adapt the learning rate for each parameter. The Adam optimization algorithm changes by appplication and build, some examples:

![image](https://user-images.githubusercontent.com/88499318/209719568-25904de6-3f03-4170-bbf8-34408d89e07b.png)


Even though it has good training time, Adam in some areas does not converge to an optimal solution, so for some tasks (such as image classification on popular CIFAR datasets) state-of-the-art results are still only achieved by applying SGD with momentum.

![image](https://user-images.githubusercontent.com/88499318/209719581-bf735cde-50b9-44c8-bcf2-e33c88e5b245.png)
