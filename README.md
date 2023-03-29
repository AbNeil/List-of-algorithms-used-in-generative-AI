# Generative Deep Neural Network models and their categories:

## Text-to-text

LLMs like Google Bard, Bing, and Stanford's [Alpaca](https://alpaca-ai-custom3.ngrok.io/) trimmed from [LLaMa](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) trained on 65 Billion parameters yet outperforming [Minerva's PALM](https://arxiv.org/pdf/2206.14858.pdf), the largest language model trained on 540 Billion parameters. This means that it is possible to train state-of-the-art language models using publicly available datasets exclusively, without resorting to proprietary and inaccessible datasets even LLaMa13B outperforms GPT3.

Visual Foundation Models like MS [Visual GPT](https://arxiv.org/abs/2303.04671) (Generative Pre-trained Transformer) use stable diffusion with controlnets and visual transformers.

- [PaLM-E](https://arxiv.org/abs/2303.03378) (Pathways Language Model with Embodied) uses multi-modal information (text, images, audio, and video. Sensor data like temperature or humidity, location data aka GPS coordinates, and time-series data like stock prices or weather patterns).

## Text-to-robotics

Do As I Can, Not As I Say, aka [SayCan](https://say-can.github.io/) introduces using a grounding model to add physical context in robotics. It uses an LLM and affordance functions to select among a pre-defined set of primitives. This pre-defined set of primitives enables SayCan to use the so-called "scoring-mode" of the LLM to calculate the probability of the textual representation of the primitive in relevance to the high-level task. However, this requirement of exhaustive enumeration of all possible primitives limits the applicability of SayCan in scenarios with many possible skills, such as open-vocabulary or combinatorial tasks. [Grounded decoding](https://grounded-decoding.github.io/), which was recently introduced, does this through its flexible and expressive token-level grounding.

#### Text-to-image

[Stable diffusion](https://jalammar.github.io/illustrated-stable-diffusion/), imagen, midjourney work using and encoder netwwork like frozen CLIP and classifier free guidance, in the latent space: UNet with a scheduler to draw by detecting edges and curves in a preset or randomly generated "seed" noisy image and expands out of latent space using a variational autoencoder decoder that cleans the images of any remaining noise, controlnets add a versatile method of guiding the diffusion

#### Image-to-Text 

[VisualGPT](https://arxiv.org/abs/2102.10407) attempts to bridge the semantic gap between different modalities, a novel encoder-decoder attention mechanism is designed with an unsaturated rectified gating function it outperforms the best baseline model by up to 10.8% CIDEr on MS COCO and up to 5.4% CIDEr on Conceptual Captions, and achieves the state-of-the-art result on IU X-ray, a medical report generation dataset.Critically, the biggest advantage of this model is that it does not need for as much data as other image-to-text models.

[BLIP](https://github.com/salesforce/BLIP) by salesforce for image classsification.

#### Text-to-Video

[Make-A-Video](https://makeavideo.studio/)

[Phenaki](https://phenaki.video/): by Google, can generate realistic videos from a sequence of textual prompts. It can be accessed via its API on GitHub, and is the first model that can generate videos from open domain time variable prompts. It achieves this by jointly training on a large image-text pairs dataset and a smaller number of video-text examples, resulting in generalization beyond what is available in video datasets.

Two open source demo models [CogVideo](https://github.com/THUDM/CogVideo) by a groups of cs students and [a model by Antonia Antonova](https://antonia.space/text-to-video-generation) 
 
#### Video-to-text

[Google Muse](https://muse.ai/)

[Soundify](https://research.runwayml.com/soundify-matching-sound-effects-to-video): by Runway, is a system that matches sound effects to video for professional video editing. It uses quality sound effects libraries and a neural network with zero-shot image classification capabilities (CLIP) to classify, synchronize, and mix sound effects with a video. The video is split based on color histogram distances to reduce distinct sound emitters, and intervals are identified by comparing the effects label with each frame and pinpointing consecutive matches above a threshold. Effects are then split into one-second chunks and stitched via crossfades.

#### Image(s)-to-Video

[Dreamix](https://dreamix-video-editing.github.io/) by google can take a single image or many of a subject for live textual inversion

#### Video-to-video

[Text to live](https://text2live.github.io/) augments the scene with new visual effects based on input text, Dreamix can do this as well

####  Text-to-3Dvideo

[MAV3](https://make-a-video3d.github.io/) by Facebook uses a 4D dynamic Neural Radiance Field (which is just a way of saying it can make videos look really realistic and believable. It does this by using a special model that can understand both text and pictures, and can make videos that look like they're in different places and angles, the best part is it doesn't need any 3D or 4D data to work), it is optimized for a scene's "appearance, density, and motion consistency" by querying a t2v diffusion model. The dynamic video output generated from the provided text can be viewed from any camera location and angle, and can be composited into any 3D environment. MAV3D does not require any 3D or 4D data and the T2V model is trained only on Text-Image pairs and unlabeled videos. 

####  Text-to-3Dmodel
 
[Dreamfusion](https://dreamfusion3d.github.io/): DreamFusion is a text-to-3D model developed by Google that uses a pretrained 2D text-to-image diffusion model to perform textto-3D synthesis. In particular, Dreamfusion replaces previous CLIP techniques with a loss derived from distillation of a 2D diffusion model. Concretely, the diffusion model can be used as a loss within a generic continuous optimization problem to generate samples. Critically, sampling in parameter space is much harder than in pixels as we want to create 3D models that look like good images when rendered from random angles. To solve the issue, this model uses a differentiable generator. 

[Magic3D](https://research.nvidia.com/labs/dir/magic3d/) is another text to 3D model made by NVIDIA.

#### Image-to-3Dmodel

[Get3d](https://github.com/nv-tlabs/GET3D) and the more streamlined [Instant NGP](https://github.com/NVlabs/instant-ngp) are two open source CUDA models for generating a 3d map of an image(s) [made by NVidia](https://nvlabs.github.io/instant-ngp/)

#### Text-to-Audio

[MusicLM](https://google-research.github.io/seanet/musiclm/examples/): by Google for high-quality music generation.


[Jukebox](https://openai.com/blog/jukebox/): a neural net that generates music, including rudimentary singing, as raw audio in a variety of genres and artist styles. We’re releasing the model weights and code, along with a tool to explore the generated samples.

[AudioLM](https://google-research.github.io/seanet/audiolm/examples/): by Google for high-quality audio generation with long-term consistency.

VALL-E (neural codec language model) based on Meta's EnCodec, trained for speech-synthesis capabilities on the audio library LibriLight which contains 60,000 hours of English language speech from more than 7,000 speakers, curated from LibriVox. It can allegedly simulate anyone’s voice with 3 seconds of audio input.

[nvidia Riva](https://www.nvidia.com/en-us/ai-data-science/products/riva/) is a GPU-accelerated automatic speech recognition (ASR) SDK for building fully customizable, real-time pipelines and deploying them in clouds, in data centers, at the edge, or on embedded devices.
 
#### Audio-to-text

Shazam and google search using Discrete FFT fast fourier transforms for audio classsification

####  Audio to Audio

[SingSong](https://storage.googleapis.com/sing-song/index.html) Generates Accompanying Music for Singing.

* Audio to Midi

[Spotify Basic Pitch](https://basicpitch.spotify.com/) 


## Deep Neural Networks 

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

#### GANs (Generative adversarial networks)

* Generator network
* Discriminator network
* Loss function
* Optimization algorithm (e.g., SGD, Adam)
* Minimax game

#### Transformer models

* Relative positional encoding
* Adaptive softmax
* Permutation language modeling
* Span-based language modeling
* Reversible transformer
* Adaptive attention span
* Dynamic convolutional layers

#### BERT (Bidirectional Encoder Representations from Transformers)

* Multi-layer bidirectional transformer
* Self-attention
* Multi-head attention
* Positional encoding
* Feedforward networks
* Dropout
* Adam optimization
* Bidirectional training (specific to BERT)
* WordPiece embeddings (specific to BERT)
* Masked language modeling (specific to BERT)
* Next sentence prediction (specific to BERT)
* Fine-tuning (specific to BERT)


#### Visual Foundation Models 

* ControlNets: MLSD Canny Depth and openpose
* Visual Transformers 
* Latent Diffusion Models: i.e Stable


 ##### U-Nets which use:
* Encoder network
* Decoder network
* Skip connections
* Loss function
* Optimization algorithm (e.g., SGD, Adam)
* Up-sampling layer



## Paradigms with algorithms

#### Online learning
#### Semi-supervised learning
#### Self-supervised learning
#### Batch learning

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

* Autoencoders
* Factor analysis
* K-means clustering
* Hierarchical clustering
* Self-organizing maps (SOMs)
* Expectation-maximization (EM)
* Principal component analysis (PCA)
* Singular value decomposition (SVD)
* Independent component analysis (ICA)
* [Markov Chain Monte Carlo](https://jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/) (MCMC)
* Deep belief networks (DBNs)
* Hidden Markov models (HMMs)
* Markov random fields (MRFs)
* Deep Belief Networks (DBNs) consist of two different types of neural networks 
    * Restricted Boltzmann machines (RBMs) a type of generative stochastic ANN that can learn a probability distribution over its set of inputs. They are often used for dimensionality reduction, feature learning, and collaborative filtering.
    * Belief Network aka Bayesian Network an acyclic directed graphs (DAG) where the nodes are random variables. It defines a factorization of the joint probability distribution where the conditional probabilities form factors that are multiplied together. Bayesian/Belief Networks capture both conditionally dependent and conditionally independent relationships between random variables and compactly specify their joint distributions
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

# Explanation with formulas, equations or processes

#### Learning analysis 
### Supervised learning

 Linear regression: When you have 2 variables, an independent variable (lets call it X) and the dependent variable (lets call it Y). Linear regression is a test to do 2 things:

 * To see how closely related are these two variables, applying linear regression gives us a number between -1 and 1 that gives us an indication of the the strength of correlation between the two. 0 means they aren't related. 1 means they are positively correlated (an increase in X means an increase in Y). -1 means negatively correlated (increase in X means a decrease in Y and vice versa) it is easy to extend to more than 2 variables. The only difference is instead of line you will get a plane in 3 and a hyper plane in 4+D.

 * For prediction. If we know the rough relationship between X and Y, then we can use this relationship to predict values of Y for a value of X we want.

For example: Lets say X is the number of workers in a painting job and Y is the amount of time needed to finish a job. You do several jobs with different numbers of workers and you time how much it takes to finish each job.

We put those pretty numbers in a graph and do a simple linear regression and we learn 2 things: Does increasing the number of workers really decrease the time needed to finish a job? (i.e. are they correlated and how much). Second, if we get a customer who wants the job done in a very short time, then we can use our study to predict how many workers it might need to finish it.

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

Here’s an analogy to help you understand the differences between AdaGrad, Adam and SGD with Momentum:

Imagine you are a hunter-gatherer searching for food in a vast landscape. The food represents the minimum of the loss function that you want to find.

**SGD with Momentum**: This is like having a map and a compass to guide you towards the food. You take steps in the direction that your compass (gradient) points and also take into account your previous step (momentum) to help you move faster towards your goal.

**AdaGrad**: This is like having a map that adapts to the terrain as you explore it. The map helps you navigate more effectively through difficult terrain (high curvature regions) by adjusting your step size based on past observations.

**Adam**: This is like having both an adaptive map and a compass with momentum. Adam combines the best properties of AdaGrad and SGD with Momentum to help you navigate more effectively through difficult terrain while also taking into account your previous steps to move faster towards your goal.

![image](https://user-images.githubusercontent.com/88499318/209719581-bf735cde-50b9-44c8-bcf2-e33c88e5b245.png)

Markov Chain Monte Carlo (MCMC) is a class of algorithms used for sampling from a high-dimensional probability distribution. The basic idea is to use a Markov chain to explore the state space of the distribution, with the stationary distribution of the chain being the target distribution. The algorithm iteratively generates samples from the chain, with each sample providing an approximation of the target distribution. Some common examples of MCMC algorithms include Metropolis-Hastings, Gibbs sampling, and the Hamiltonian Monte Carlo.

Metropolis-Hastings algorithm: This is one of the simplest MCMC algorithms and involves proposing a new state based on the current state, and accepting or rejecting the proposal based on a random acceptance/rejection rule.

Gibbs sampling: This algorithm is used when the target distribution can be decomposed into a series of conditionally independent distributions. The algorithm samples from each of these distributions in turn to obtain a sample from the target distribution.

Hamiltonian Monte Carlo (HMC): This is a more advanced MCMC algorithm that takes into account the gradient information of the target distribution. It uses gradient information to propose a new state in such a way that the acceptance rate is high, leading to faster convergence.

Slice sampling: This is a simple and efficient MCMC algorithm that can be used when the target distribution is univariate (one-dimensional). It works by sampling uniformly from a slice of the target distribution.

Each of these sub-algorithms has its own advantages and disadvantages, and the choice of which one to use depends on the problem at hand.

![image](https://user-images.githubusercontent.com/88499318/224458032-6524d0f5-96a7-4451-847b-a02ea91a86d0.png)

# Misc

### Transformer Models

- XLNet ([eXtreme MultiLingual Language Understanding](https://github.com/zihangdai/xlnet))
- T5 ([Text-to-Text Transfer Transformer](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html))
- RoBERTa ([Robustly Optimized Bidirectional Encoder Representations from Transformers Approach](https://arxiv.org/abs/1907.11692))
- ELECTRA ([Efficiently Learning an Encoder that Classifies Token Replacements Accurately](https://openreview.net/pdf?id=r1xMH1BtvB))
