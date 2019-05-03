## Keras-GAN
Collection of Keras implementations of Generative Adversarial Networks (GANs) suggested in research papers. These models are in some cases simplified versions of the ones ultimately described in the papers, but I have chosen to focus on getting the core ideas covered instead of getting every layer configuration right. Contributions and suggestions of GAN varieties to implement are very welcomed.

Note that this repo is folked from https://github.com/eriklindernoren/Keras-GAN.
And is modified for experiments in I/O scheduling algorithms.

## Table of Contents
  * [Installation](#installation)
  * [Implementations](#implementations)
    + [Conditional GAN](#cgan)
    + [Deep Convolutional GAN](#dcgan)
    + [Generative Adversarial Network](#gan)

## Installation
    $ git clone https://github.com/xxfwin/Keras-GAN
    $ cd Keras-GAN/
    $ sudo pip3 install -r requirements.txt

## Implementations   
### CGAN
Implementation of _Conditional Generative Adversarial Nets_.

[Code](cgan/cgan.py)

Paper:https://arxiv.org/abs/1411.1784

#### Example
```
$ cd cgan/
$ python3 cgan.py
```

<p align="center">
    <img src="http://eriklindernoren.se/images/cgan.gif" width="640"\>
</p>



### DCGAN
Implementation of _Deep Convolutional Generative Adversarial Network_.

[Code](dcgan/dcgan.py)

Paper: https://arxiv.org/abs/1511.06434

#### Example
```
$ cd dcgan/
$ python3 dcgan.py
```

<p align="center">
    <img src="http://eriklindernoren.se/images/dcgan2.png" width="640"\>
</p>


### GAN
Implementation of _Generative Adversarial Network_ with a MLP generator and discriminator.

[Code](gan/gan.py)

Paper: https://arxiv.org/abs/1406.2661

#### Example
```
$ cd gan/
$ python3 gan.py
```

<p align="center">
    <img src="http://eriklindernoren.se/images/gan_mnist5.gif" width="640"\>
</p>

