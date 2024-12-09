# Gaussian Process Multi-Dynamical Models for Human Activity Recognition

This repository implements a variant of Gaussian Process Dynamical Models (GPDMs) that learns multiple dynamics functions in a shared latent space. We combine this model with a particle filter to perform human activity recognition from human skeletal joint data. To test the model, I apply it to walking and running data from the CMU Motion Capture Database. In future work, this approach could be applied to more complex datasets, including datasets which include switching between actions.

## Table of Contents
- [Demo](#demo)
- [Repository Structure](#repository-structure)
- [Brief Technical Overview](#brief-technical-overview)
  - [Gaussian Process Multi-Dynamical Models]($gaussian-process-multi-dynamical-models)
  - [Particle Filtering for Real-Time Classification]($particle-filtering-for-real-time-classification)
- [Results](#results)

## Demo
![Demo of my model for a running trial](https://raw.githubusercontent.com/Priyanshu4/gpmdm/refs/heads/demo-gif/demo.gif)
The GIF above shows a demonstration of my model for a test trial corresponding to running. On the left, we see the motion capture data from the running person. 
On the right, we see the latent space. The blue points are the training data points for walking. The red points are the training data points for running. 
Each point in the latent space corresponds to a set of joint angles for the human skeleton. The green trajectory shows the mean of the particle filter's estimate of the latent state for the person on the left.
As we can see, the particle filter's estimate closely follows the training data for running. 

## Repository Structure
- `amc_parser`: Utilities to parse and view .amc files from the CMU Mocap Database. Adapted from https://github.com/CalciferZh/AMCParser with significant additions.
- `dataset_utils`: Test and training set split as well as other utilities for the dataset.
- `mocap`: Not included in the repository. To recreate results, download all CMU Mocap Database AMC files and place in this folder.
- `gpmdm`: Python package with core models and algorithms.
    - `gpmdm.py`: PyTorch implementation of Gaussian Process Multi-Dynamical Models. Adapted from https://github.com/fabio-amadio/cgpdm_lib with significant modifications.
    - `gpmdm_pf.py`: Implementation of a particle filter specifically designed for latent state estimation and classification with GPMDMs.
- `notebooks`: Jupyter notebooks with training and test code for the Mocap dataset.
    - `train_gpmdm.ipynb`: Train the model.
    - `test_gpmdm_pf.ipynb`: Recreate my results.
    - `view_gpmdm_pf.ipynb`: Recreate the animation showed in the demo.
 
### Dependencies
torch, torchtyping, pandas, numpy, sklearn, matplotlib, plotly 

## Brief Technical Overview

### Gaussian Process Multi-Dynamical Models
In brief, Gaussian Process Dynamical Models (Wang et. al, 2007) are a non-parametric generative Bayesian machine learning model for time-series data. 
After training by minimizing negative log likelihood using gradient descent, they provide:
- A latent representation of the training observations
- A gaussian process mapping from latent space to observation space
- A gaussian process mapping from each latent state to the next latent state (Markovian dynamics in the latent space).

In this repository, I extend GPDMs by allowing them to learn multiple dynamics mappings for each class of data. The latent representation and mapping to observation space is not modified. This means all classes are embedded in a shared latent space, but each has their own dynamics. This is done by modifying the gaussian process kernel matrix for the dynamics mapping such that entries which correlate data points from different classes are set to 0. This is what I refer to as a Gaussian Process Multi-Dynamics Model (GPMDM).

### Particle Filtering for Real-Time Classification
After we have trained the GPMDM on data, we can use it to classify between actions. However, GPDMs do not give us a mapping from observation space to latent space. Therefore, use particle filtering to infer the latent state and class of the system. 

Each particle i at time t is represented by tuple (x_t, c_t, w_t) where:
- c_t: class of the system
- x_t: latent state of the system
- w_t: weight of the particle
  
At each time step, we receive a new observation z_t and every particle is updated:
- c_t is sampled from T * c_{t-1}, where T is transition matrix between classes
- x_t is sampled from the dynamics GP: x_t ~ dynamics_gp(x_{t-1}, c_t) 
- w_t ∝ P(z_t | x_t), which is computed by propogating particles through the observation GP

We then resample the particles randomly according to their weight. Particles with high weight will have multiple copies.

The particle filter gives us a probability distribution over the latent state and class at each timestep. We use the highest probability class as the predicted class.

## Results
The model was trained on 19 total trials (2 to 5 seconds each) of walking and running data. The trials are recorded at 120 fps in the original dataset, but all data I used was downsampled to 30 fps. Although my dataset had no examples of switching between actions, in testing I used a transition matrix with a 10% chance of switching at each timestep. With 100 particles in the particle filter, the model achieved greater than 0.90 accuracy and F1-score on a per-frame basis. On a per trial basis, only one trial was misclassified out of 39 (misclassification means less than 50% accuracy for a trial). With 100 particles, the classification ran at 10-15 fps on my laptop CPU. Since we are using particle filters, there is some variability to the results. Increasing the number of particles should in theory increase the performance but also increase inference time. Adaptive particle filtering or other techniques should also improve the performance. Furthermore, using more training data should improve performance, but will increase both training and inference time. The approach demonstrated promising results on the simple case of running versus walking and could be applied to more complex problems in future results. The approach is general to time-series classification and prediction. 

