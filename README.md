# Deep Learning Model Evaluation & Calibration

This repository contains the implementation and analysis of Model Calibration for Convolutional Neural Networks (CNNs). The project investigates the reliability of confidence scores in deep learning models and applies Temperature Scaling to mitigate overconfidence.

**Project Report:** [Project Report (PDF)](./DL_Project.pdf)

## Project Goal
In safety-critical applications (e.g., autonomous driving, medical diagnosis), high accuracy is not enough; models must also be calibrated. A calibrated model's predicted confidence should reflect its true probability of correctness.

This project analyzes:

**Architectures:** ResNet-18 vs. DenseNet-121

**Datasets:** CIFAR-10 vs. CIFAR-100

**Metrics:** Expected Calibration Error (ECE) & Negative Log Likelihood (NLL)

## Tech Stack & Setup
**Framework:** PyTorch

**Hardware:** NVIDIA GTX 1650 Ti (Local Machine)

**Training Duration:** 10 Epochs (Limited for experimental observation)

## Installation
**Clone the repository:** git clone

**Install dependencies:** pip install torch torchvision matplotlib numpy

## How to Run
To train the models, visualize results, and apply calibration, simply run the main script:

python main_project.py

**The script will:**

Download CIFAR-10/100 datasets automatically.

Train ResNet-18 and DenseNet-121 models.

Evaluate uncalibrated performance (ECE, NLL).

Apply Temperature Scaling using the validation set.

Generate Reliability Diagrams in the proje_sonuclari folder.

## Key Results (Summary)
The experiments revealed that Temperature Scaling provides marginal improvements when models are trained for a limited duration (10 epochs). This confirms the hypothesis that severe overconfidence typically emerges after prolonged training (overfitting).

Note: Since T > 1.0 in all cases, the models exhibited mild overconfidence even at early training stages.

## Repository Structure
DL_Project.pdf : Detailed academic report

main_project.py : Main source code for training & calibration

proje_sonuclari/ : Output folder for graphs and logs

README.md : Project documentation

.gitignore : Config to exclude large dataset files

## Author

**Emre Saygın**

**Student ID: 120200069**

**Course: CMPE 460 / Deep Learning**
