# Convolutional Neural Networks
Joint work with Zuzanna Kotlińska on the Computer Vision project as part of the Deep Learning course at Warsaw University of Technology.

## Dataset

The data contains 9000 examples of each of the following ten classes: *airplane, automobile, bird, cat, deer, dog, frog, horse, ship truck*.
Dataset is availble on Kaggle under this [link](https://www.kaggle.com/datasets/eclaircat/cinic-eclair-test?resource=download).

## Task

The aim of this project is to design and compare various Deep Learning architectures to tackle the problem of image classification (mutliclass).

## Instruction
Install all necessary libraries, specified in *requirements.txt*.

Download the dataset and place it in the *data/* dir in the root of the project.

To obtain the results, simply run one of the following Python scripts:
* `python cnn.py` - specify the variant of the architecture by providing the model type ("cnn", "cnn2", "cnn3"),
* `python resnet.py` - fine-tune pretrained version of ResNet50 on the CINIC image classification task,
* `python ensemble.py` - train ensemble network, consisting of several CNNs.

