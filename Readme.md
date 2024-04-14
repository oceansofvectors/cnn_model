# Facial Expression Recognition using ResNet

This repository contains a Convolutional Neural Network (CNN) model that utilizes ResNet architecture to predict facial expressions. The model is trained on the FER 2013 dataset and achieves an accuracy of `60%`. While this may seem low, it is significant considering the complexity of the dataset, with state-of-the-art models only achieving around `65%` accuracy.

## About the FER 2013 Dataset

The FER 2013 dataset was created for the Facial Expression Recognition Challenge, and it contains images generally categorized into 7 classes: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, and 6=Neutral. The images are all grayscale, 48x48 pixels, and the dataset is notoriously challenging due to the variability in emotion representation across different faces.

## Prerequisites

Before running the model training script, you need to set up your environment properly:

1. **Python Environment**: Make sure you have Python installed, along with the libraries `torch`, `torchvision`, `numpy`, and `matplotlib`.

2. **Data Preparation**:
   - Download the `data.tgz` file containing the FER 2013 dataset.
   - Unpack the dataset to the `data/` directory in the root of this project repository. You can use the following command:
     ```bash
     tar -xvzf data.tgz -C data/
     ```

## Model Training

The training script can automatically detect whether to use a CPU or a GPU (with CUDA available). It will utilize the GPU if available, which significantly speeds up the training process.

### Running the Training Script

To start the training process, run the `train.py` script located in the root directory:

```bash
python train.py
