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
## Data Preprocessing and Normalization

### Transformations

For effective training of the neural network, we apply a series of preprocessing transformations to the images in the FER 2013 dataset. These transformations are crucial for normalizing the input data and enhancing the model's ability to generalize across different facial expressions under various lighting and background conditions. Here is a detailed breakdown of each transformation:

1. **Grayscale Conversion**: 
   - Each image is converted to grayscale to reduce the computational complexity and to focus the model on learning textural and structural patterns rather than color information which is less relevant for emotion recognition.

2. **Tensor Conversion**:
   - Images are converted from PIL format to tensors to facilitate operations in PyTorch which is the framework used for developing the model. This conversion is essential for enabling batch processing of images during model training.

3. **Random Rotation**:
   - This involves randomly rotating the images by up to 20 degrees. This augmentation helps the model become robust to variations in head posture.

4. **Color Jitter**:
   - We apply random adjustments to brightness, contrast, and saturation with a factor of 0.5. This technique improves the model's tolerance against different lighting conditions and photographic qualities.

5. **Normalization**:
   - The pixel values of images are standardized based on the mean and standard deviation of the dataset. This normalization helps in speeding up the learning process by ensuring that the input values (pixel intensity values) have similar data distributions, which optimizes gradient descent algorithms used during training.

```python
final_transform = lambda mean, std: transforms.Compose([
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean.tolist(), std.tolist())
])


## Model Training

The training script can automatically detect whether to use a CPU or a GPU (with CUDA available). It will utilize the GPU if available, which significantly speeds up the training process.

### Running the Training Script

To start the training process, run the `train.py` script located in the root directory:

```bash
python train.py
