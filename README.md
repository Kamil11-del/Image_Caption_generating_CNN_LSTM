# Image_Caption_generating_CNN_LSTM
This repository is about generating caption for image while  detecting features in it using CNN and LSTM
Here is a formal and easy-to-understand description for the README file of your image captioning model:

---

## Image Captioning Model

This repository contains an implementation of an image captioning model using deep learning techniques. The model generates textual descriptions for images by combining Convolutional Neural Networks (CNNs) with Long Short-Term Memory (LSTM) networks.

### Overview

The image captioning model is designed to:
1. Extract features from images using a pre-trained CNN (Xception).
2. Generate captions for these images by leveraging an LSTM-based sequence model trained on the extracted features.

### Dataset

The dataset used for training and evaluating the model consists of images and corresponding captions. You can access and download the dataset from Kaggle:

[Image Captioning Dataset on Kaggle](https://www.kaggle.com/datasets/muhammmadkamilkhan/imagecaptiondataset)

### Model Description

1. **Feature Extraction**:
   - Utilizes the Xception model to extract features from images. The Xception model, with its pre-trained weights, is used to convert images into a feature vector representation.

2. **Caption Generation**:
   - Employs an LSTM-based model to generate captions. The model combines features from the CNN with sequential data to predict the next word in the caption sequence.

### Components

- **`extract_features(directory)`**: Extracts image features from a specified directory using the Xception model.
- **`create_sequences(tokenizer, max_length, desc_list, feature)`**: Creates input sequences for training by processing descriptions and image features.
- **`data_generator(descriptions, features, tokenizer, max_length)`**: Generates batches of data for training the model.
- **`define_model(vocab_size, max_length)`**: Defines and compiles the image captioning model architecture.
- **`generate_desc(model, tokenizer, photo, max_length)`**: Generates a textual description for a given image.

### Usage

1. **Prepare the Data**:
   - Download the dataset and place the image and text files in their respective directories.
   - Load and preprocess the data to extract features and clean descriptions.

2. **Train the Model**:
   - Train the model using the provided data generator and save the trained model checkpoints.

3. **Generate Captions**:
   - Use the trained model to generate captions for new images by extracting features and predicting descriptions.

### Installation

To set up the environment, ensure you have the necessary libraries installed. You can install the required libraries using `pip`:

```bash
pip install tensorflow keras numpy pillow
```

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
