Image Classifier Project

This project is an image classification application that utilizes a pre-trained deep learning model to predict the class of an input image. The model is built using TensorFlow and TensorFlow Hub.

Features

Load a pre-trained TensorFlow model

Process images to match the model's input format

Predict the top-k most probable classes for an image

Map predicted class indices to human-readable labels

Requirements

Ensure you have the following dependencies installed:

pip install tensorflow tensorflow-hub numpy matplotlib pillow argparse

Files

predict.py: A script to load a model and classify an image.

Project_Image_Classifier_Project.ipynb: Jupyter notebook for training and testing the image classifier.

label_map.json: JSON file mapping class indices to human-readable names (expected input for predict.py).

Usage

Running Prediction Script

python predict.py <image_path> <model_path> --top_k <k> --category_names <category_file>

Arguments

<image_path>: Path to the image to be classified.

<model_path>: Path to the saved TensorFlow model.

--top_k <k> (optional): Number of top predictions to return (default is 5).

--category_names <category_file> (optional): JSON file mapping class indices to class names (default is label_map.json).

Example

python predict.py test_image.jpg my_model.h5 --top_k 3 --category_names label_map.json

Output Example

[[0.85 0.10 0.05]]
[[15 27 8]]

The first array represents the probabilities of the top-k predictions, and the second array contains their corresponding class indices.

Notes

The model should be trained and saved as a .h5 file before using predict.py.

Ensure that label_map.json correctly maps model output indices to class names.

License

This project is open-source and available for modification and distribution.

