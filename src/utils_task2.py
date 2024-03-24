import os
import numpy as np
import tensorflow as tf
from src.utils_task1 import decode_info
from sklearn.model_selection import train_test_split


def extract_label_and_img(tfrecord_folder_path:str):
    '''
    Extracts images and labels from TFRecord files.

    Args:
        tfrecord_folder_path (str): Path to the folder containing TFRecord files.

    Returns:
        tuple: A tuple containing two lists: images and labels. 
            The images list contains decoded images, and the labels list contains corresponding labels.
    '''
    
    images = []
    labels = []
    for tfr_path in os.listdir(tfrecord_folder_path):
        for tfr_string in tf.data.TFRecordDataset(os.path.join(tfrecord_folder_path, tfr_path)).map(decode_info):
            
            filename = tfr_string['image/filename'].numpy().decode('utf-8')
            label_join = filename.split(".")[0].split("_")[:-1]
            label = "_".join(label_join)
            
            image_raw =tfr_string['image/encodedrawdata']
            decoded_image = tf.io.parse_tensor(image_raw, out_type=tf.uint8).numpy()
            
            decoded_image = decoded_image / 255.0           # Normalize
            
            images.append(decoded_image)
            labels.append(label)
    
    return images, labels

def transform_label_to_OneHotEncoded(labels:list, out_classes:int):
    '''
    Transforms categorical labels into one-hot encoded vectors.

    Args:
        labels (list): A list containing categorical labels.
        out_classes (int): The number of output classes for one-hot encoding.

    Returns:
        numpy.ndarray: A numpy array containing one-hot encoded labels.
        
    Raises:
        ValueError: If the number of converted classes does not match the expected number of classes.
    '''
    
    label_to_index = {}
    index = 0
    for label in labels:
        if label not in label_to_index:
            label_to_index[label] = index
            index += 1
    numeric_labels = [label_to_index[label] for label in labels]

    # One-Hot-Encoding
    num_classes = len(label_to_index)
    one_hot_labels = np.eye(num_classes)[numeric_labels]
    
    if not len(label_to_index.keys()) == out_classes:
        converted_classes = len(label_to_index.keys())
        raise ValueError(f"Conversion failed: The number of converted classes ({converted_classes}) does not match the expected number of classes ({out_classes}).")
    else:
        print(f"Info:\t\tConvertion to One-Hot-Encoding was succsesfull")

    return one_hot_labels

def transform_label_to_SparseCategorical(labels):

    label_to_index = {}
    index = 0
    for label in labels:
        if label not in label_to_index:
            label_to_index[label] = index
            index += 1
    
    numeric_labels = [label_to_index[label] for label in labels]
    print(f"Info:\t\tConvertion to Sparse-Encoding was succsesfull")
    
    return np.array(numeric_labels)

def create_train_and_validation_dataset(images:list, one_hot_labels:list, val_size:int, batch_size:int):
    '''
    Creates training and validation datasets from images and labels.

    Args:
        images (list): A list containing images.
        one_hot_labels (list): A list containing one-hot encoded labels.
        val_size (int): The size of the validation set.
        batch_size (int): Batch size for the datasets.

    Returns:
        tuple: A tuple containing training and validation datasets.
    '''
    
    train_images, val_images, train_labels, val_labels = train_test_split(images, one_hot_labels, test_size=val_size, random_state=42)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))

    train_dataset = train_dataset.shuffle(buffer_size=len(train_images)).batch(batch_size)
    val_dataset = val_dataset.shuffle(buffer_size=len(val_images)).batch(batch_size)
    
    return train_dataset, val_dataset

def create_test_dataset(images:list, one_hot_labels:list, batch_size:int):
    '''
    Creates a test dataset from images and labels.

    Args:
        images (list): A list containing images.
        one_hot_labels (list): A list containing one-hot encoded labels.
        batch_size (int): Batch size for the dataset.

    Returns:
        tf.data.Dataset: A TensorFlow dataset for testing.
    '''
    
    test_dataset = tf.data.Dataset.from_tensor_slices((images, one_hot_labels))
    test_dataset = test_dataset.shuffle(buffer_size=len(images)).batch(batch_size)
    
    return test_dataset
