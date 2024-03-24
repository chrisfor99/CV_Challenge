import argparse
import os
import glob
import csv

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from scipy.interpolate import  interp1d
from scipy.signal import savgol_filter

def create_run_directory(project_name):
    base_dir = os.path.join('train_logs', project_name)
    os.makedirs(base_dir, exist_ok=True)  # Create project directory if it doesn't exist
    run_dirs = glob.glob(os.path.join(base_dir, 'runs*'))  # Find all existing run directories
    next_run_number = len(run_dirs)  # Determine the next run number
    run_dir = os.path.join(base_dir, f'runs{next_run_number}')  # Create new run directory name
    os.makedirs(run_dir)  # Create the run directory
    return run_dir


def initialize_logging(run_dir):
    log_path = os.path.join(run_dir, 'training_log.csv')
    with open(log_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'Train Accuracy', 'Validation Accuracy'])
    return log_path

def update_log(log_path, epoch, train_loss, val_loss, train_accuracy, val_accuracy):
    with open(log_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, train_loss, val_loss, train_accuracy, val_accuracy])

def plot_metrics(log_path, run_dir):
    epochs, train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], [], []
    with open(log_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            epochs.append(int(row[0]))
            train_losses.append(float(row[1]))
            val_losses.append(float(row[2]))
            train_accuracies.append(float(row[3]))
            val_accuracies.append(float(row[4]))

    # Minimum number of points for cubic spline
    # Define the window length (must be odd) and the polynomial order for the Savitzky-Golay filter
    window_length = 5  # Adjust based on your data; should be less than the number of data points and odd
    poly_order = 3  # Polynomial order; adjust based on your needs

    # Create smoothed curves if we have enough data points; otherwise, use the original data
    if len(epochs) >= window_length:  # Ensure we have enough points for the Savitzky-Golay filter window
        epochs_smooth = epochs
        # Apply Savitzky-Golay filter for smoothing
        train_accuracies_smooth = savgol_filter(train_accuracies, window_length, poly_order)
        val_accuracies_smooth = savgol_filter(val_accuracies, window_length, poly_order)

        train_losses_smooth = savgol_filter(train_losses, window_length, poly_order)
        val_losses_smooth = savgol_filter(val_losses, window_length, poly_order)
    else:
        # Fallback: Use linear interpolation or simply plot the original points without smoothing
        epochs_smooth = epochs
        
        # Linear interpolation as a simple fallback
        train_accuracies_smooth = interp1d(epochs, train_accuracies, kind='linear')(epochs_smooth)
        val_accuracies_smooth = interp1d(epochs, val_accuracies, kind='linear')(epochs_smooth)

        train_losses_smooth = interp1d(epochs, train_losses, kind='linear')(epochs_smooth)
        val_losses_smooth = interp1d(epochs, val_losses, kind='linear')(epochs_smooth)

    # Plotting
    plt.figure(figsize=(10, 5))

    # Plot for Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, alpha=0.3, color='#FF5733')
    plt.plot(epochs, val_losses, alpha=0.3, color='blue')
    plt.plot(epochs_smooth, train_losses_smooth, label='Train Loss Smooth',color='#FF5733')
    plt.plot(epochs_smooth, val_losses_smooth, label='Validation Loss Smooth',color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--')

    # Plot for Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, alpha=0.3, color='#FF5733')
    plt.plot(epochs, val_accuracies, alpha=0.3, color='blue')
    plt.plot(epochs_smooth, train_accuracies_smooth, label='Train Accuracy',color='#FF5733')
    plt.plot(epochs_smooth, val_accuracies_smooth, label='Validation Accuracy',color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'training_metrics.png'))
    plt.close()
    
class CustomCallbacks(tf.keras.callbacks.Callback):
    def __init__(self, log_path, run_dir):
        super(CustomCallbacks, self).__init__()
        self.log_path = log_path
        self.run_dir = run_dir

    def on_epoch_end(self, epoch, logs=None):
        # Update log with current epoch metrics
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        train_accuracy = logs.get('accuracy')
        val_accuracy = logs.get('val_accuracy')
        update_log(self.log_path, epoch, train_loss, val_loss, train_accuracy, val_accuracy)
        
        # Plot metrics
        plot_metrics(self.log_path, self.run_dir)