# Speech-Based-Person-Recognition-for-Biometric-Applications
Using a one-dimensional Convolutional Neural Network (1D CNN) trained on speech signals’ Fourier Magnitude spectra, the uniqueness of speech based on frequency response is studied. The paper demonstrates 3 different models being trained and the results are compared.

## Table of Contents

- [Features](#features)
- [Dataset](#dataset)
- [Results](#results)
- [Future Work](#future-work)
- [Contributors](#contributors)

## Features

- **Biometric Authentication:** Secure user identification using speech.
- **1D CNN Implementation:** Deep learning model for high accuracy in speaker recognition.
- **Data Augmentation:** Techniques such as time-stretching, pitch shifting, and noise addition to enhance model robustness.
- **User Interface:** Intuitive graphical interface for easy interaction and predictions.

## Dataset

The dataset used for training and testing is sourced from Kaggle and consists of over 12,000 audio samples. It includes CSV files that provide paths to the audio files and the corresponding speaker names.

## Results

The implementation includes various models for comparison:

- Logistic Regression: Achieved the highest accuracy among tested models, serving as a baseline.
- 1D CNN (without windowing): Performed well, though not as effectively as logistic regression.
- 1D CNN (with windowing): Showed lower performance due to potential data suppression from windowing techniques.
  
All models achieved over 85% accuracy, demonstrating the effectiveness of this approach for biometric recognition.

## Future Work
- Improve the model's handling of background noise to increase accuracy in real-world scenarios.
- Train the model with real-time datasets to enhance performance and robustness.
- Explore integration with other biometric modalities, such as face recognition, for improved security.

## Contributors
Garima Ranjan 

Feel free to reach out with any questions or contributions!
