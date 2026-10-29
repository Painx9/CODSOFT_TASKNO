# Task 5: Handwritten Text Generation (RNN & Handwriting Synthesis)

## Overview

This project is part of the CodSoft Machine Learning virtual internship. The objective is to implement a character-level Recurrent Neural Network (RNN) to generate text, and subsequently render that text to mimic human handwriting using a dataset of character images.

## Datasets

This hybrid pipeline utilizes two distinct datasets:

1. **Text Corpus:** A subset of Friedrich Nietzsche's writings to train the RNN on English spelling, grammar, and sequential character patterns.
2. **Image Dataset:** A collection of 3,410 individual handwritten character images (A-Z, a-z, 0-9) used to synthesize the final visual output.

## Methodology

This task was solved using a two-part automated pipeline:

### 1. The Brain: Sequence Generation (LSTM)

- **Data Preprocessing:** Cut the raw text corpus into overlapping sequences of 40 characters and applied One-Hot Encoding to convert characters into three-dimensional boolean arrays.
- **Model Architecture:** Built a Deep Learning sequence model using **TensorFlow/Keras** featuring an **LSTM layer (128 units)** and a Dense Softmax output layer.
- **Training:** The model was trained for 50 epochs to deeply map English linguistic patterns without relying on a predefined dictionary.
- **Text Generation:** Implemented a prediction loop utilizing a "temperature" parameter (set to 0.2) to strictly sample highly probable English character sequences while avoiding gibberish.

### 2. The Hand: Handwriting Synthesis (Computer Vision)

- **Image Stitching:** Developed a custom Python function using **OpenCV** to intercept the RNN's autonomously generated text string.
- **Character Matching:** The script strictly matches the case of each generated character (e.g., lowercase "p") to a randomly selected image of that exact character from the provided image dataset.
- **Final Output:** The script dynamically resizes (32x32 pixels) and horizontally concatenates the individual character images, inserting blank arrays for spaces, to output a cohesive, handwritten-like image of the RNN's hallucinated text.

## Results

The model successfully learned fundamental English phonetic and structural patterns. By connecting the LSTM's output directly to the OpenCV image-stitching script, the pipeline autonomously generates and visually writes out unique, coherent English phrases that mimic authentic human handwriting.

## Technologies Used

- Python
- TensorFlow / Keras (Sequential, LSTM, Dense)
- OpenCV (`cv2`) for Image Processing
- Pandas & NumPy
- Matplotlib (Data Visualization)
