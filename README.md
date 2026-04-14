# CIFAR-10 Image Classifier using Deep Learning

An interactive web app built with **Streamlit** and **TensorFlow** that classifies images into the 10 CIFAR-10 categories using a trained deep learning model.

The project includes both the training notebook and the deployment app:
- `Image Classification using DL.ipynb` for model training and experimentation
- `app/streamlit_app.py` for the live Streamlit inference interface
- `TransferlearningSP.keras` for the saved transfer-learning model used in production

## Live Demo

Deployed app: https://cifar-10-deep-learning-image-classifier-sp.streamlit.app

## Features

- Upload an image in `PNG`, `JPG`, or `JPEG` format
- Automatically resizes the image to `32x32` for model inference
- Shows the predicted class and confidence score in real time
- Supports all 10 CIFAR-10 classes:
  - airplane
  - automobile
  - bird
  - cat
  - deer
  - dog
  - frog
  - horse
  - ship
  - truck
- Uses a cached TensorFlow model for faster repeat predictions

## How It Works

1. The user uploads an image through the Streamlit interface.
2. The app converts the image to RGB and resizes it to `32x32`.
3. Pixel values are normalized to the `[0, 1]` range.
4. The pre-trained Keras model in `TransferlearningSP.keras` generates a prediction.
5. The app displays the predicted CIFAR-10 class and confidence percentage.

## Model Details

- Dataset: CIFAR-10
- Model type: transfer learning
- Base network: ResNet50, trained in the notebook workflow
- Output classes: 10 CIFAR-10 categories
- Saved model: `TransferlearningSP.keras`

## Project Structure

```text
.
├── Image Classification using DL.ipynb
├── ImageClassificationModelSP.keras
├── TransferlearningSP.keras
└── app/
    ├── requirements.txt
    └── streamlit_app.py
```

## Local Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd CIFAR-10-Deep-Learning-Image-Classifier-Model-main
```

### 2. Install dependencies

It is recommended to install packages from the `app/requirements.txt` file:

```bash
pip install -r app/requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

If Streamlit does not open automatically, visit the local URL shown in the terminal, usually `http://localhost:8501`.

## Requirements

- Python 3.10+ recommended
- Streamlit
- TensorFlow
- NumPy
- Pillow

## Notes

- The app expects images that represent CIFAR-10-style objects and works best when the uploaded image is clear and centered.
- Although the uploader accepts larger files, the model always evaluates the image after resizing it to `32x32`.
- If you retrain the model, replace `TransferlearningSP.keras` with the new exported file.

## Credits

This project was built as a CIFAR-10 image classification demo using deep learning and Streamlit.
