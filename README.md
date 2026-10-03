# CIFAR-10 Image Classifier

An end-to-end computer-vision project that explores image classification on CIFAR-10 and serves predictions through a Streamlit application. The repository includes a training and experimentation notebook, saved Keras models, and an image-upload inference interface.

[Open the live demo](https://cifar-10-deep-learning-image-classifier-sp.streamlit.app)

## Project at a Glance

| | Details |
|---|---|
| Task | 10-class image classification |
| Dataset | CIFAR-10: 60,000 color images, 32 x 32 pixels |
| Notebook split | 50,000 original training images; 10,000 held out for test; 20% of the original training split reserved for validation |
| Model input | 32 x 32 x 3 RGB image, scaled to [0, 1] |
| App model | Keras CNN loaded from `TransferlearningSP.keras` |
| Interface | Streamlit image upload and predicted class/confidence display |
| Main libraries | TensorFlow/Keras, NumPy, Pillow, Streamlit, Matplotlib, Seaborn, scikit-learn |

The CIFAR-10 labels are airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## What the App Does

1. Accepts a PNG, JPG, or JPEG upload.
2. Converts the image to RGB, resizes it to 32 x 32, and scales pixel values to [0, 1].
3. Adds a batch dimension and passes the image to the cached Keras model.
4. Displays the predicted class and the largest softmax score as a confidence percentage.

The model was trained for CIFAR-10's small 32 x 32 images. Resizing a general-purpose photo to that resolution discards detail, so the demo is best suited to clear images with a prominent CIFAR-10-like object. The displayed softmax score is not a calibrated probability.

## Modeling Workflow

### CNN classifier

The notebook defines a Sequential CNN trained from scratch:

- Three valid-padding 3 x 3 convolution layers with 32, 64, and 64 filters; ReLU activations.
- Two 2 x 2 max-pooling layers, followed by flattening.
- A 64-unit ReLU dense layer and a 10-unit softmax output layer.
- Adam optimizer, categorical cross-entropy loss, and accuracy as the training metric.
- Training configuration: batch size 64, up to 20 epochs, and early stopping with patience 5 and best-weight restoration.
- Training-only augmentation: rotations up to 15 degrees, horizontal/vertical shifts up to 10%, and horizontal flips.

The Streamlit app loads `TransferlearningSP.keras`. Inspection of that saved artifact shows a convolutional model, not a ResNet50 transfer-learning model. `ImageClassificationModelSP.keras` is also a convolutional model.

### Transfer-learning experiment

The notebook also defines an ImageNet-pretrained ResNet50 feature extractor with its base layers frozen, followed by global average pooling, a 128-unit dense layer, 30% dropout, and a 10-class softmax head. Its configured run is 10 epochs with batch size 64. This is an experiment in the notebook; it is not the architecture contained in the app's `TransferlearningSP.keras` artifact.

## Evaluation Status

No test accuracy, precision, recall, F1 score, or confusion matrix is reported here because the notebook's current evaluation code does not produce valid test metrics: it predicts using `X_train` and compares those predictions with `y_test`. The sample counts and data do not match. In addition, the notebook's save calls use `model.save(...)` rather than `transfer_model.save(...)`, so the file named `TransferlearningSP.keras` is not the transfer-learning model.

These should be corrected before quoting model performance: evaluate the intended model on `X_test`, compare it with `y_test`, and save the same model that the app loads. The notebook currently has no executed training history from which reproducible performance numbers can be verified. This README intentionally does not claim an accuracy figure.

## Repository Layout

```text
.
├── Image Classification using DL.ipynb  # Dataset exploration and model experiments
├── ImageClassificationModelSP.keras     # Saved CNN model
├── TransferlearningSP.keras             # CNN artifact loaded by the Streamlit app
├── app/
│   ├── requirements.txt
│   └── streamlit_app.py
└── .devcontainer/
    └── devcontainer.json
```

## Run Locally

Run the commands from the repository root so the app can find `TransferlearningSP.keras`.

```bash
python -m venv .venv
```

Activate the environment, then install the app dependencies and launch Streamlit:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux (use this instead on those platforms)
# source .venv/bin/activate

python -m pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit, typically `http://localhost:8501`. The devcontainer configuration uses a Python 3.11 base image. TensorFlow must be installable for the Python version and platform in use.

## Tools and Skills

Python, TensorFlow/Keras, convolutional neural networks (CNNs), transfer learning, image preprocessing, data augmentation, train/validation/test splits, model serialization, Streamlit, NumPy, Pillow, Matplotlib, Seaborn, and scikit-learn classification metrics.

