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
| App model | Keras model loaded from `TransferlearningSP.keras`; app supports the checked-in CNN and the notebook's ResNet50 model |
| Interface | Streamlit image upload and predicted class/confidence display |
| Main libraries | TensorFlow/Keras, NumPy, Pillow, Streamlit, Matplotlib, Seaborn, scikit-learn |

The CIFAR-10 labels are airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## What the App Does

1. Accepts a PNG, JPG, or JPEG upload.
2. Converts the image to RGB, resizes it to 32 x 32, and scales pixel values to [0, 1].
3. Adds a batch dimension and applies ResNet50 preprocessing when the loaded model contains a ResNet50 backbone; scratch-CNN inputs remain scaled to [0, 1].
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

The notebook saves the scratch CNN to `ImageClassificationModelSP.keras` and the transfer-learning model to `TransferlearningSP.keras`. The currently checked-in `TransferlearningSP.keras` file is still the older scratch CNN; rerun the corrected notebook to replace it with the ResNet50 transfer-learning model.

### Transfer-learning experiment

The project includes a reproducible ResNet50 transfer-learning training script at `train_transfer_learning.py`. It mirrors the notebook workflow: an ImageNet-pretrained ResNet50 backbone with frozen base layers, global average pooling, a 128-unit dense layer, 30% dropout, and a 10-class softmax head. The script trains for 10 epochs with batch size 64 and uses the ResNet50 ImageNet preprocessing function for both training and inference.

The saved `TransferlearningSP.keras` file in the repository is currently the verified working app artifact, not the finished ResNet50 model produced by that training script. If you want the full ResNet50 checkpoint, run `python train_transfer_learning.py` after installing the requirements.

## Evaluation Status

The checked-in `TransferlearningSP.keras` artifact is the working app model, and the repository's current verified evaluation is based on that model. The ResNet50 transfer-learning path is implemented as a training script and is intended to replace the current app model after a full training run.

I evaluated the current app model on the full held-out CIFAR-10 test set (10,000 images), using the app's RGB conversion, 32 x 32 resize, and [0, 1] scaling:

| Metric | Test result |
|---|---:|
| Accuracy | 72.31% |
| Macro precision | 72.30% |
| Macro recall | 72.31% |
| Macro F1 score | 71.89% |

These results describe the existing CNN artifact, not the ResNet50 transfer-learning model. The corrected notebook pipeline saves the ResNet50 model and computes its own held-out test metrics when run; it has not completed a training run yet. The train/validation split uses a fixed random seed and stratifies by class.

## Repository Layout

```text
.
├── Image Classification using DL.ipynb  # Dataset exploration and model experiments
├── ImageClassificationModelSP.keras     # Saved CNN model
├── TransferlearningSP.keras             # App model; regenerate by rerunning the notebook
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

