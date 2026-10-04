import os

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


DATASET_PATH = "TransferlearningSP.keras"


def main():
    print("Loading CIFAR-10 dataset...")
    (x_train_all, y_train_all), (x_test, y_test) = cifar10.load_data()

    x_train_all = x_train_all.astype("float32")
    x_test = x_test.astype("float32")

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_all,
        y_train_all,
        test_size=0.2,
        random_state=42,
        stratify=y_train_all,
    )

    y_train_cat = to_categorical(y_train, 10)
    y_val_cat = to_categorical(y_val, 10)
    y_test_cat = to_categorical(y_test, 10)

    x_train_resnet = tf.keras.applications.resnet50.preprocess_input(x_train * 255.0)
    x_val_resnet = tf.keras.applications.resnet50.preprocess_input(x_val * 255.0)
    x_test_resnet = tf.keras.applications.resnet50.preprocess_input(x_test * 255.0)

    print("Building transfer-learning model...")
    base_model = tf.keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(32, 32, 3),
    )
    base_model.trainable = False

    transfer_model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(10, activation="softmax"),
        ]
    )

    transfer_model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("Training model...")
    history = transfer_model.fit(
        x_train_resnet,
        y_train_cat,
        validation_data=(x_val_resnet, y_val_cat),
        epochs=10,
        batch_size=64,
        verbose=1,
    )

    print("Evaluating model...")
    loss, acc = transfer_model.evaluate(x_test_resnet, y_test_cat, verbose=0)
    print(f"Test accuracy: {acc:.4f}")

    transfer_model.save(DATASET_PATH)
    print(f"Saved model to {DATASET_PATH}")

    print("Training history keys:", list(history.history.keys()))


if __name__ == "__main__":
    main()
