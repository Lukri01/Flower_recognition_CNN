import os

import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

import wandb
from keras import layers, models, Input
from keras.callbacks import EarlyStopping
from keras.src.applications.efficientnet import preprocess_input, EfficientNetB1
from keras.utils import image_dataset_from_directory
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import compute_class_weight
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

from split_data import split_data

EPOCHS = 40
BATCH_SIZE = 32
IMG_SIZE = (255, 255)
SEED = 42

# Calculate the class weights for a dataset
def get_class_weights(ds, class_names):
    labels = []
    for _, y in ds.unbatch():
        labels.append(np.argmax(y.numpy()))

    weights = compute_class_weight(class_weight='balanced', classes=np.arange(len(class_names)), y=labels)

    return dict(enumerate(weights))

# Load datasets for training, validation and test purposes
# The data has already been split and placed into
def create_image_datasets_from_directory(folder_path):
    train_ds = image_dataset_from_directory(
        f"{folder_path}/train", image_size=IMG_SIZE,
        batch_size=BATCH_SIZE, seed=SEED, label_mode="categorical"
    )
    val_ds = image_dataset_from_directory(
        f"{folder_path}/val", image_size=IMG_SIZE,
        batch_size=BATCH_SIZE, seed=SEED, label_mode="categorical"
    )
    test_ds = image_dataset_from_directory(
        f"{folder_path}/test", image_size=IMG_SIZE,
        batch_size=BATCH_SIZE, seed=SEED, label_mode="categorical"
    )

    class_names = train_ds.class_names

    # Automatically choose optimal buffer size
    AUTOTUNE = tf.data.AUTOTUNE
    # cache() - cache the dataset in the memory after the first epoch
    # shuffle(1000) - shuffles the dataset before every epoch to prevent overfitting and poor generalization
    # prefetch(buffer_size) - fetches next batch of data while the model is training on the previous one
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names

# Define model
def define_transfer_learning_model(num_classes):
    base_model = EfficientNetB1(include_top=False, input_shape=IMG_SIZE + (3,), weights='imagenet')
    base_model.trainable = False

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    inputs = Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

def train_model(model, train_ds, val_ds, class_names):
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    class_weights = get_class_weights(train_ds, class_names=class_names)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(patience=5, restore_best_weights=True),
            WandbMetricsLogger(),
            WandbModelCheckpoint("best_model.keras", save_best_only=True, monitor="val_accuracy", mode="max"),
        ]
    )
    return model

# Evaluate model and print metrics
def evaluate_model(model, test_ds, class_names):
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred_prob = model.predict(test_ds)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_true, axis=1)

    wandb.log({
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_precision": precision_score(y_true, y_pred, average='macro'),
        "test_recall": recall_score(y_true, y_pred, average='macro'),
        "test_f1": f1_score(y_true, y_pred, average='macro')
    })

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, xticklabels=class_names, yticklabels=class_names, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def main():
    wandb.login(key="7a86129a97dfdef43fa742cd1054fddad6f1df11")
    wandb.init(project="flower-classification", config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "img_size": IMG_SIZE,
        "model": "EfficientNetB1",
        "optimizer": "adam",
        "learning_rate": 1e-4
    })

    folder_path = "flowers"
    new_folder_path = "flower_split"
    if not os.path.exists(new_folder_path):
        split_data(folder_path, new_folder_path, 0.70, 0.15, 0.15)
    train_ds, val_ds, test_ds, class_names = create_image_datasets_from_directory(new_folder_path)

    model = define_transfer_learning_model(num_classes=len(class_names))
    model = train_model(model, train_ds, val_ds, class_names)
    evaluate_model(model, test_ds, class_names)

    wandb.finish()

if __name__ == "__main__":
    main()
