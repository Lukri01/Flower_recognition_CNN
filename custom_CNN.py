import os

os.environ["KERAS_BACKEND"] = "tensorflow"
from contextlib import redirect_stdout
from datetime import datetime

import numpy as np
import pandas as pd
from keras import Input
from sklearn.metrics import accuracy_score, precision_score, classification_report, recall_score, f1_score
from sklearn.model_selection import train_test_split
from keras import layers, models
from keras.src.callbacks import ModelCheckpoint, EarlyStopping
from keras.src.legacy.preprocessing.image import ImageDataGenerator

def evaluate_model(model, test_generator):
    predict = model.predict(test_generator)

    y_pred = np.argmax(predict, axis=1)
    y_true = test_generator.classes
    class_names = list(test_generator.class_indices.keys())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall_s = recall_score(y_true, y_pred, average='macro')
    f1_s = f1_score(y_true, y_pred, average='macro')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall_s}")
    print(f"F1 score: {f1_s}")

    print(f"Classification Report per Class:\n{classification_report(y_true, y_pred, target_names=class_names)}")

    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join("logs", f"{timestamp}.log"), "a") as f:
        f.write("\nModel Summary:\n")
        with redirect_stdout(f):
            model.summary()
        f.write("\n\nEvaluation:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall_s:.4f}\n")
        f.write(f"F1-score: {f1_s:.4f}\n\n")
        f.write("Classification Report per Class:\n")
        f.write(f"{classification_report(y_true, y_pred, target_names=class_names)}")

def train_model(model, train_generator, validation_generator):
    model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=["accuracy"])

    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=30,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True),
                   ModelCheckpoint("best_model_1.keras", save_best_only=True)]
    )

    return model

def define_cnn_model(num_classes):
    model = models.Sequential([
        Input(shape=(150,150,3)),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

def create_image_data_generators(train_df, val_df, test_df):
    train_data_generator = ImageDataGenerator(rescale=1./255, rotation_range=25, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")
    val_test_data_generator = ImageDataGenerator(rescale=1./255)

    train_generator = train_data_generator.flow_from_dataframe(train_df, x_col="filename", y_col="label",target_size=(150,150), class_mode="categorical", batch_size=32, shuffle=True, seed=42)
    val_generator = val_test_data_generator.flow_from_dataframe(val_df, x_col="filename", y_col="label", target_size=(150,150), class_mode="categorical", batch_size=32, shuffle=True, seed=42)
    test_generator = val_test_data_generator.flow_from_dataframe(test_df, x_col="filename", y_col="label", target_size=(150,150), class_mode="categorical", batch_size=32, shuffle=False, seed=42)

    return train_generator, val_generator, test_generator

def split_data(data_df):
    train_df, temp_df = train_test_split(data_df, test_size=0.3, stratify=data_df["label"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

    return train_df, val_df, test_df

def construct_dataframe_from_folder(parent_folder_path):
    if not os.path.isdir(parent_folder_path):
        raise FileNotFoundError(parent_folder_path)

    sub_folders_names = os.listdir(parent_folder_path)

    paths = []
    labels = []

    for sub_folder_name in sub_folders_names:
        sub_folder_path = os.path.join(parent_folder_path, sub_folder_name)
        if not os.path.isdir(sub_folder_path):
            continue

        for image_name in os.listdir(sub_folder_path):
            if image_name.endswith(".jpg"):
                image_path = os.path.join(sub_folder_path, image_name)
                paths.append(image_path)
                labels.append(sub_folder_name)

    df = pd.DataFrame({"filename": paths, "label": labels})

    return df, sub_folders_names

def main():
    data_df, classes = construct_dataframe_from_folder("flowers")

    train_df, val_df, test_df = split_data(data_df)

    train_generator, val_generator, test_generator = create_image_data_generators(train_df, val_df, test_df)

    model = define_cnn_model(len(classes))

    model = train_model(model, train_generator, val_generator)

    evaluate_model(model, test_generator)

if __name__ == '__main__':
    main()