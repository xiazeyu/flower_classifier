from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.applications import MobileNetV2
from keras.models import load_model
from keras.layers import Dense
from keras import Model
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import shutil

dataset_path = Path('./small_flower_dataset')
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

def my_team() -> 'list[tuple[int, str, str]]':
    """
    Returns
    -------
    list[tuple[int, str, str]]
        The list of the team members of this assessment submission as a list of
        triplet of the form (student_number, first_name, last_name).

    """
    return [(11262141, 'Tian-Ching', 'Lan'), (11398299, 'Zeyu', 'Xia')]

def task_1_dataset_sanity_check():
    try:
        assert dataset_path.exists()
        assert dataset_path.is_dir()

    except:
        print('Dataset not found. Make sure it is under ./small_flower_dataset.')

    try:
        existing_folders = [item.name for item in dataset_path.iterdir() if item.is_dir()]
        assert set(class_names) == set(existing_folders)
    except:
            print('One of the flower folder not found. Make sure it is under ./small_flower_dataset.')

    
    
    # TODO: check if image counts are all 200

    try:
        for flower_folder in class_names:
            fullpath = dataset_path / flower_folder
            file_paths  = fullpath.rglob("*")
            files = [file for file in file_paths if file.is_file()]
            assert len(files) == 200
            file_paths  = fullpath.rglob("*.jpg")
            files = [file for file in file_paths if file.is_file()]
            assert len(files) == 200
    except:
        print('One of the flower folder contains not 200 files')
        
    try:
        for flower_folder in class_names:
            fullpath = dataset_path / flower_folder
            file_paths  = fullpath.rglob("*.jpg")
            files = [file for file in file_paths if file.is_file()]
            for image in files:
                Image.open(image).verify()
    except:
        print('At least one of the image is corrupted')

def task_2_download_pretrained_model():
    model_path = Path('mobilenetv2.h5')
    if model_path.exists():
        return
    # Using the tf.keras.applications module download a pretrained MobileNetV2 network
    model = MobileNetV2(weights='imagenet')
    model.save(model_path)
    assert model_path.exists()

def task_3_replace_last_layer():
    model_path = Path('mobilenetv2.h5')
    model = load_model(model_path)
    # model.summary()
    flower_output = Dense(5, activation='softmax')  
    flower_output = flower_output(model.layers[-2].output)
    flower_input  = model.input
    flower_model = Model(inputs = flower_input, outputs = flower_output)
    for layer in flower_model.layers[:-1]:
        layer.trainable = False
    return flower_model

def task_4_prepare_dataset(train_ratio = 0.8, val_ratio = 0.1, test_ratio = 0.1, seed = 42, batch_size = 32):
    assert train_ratio + val_ratio + test_ratio == 1
    
    dataset_subpath = 'dataset'
    
    shutil.rmtree(Path(dataset_subpath), ignore_errors=True)
    
    file_list = list(dataset_path.glob('**/*.jpg'))
    print(f'Imported {len(file_list)} images.')
    
    df = pd.DataFrame(file_list)
    df = df.sample(frac=1, random_state=seed)
    
    total_rows = len(df)
    train_size = int(train_ratio * total_rows)
    val_size = int(val_ratio * total_rows)
    # test_size = int(test_ratio * total_rows)
    
    train_path = df[:train_size]
    val_path = df[train_size:train_size + val_size]
    test_path = df[train_size + val_size:]
    
    print(f'Seperated into Train ({len(train_path)}), Val ({len(val_path)}), Test ({len(test_path)})')\
    
    for index, filepath in train_path.iterrows():
        src = filepath[0]
        dst = Path(dataset_subpath) / 'train' / filepath[0].relative_to(dataset_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)

    for index, filepath in val_path.iterrows():
        src = filepath[0]
        dst = Path(dataset_subpath) / 'val' / filepath[0].relative_to(dataset_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)

    for index, filepath in test_path.iterrows():
        src = filepath[0]
        dst = Path(dataset_subpath) / 'test' / filepath[0].relative_to(dataset_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        Path(dataset_subpath) / 'train',
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=(224, 224),
        shuffle=True,
        seed=seed,
        interpolation='bilinear',
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        Path(dataset_subpath) / 'val',
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=(224, 224),
        shuffle=True,
        seed=seed,
        interpolation='bilinear',
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        Path(dataset_subpath) / 'test',
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=(224, 224),
        shuffle=True,
        seed=seed,
        interpolation='bilinear',
    )
    
    return train_ds, val_ds, test_ds
    

if __name__ == "__main__":
    print(my_team())  # should print your team

    # Check if TensorFlow is built with CUDA support
    if tf.test.is_built_with_cuda():
        print("TensorFlow is built with CUDA support.")
    else:
        print("TensorFlow is not built with CUDA support.")

    gpu_devices = tf.config.list_physical_devices('GPU')
    for device in gpu_devices:
        print("GPU device name:", device.name)
        print("GPU details:")
        print(tf.config.experimental.get_device_details(device))
    if len(gpu_devices) == 0:
        print("No GPU available.")
    
    #task_1_dataset_sanity_check()
    #task_2_download_pretrained_model()
    #model = task_3_replace_last_layer()
    # model.summary()
    task_4_prepare_dataset()