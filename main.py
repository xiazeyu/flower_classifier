from keras import Model
from keras.applications import MobileNetV2
from keras.layers import Dense
from keras.models import load_model
from keras.callbacks import EarlyStopping
from pathlib import Path
from PIL import Image
from tensorflow import keras
import tensorflow as tf
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Constants

dataset_path = Path('./small_flower_dataset')
pretrained_model_path = Path('mobilenetv2.h5')
class_names: 'list[str]' = [
    'daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Utils


def get_dataset_length(dataset: tf.data.Dataset) -> int:
    """Get the length of a tensorflow dataset.

    Args:
        dataset (tf.data.Dataset): A dataset that has not been batched.

    Returns:
        int: The length of the dataset.
    """
    return tf.data.experimental.cardinality(dataset).numpy()

def evaluate_model(model: Model, dataset: tf.data.Dataset):
    pass

def select_best_model( model: list[tuple[str, Model]] = None):
    pass

# Tasks


def my_team() -> 'list[tuple[int, str, str]]':
    """Return the list of team members of this assessment submission
    as a list of triplet of the form (student_number, first_name, last_name).

    Returns:
        list[tuple[int, str, str]]: (student_number, first_name, last_name)
    """

    return [(11262141, 'Tian-Ching', 'Lan'), (11398299, 'Zeyu', 'Xia')]


def env_check() -> None:
    """Quick environment check to ensure TensorFlow is installed correctly.
    And check if the GPU is available.
    """

    # Check if TensorFlow is built with CUDA support
    print("*** Environment Check ***")

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


def task_1_dataset_sanity_check() -> None:
    """Download the small flower dataset from Canvas.
    Check if the dataset is downloaded and extracted correctly.

    Raises:
        Exception: Dataset not found. Make sure it is under ./small_flower_dataset.
        Exception: One of the flower folder not found. Make sure it is under ./small_flower_dataset.
        Exception: One of the flower folder not contains 200 files.
        Exception: At least one of the image is corrupted.
    """

    print("*** Task 1 ***")

    # Check if the dataset is downloaded and extracted.
    try:
        assert dataset_path.exists()
        assert dataset_path.is_dir()
    except:
        raise Exception(
            'Dataset not found. Make sure it is under ./small_flower_dataset.')

    # Check if the dataset contains 5 folders.
    try:
        existing_folders = [
            item.name for item in dataset_path.iterdir() if item.is_dir()]
        assert set(class_names) == set(existing_folders)
    except:
        raise Exception(
            'One of the flower folder not found. Make sure it is under ./small_flower_dataset.')

    # Check if image counts are all 200
    try:
        for flower_folder in class_names:
            full_path = dataset_path / flower_folder
            files = [file for file in full_path.rglob("*") if file.is_file()]
            jpg_files = [file for file in full_path.rglob(
                "*.jpg") if file.is_file()]
            assert len(files) == 200
            assert set(jpg_files) == set(files)
    except:
        raise Exception('One of the flower folder not contains 200 files.')

    # Check if all images are not corrupted
    try:
        for flower_folder in class_names:
            full_path = dataset_path / flower_folder
            file_paths = full_path.rglob("*.jpg")
            files = [file for file in file_paths if file.is_file()]
            for image in files:
                Image.open(image).verify()
    except:
        raise Exception('At least one of the image is corrupted.')

    print('ok')


def task_2_download_pretrained_model() -> None:
    """Using the tf.keras.applications module download a pretrained MobileNetV2 network.
    Download a pretrained MobileNetV2 network to {pretrained_model_path}.
    """

    print("*** Task 2 ***")

    if pretrained_model_path.exists():
        print('Pretrained model already exists.')
        return
    # Using the tf.keras.applications module download a pretrained MobileNetV2 network
    model: Model = MobileNetV2(weights='imagenet')
    model.save(pretrained_model_path)
    assert pretrained_model_path.exists()
    print('Pretrained model downloaded.')


def task_3_replace_last_layer(classes: int = 5, print_model: bool = False) -> Model:
    """Replace the last layer of the downloaded neural network with a Dense layer of the
    appropriate shape for the 5 classes of the small flower dataset
    {(x1,t1), (x2,t2),..., (xm,tm)}.

    Args:
        classes (int, optional): The number of classes in the dataset. Defaults to 5.
        print_model (bool, optional): Whether to print the model summary. Defaults to False.

    Returns:
        Model: The new model.
    """

    print("*** Task 3 ***")

    model: Model = load_model(pretrained_model_path)
    print('Pretrained model loaded.')
    if print_model:
        print('%' * 50)
        model.summary()
        print('%' * 50)
    flower_input: 'list' = model.input
    flower_output = Dense(classes, activation='softmax')
    flower_output = flower_output(model.layers[-2].output)
    flower_model = Model(inputs=flower_input, outputs=flower_output)
    for layer in flower_model.layers[:-1]:
        layer.trainable = False
    print('New model created.')
    if print_model:
        print('%' * 50)
        flower_model.summary(show_trainable=True)
        print('%' * 50)

    return flower_model


def task_4_prepare_dataset(train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42, batch_size: int = 32) -> 'tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]':
    """Prepare your training, validation and test sets for the non-accelerated version of transfer learning.
    The dataset has been resized to 224x224 pixels, normalized to [0,1], batched, and prefetched.

    Args:
        train_ratio (float, optional): dataset split ratio for training set. Defaults to 0.7.
        val_ratio (float, optional): dataset split ratio for validation set. Defaults to 0.1.
        test_ratio (float, optional): dataset split ratio for test set. Defaults to 0.2.
        seed (int, optional): seed for random number generator. Defaults to 42.
        batch_size (int, optional): batch size. Defaults to 32.

    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: train, val, test datasets.
    """

    # Prepare your training, validation and test sets for the non-accelerated version of transfer learning.
    print("*** Task 4 ***")

    assert train_ratio + val_ratio + test_ratio == 1

    ds: tf.data.Dataset = keras.utils.image_dataset_from_directory(
        dataset_path,
        batch_size=None,
        image_size=(224, 224),
        shuffle=True,
        seed=seed,
    )

    total_size = get_dataset_length(ds)
    print(f'Imported {total_size} images.')
    class_names: 'list[str]' = ds.class_names
    print(f'Class names: {class_names}')

    # Normalize pixel values to [0, 1]
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    ds = ds.map(lambda x, y: (normalization_layer(x), y))

    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)

    # perform dataset splitting
    # enable caching to improve performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = ds.take(train_size).cache()
    val_ds = ds.skip(train_size).take(val_size).cache()
    test_ds = ds.skip(train_size).skip(val_size).cache()

    print(f'Seperated into Train ({get_dataset_length(train_ds)}), Val ({get_dataset_length(val_ds)}), Test ({get_dataset_length(test_ds)})')\

    # batch the datasets
    # enable prefetching to improve performance
    train_ds = train_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

    # check if the dataset is loaded correctly
    if False:
        for idx, (image_batch, labels_batch) in enumerate(train_ds):
            print(
                f'Iter {idx+1}/{len(train_ds)}: image {image_batch.shape}. label {labels_batch.shape}.')

    return train_ds, val_ds, test_ds


def task_5_compile_and_train(model: Model, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, learning_rate: float = 0.01, momentum: float = 0, nesterov: bool = False, min_delta: float = 0.001, patience: int = 5, max_epoch: int = 1000) -> keras.callbacks.History:
    """Compile and train your model with an SGD optimizer using the following parameters learning_rate=0.01, momentum=0.0, nesterov=False.
    This is also a wrapper function for the model.fit() method.

    Args:
        model (Model): Model to be trained.
        train_ds (tf.data.Dataset): Dataset for training.
        val_ds (tf.data.Dataset): Dataset for validation.
        learning_rate (float, optional): The learning rate. Defaults to 0.01.
        momentum (float, optional): Accelerates gradient descent in the relevant direction and dampens oscillations. Defaults to 0.
        nesterov (bool, optional): Whether to apply Nesterov momentum. Defaults to False.
        min_delta (float, optional): Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement. Defaults to 0.001.
        patience (int, optional): Number of epochs with no improvement after which training will be stopped. Defaults to 5.
        max_epoch (int, optional): Maximum number of epochs to train the model. Defaults to 1000.

    Returns:
        keras.callbacks.History: _description_
    """

    print("*** Task 5 ***")
   
    optimizer = tf.keras.optimizers.experimental.SGD(
        learning_rate=learning_rate,
        momentum=momentum,
        nesterov=nesterov,
    )

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    metrics = ['accuracy']

    # applying early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience)

    model.compile(optimizer=optimizer, loss=loss_object, metrics=metrics)

    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epoch,
        callbacks=[
            early_stopping,
            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        ]
    )


def task_6_plot_metrics(histories: 'list[tuple[str, keras.callbacks.History]]', plt_name: str = None) -> None:
    # Plot the training and validation errors vs time as well as the training and validation accuracies.
    print("*** Task 6 ***")

    fig, axs = plt.subplots(2, 2, figsize=(10, 5), layout='constrained') # type: tuple[plt.Figure, list[plt.Axes]]

    ax: 'dict[str, plt.Axes]' = {
        'training_loss': axs[0, 0],
        'training_accuracy': axs[0, 1],
        'validation_loss': axs[1, 0],
        'validation_accuracy': axs[1, 1],
    }

    ax['training_loss'].set_title('Training Loss')
    ax['training_accuracy'].set_title('Training Accuracy')
    ax['validation_loss'].set_title('Validation Loss')
    ax['validation_accuracy'].set_title('Validation Accuracy')

    ax['training_loss'].set_ylabel('Loss')
    ax['training_accuracy'].set_ylabel('Accuracy')
    ax['validation_loss'].set_ylabel('Loss')
    ax['validation_accuracy'].set_ylabel('Accuracy')
    
    for name, history in histories:
        training_loss = history.history['loss']
        training_accuracy = history.history['accuracy']
        validation_loss = history.history['val_loss']
        validation_accuracy = history.history['val_accuracy']
        x = np.arange(1, len(training_loss)+1)
        ax['training_loss'].plot(x, training_loss, label=name)
        ax['training_accuracy'].plot(x, training_accuracy, label=name)
        ax['validation_loss'].plot(x, validation_loss, label=name)
        ax['validation_accuracy'].plot(x, validation_accuracy, label=name)

    for i_ax in ax.values():
        i_ax.set_xlabel('Epoch')
        i_ax.grid(True)
        i_ax.legend()

    plt.tight_layout()
    if plt_name:
        plt.savefig(plt_name)
    plt.show()
    pass


def task_7_expriment_with_different_hyperparameters():
    # Experiment with 3 different orders of magnitude for the learning rate. Plot the results, draw conclusions.
    print("*** Task 7 ***")

    pass


def task_8_expriment_with_different_hyperparameters():
    # With the best learning rate that you found in the previous task, add a non zero momentum to the training with the SGD optimizer (consider 3 values for the momentum). Report how your results change.
    print("*** Task 8 ***")

    pass


def task_9_generate_acceleated_datasets():
    # Prepare your training, validation and test sets. Those are based on {(F(x1).t1), (F(x2),t2),...,(F(xm),tm)},
    print("*** Task 9 ***")

    pass


def task_10_train_on_accelerated_datasets():
    # Perform Task 8 on the new dataset created in Task 9.
    print("*** Task 10 ***")

    pass


if __name__ == "__main__":
    pass

    print(my_team())
    env_check()
    task_1_dataset_sanity_check()
    task_2_download_pretrained_model()
    model = task_3_replace_last_layer()
    train_ds, val_ds, test_ds = task_4_prepare_dataset()
    task_5_history = task_5_compile_and_train(model, train_ds, val_ds, max_epoch=5)
    task_6_plot_metrics([('Task 5', task_5_history)], 'Task 6')
    exit()
    task_7_expriment_with_different_hyperparameters()
    best_learning_rate = 0.01
    task_8_expriment_with_different_hyperparameters(best_learning_rate)
    task_9_generate_acceleated_datasets()
    task_10_train_on_accelerated_datasets()

# TODOs:
# Header comments
# Each function parameter documented including type and shape of parameters
# return values clearly documented
