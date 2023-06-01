from keras import Model
from keras.applications import MobileNetV2
from keras.layers import Dense
from keras.models import load_model
from pathlib import Path
from PIL import Image
from tensorflow import keras
import tensorflow as tf

# Constants

dataset_path = Path('./small_flower_dataset')
pretrained_model_path = Path('mobilenetv2.h5')
class_names: 'list[str]' = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# Utils


def get_dataset_length(dataset: tf.data.Dataset) -> int:
    """Get the length of a tensorflow dataset.

    Args:
        dataset (tf.data.Dataset): A dataset that has not been batched.

    Returns:
        int: The length of the dataset.
    """
    return tf.data.experimental.cardinality(dataset).numpy()

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


def task_4_prepare_dataset(train_ratio: float=0.7, val_ratio: float=0.1, test_ratio: float=0.2, seed: int=42, batch_size: int=32) -> 'tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]':
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


def task_5_compile_and_train(model: Model, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, learning_rate: float=0.01, momentum: float=0, nesterov: bool=False):
    """_summary_

    Args:
        model (Model): _description_
        train_ds (tf.data.Dataset): _description_
        val_ds (tf.data.Dataset): _description_
        learning_rate (float, optional): A Tensor, floating point value, or a schedule that is a tf.keras.optimizers.schedules.LearningRateSchedule, or a callable that takes no arguments and returns the actual value to use. The learning rate. Defaults to 0.01.
        momentum (float, optional): loat hyperparameter >= 0 that accelerates gradient descent in the relevant direction and dampens oscillations. Defaults to 0, i.e., vanilla gradient descent.
        nesterov (bool, optional): boolean. Whether to apply Nesterov momentum. Defaults to False.
    """
    
    # Compile and train your model with an SGD optimizer using the following parameters learning_rate=0.01, momentum=0.0, nesterov=False.
    print("*** Task 5 ***")

    optimizer = tf.keras.optimizers.experimental.SGD(
        learning_rate=learning_rate,
        momentum=momentum,
        nesterov=nesterov,
    )

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    metrics=['accuracy', 'sparse_categorical_crossentropy']

    model.compile(optimizer=optimizer, loss=loss_object, metrics=metrics)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3
    )

    pass


def task_6_plot_metrics():
    # Plot the training and validation errors vs time as well as the training and validation accuracies.
    print("*** Task 6 ***")

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
    task_5_compile_and_train(model, train_ds, val_ds)
    exit()
    task_6_plot_metrics()
    task_7_expriment_with_different_hyperparameters()
    best_learning_rate = 0.01
    task_8_expriment_with_different_hyperparameters(best_learning_rate)
    task_9_generate_acceleated_datasets()
    task_10_train_on_accelerated_datasets()

# TODOs:
# Header comments
# Each function parameter documented including type and shape of parameters
# return values clearly documented
