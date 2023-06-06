from pathlib import Path
from typing import Callable
from PIL import Image
import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Constants

dataset_path: Path = Path('./small_flower_dataset')
pretrained_model_path: Path = Path('mobilenetv2.h5')
class_names: 'list[str]' = [
    'daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
log_dir: str = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
headless: bool = False

# Utils


def get_dataset_length(dataset: tf.data.Dataset) -> int:
    """Get the length of a tensorflow dataset.

    Args:
        dataset (tf.data.Dataset): A dataset that has not been batched.

    Returns:
        int: The length of the dataset.
    """
    return tf.data.experimental.cardinality(dataset).numpy()


def select_best_model(models: 'list[tuple[str, tf.keras.Model]]', dataset: tf.data.Dataset, metric: str = 'acc') -> 'tuple[str, tf.keras.Model]':
    """Select the best model from a list of models.

    Args:
        models (list[tuple[str, Model]]): Models to select from.
        dataset (tf.data.Dataset): Dataset to evaluate the models on.
        metric (str, optional): Metric to use for selection. Defaults to 'acc'.

    Returns:
        tuple[str, Model]: The best model.
    """

    assert metric in ['acc', 'loss']

    loss_values = []
    acc_values = []

    for name, model in models:
        loss_value, metrics_value = evaluate_model(model, dataset)
        loss_values.append(loss_value)
        acc_values.append(metrics_value)

    matrix = np.array([loss_values, acc_values])

    if metric == 'acc':
        best_model_index = np.argmax(matrix[1])
    else:
        best_model_index = np.argmin(matrix[0])

    return models[best_model_index]


def evaluate_model(model: tf.keras.Model, dataset: tf.data.Dataset) -> 'tuple[float, float]':
    """Returns the loss value & metrics values for the model in test mode.

    Args:
        model (Model): Model to evaluate.
        dataset (tf.data.Dataset): Dataset to evaluate the model on.

    Returns:
        tuple[float, float]: Loss value & metrics values for the model in test mode.
    """

    return model.evaluate(dataset)

def predict(model: tf.keras.Model, dataset: tf.data.Dataset) -> 'tuple[np.ndarray, np.ndarray]':
    """Predict the model on a dataset.

    Args:
        model (Model): Model to predict.
        dataset (tf.data.Dataset): Dataset to predict the model on.

    Returns:
        tuple[np.ndarray, np.ndarray]: Prediction & ground truth.
    """

    prob_prediction = model.predict(dataset) # shape=(dataset_length, 5)
    
    prediction = np.argmax(prob_prediction, axis=1) # shape=(dataset_length)

    Xs = []
    ys = []

    for idx, (X, y) in enumerate(dataset):
        nx = X.numpy()
        ny = y.numpy()
        Xs.append(nx)
        ys.append(ny)
        print(f'Preprocessing Batch {idx+1}/{len(dataset)}.')

    Xs = np.concatenate(Xs)
    ground_truth = np.concatenate(ys)
    
    return (prediction, ground_truth, Xs)

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

    tf_version = tf. __version__

    print(
        f'Current tenforflow version is {tf_version}. Make sure it is >= 2.10.1')

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


def task_1_dataset_sanity_check() -> None:
    """Download the small flower dataset from Canvas.
    Check if the dataset is downloaded and extracted correctly.

    Raises:
        Exception: Dataset not found. Make sure it is under ./small_flower_dataset.
        Exception: One of the flower folder not found. Make sure it is under ./small_flower_dataset.
        Exception: One of the flower folder not contains 200 files.
        Exception: At least one of the image is corrupted.
    """

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

    if pretrained_model_path.exists():
        print('Pretrained model already exists.')
        return
    # Using the tf.keras.applications module download a pretrained MobileNetV2 network
    model: tf.keras.Model = tf.keras.applications.MobileNetV2(
        weights='imagenet')
    model.save(pretrained_model_path)
    assert pretrained_model_path.exists()
    print('Pretrained model downloaded.')


def task_3_replace_last_layer(classes: int = 5, print_model: bool = False) -> tf.keras.Model:
    """Replace the last layer of the downloaded neural network with a Dense layer of the
    appropriate shape for the 5 classes of the small flower dataset
    {(x1,t1), (x2,t2),..., (xm,tm)}.

    Args:
        classes (int, optional): The number of classes in the dataset. Defaults to 5.
        print_model (bool, optional): Whether to print the model summary. Defaults to False.

    Returns:
        Model: The new model.
    """

    model: tf.keras.Model = tf.keras.models.load_model(pretrained_model_path)
    print('Pretrained model loaded.')
    if print_model:
        print('%' * 50)
        model.summary()
        print('%' * 50)
    flower_input: 'list' = model.input  # (None, 224, 224, 3)
    flower_output = tf.keras.layers.Dense(
        classes, activation='softmax')  # (None, 5)
    # (None, 1280) => (None, 5)
    flower_output = flower_output(model.layers[-2].output)
    # (None, 224, 224, 3) => (None, 5)
    flower_model = tf.keras.Model(inputs=flower_input, outputs=flower_output)
    # flower_model.trainable = False # Only works in tf==2.12.0
    for layer in flower_model.layers[:-1]:
        layer.trainable = False
    flower_model.layers[-1].trainable = True
    print('New model created.')
    if print_model:
        print('%' * 50)
        flower_model.summary(show_trainable=True)
        print('%' * 50)

    return flower_model


def task_4_prepare_dataset(train_ratio: float = 0.7,
                           val_ratio: float = 0.15,
                           test_ratio: float = 0.15,
                           seed: int = 42,
                           batch_size: int = 32) -> 'tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]':
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
        Each dataset has the shape of (X, y)=((batch_size, 224, 224, 3), (batch_size))
    """

    assert train_ratio + val_ratio + test_ratio == 1

    ds: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
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
    if batch_size is not None:
        train_ds = train_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
        test_ds = test_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

    # check if the dataset is loaded correctly
    if False:
        for idx, (image_batch, labels_batch) in enumerate(train_ds):
            print(
                f'Iter {idx+1}/{len(train_ds)}: image {image_batch.shape}. label {labels_batch.shape}.')

    return train_ds, val_ds, test_ds


def task_5_compile_and_train(model: tf.keras.Model,
                             train_ds: tf.data.Dataset,
                             val_ds: tf.data.Dataset,
                             learning_rate: float = 0.01,
                             momentum: float = 0,
                             nesterov: bool = False,
                             min_delta: float = 0.001,
                             patience: int = 5,
                             max_epoch: int = 1000,
                             extra_log_path: str = None,
                             early_stopping: bool = True,
                             checkpoint_best: bool = True,
                             tensorboard: bool = True,
                             save_model: bool = True,
                             seed: int = 42,
                             ) -> 'tuple[tf.keras.Model, tf.keras.callbacks.History]':
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
        extra_log_path (str, optional): Extra log path for TensorBoard. Defaults to None.
        early_stopping (bool, optional): Whether to apply early stopping. Defaults to True.
        checkpoint_best (bool, optional): Whether to use ModelCheckpoint. Defaults to True.
        tensorboard (bool, optional): Whether to use TensorBoard. Defaults to True.
        save_model (bool, optional): Whether to save model at end of training. Defaults to True.
        seed (int, optional): Seed for random number generator. Defaults to 42.

    Returns:
        tuple[Model, tf.keras.callbacks.History]: The trained model and the training history.
    """

    tf.keras.utils.set_random_seed(seed)

    model_type = 'normal'

    if len(model.layers) == 1:
        model_type = 'accelerated'

    train_conf = f'{model_type}_lr{learning_rate}_momentum{momentum}_nesterov={nesterov}'
    if extra_log_path:
        train_conf += f'_{extra_log_path}'
    
    print(f'Training with {train_conf}...')

    optimizer = tf.keras.optimizers.experimental.SGD(
        learning_rate=learning_rate,
        momentum=momentum,
        nesterov=nesterov,
    )

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    metrics = ['accuracy']

    model.compile(optimizer=optimizer, loss=loss_object, metrics=metrics)

    callbacks = []

    if early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=min_delta, patience=patience))

    if tensorboard:
        callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir=log_dir + train_conf, histogram_freq=1))
    
    if checkpoint_best:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=log_dir + train_conf + '/best.h5', monitor='val_loss', save_best_only=True, verbose=1))

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epoch,
        callbacks=callbacks
    )

    if save_model:
        model.save(log_dir + train_conf + '/weights.h5')

    return model, history


def task_6_plot_metrics(histories: 'list[tuple[str, tf.keras.callbacks.History]]', plt_name: str = None) -> None:
    """Plot the training and validation errors vs time as well as the training and validation accuracies.
    Args:
        histories (list[tuple[str, tf.keras.callbacks.History]]): A list of tuples of the form (name, history).
        plt_name (str, optional): The name of the plot file. Defaults to None.
    """
    # type: tuple[plt.Figure, list[plt.Axes]]
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')
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
    for name, history in sorted(histories):
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
    if not headless:
        plt.show()
    pass


def task_7_expriment_with_different_learning_rates(model_gen_function: 'Callable[[], tf.keras.Model]',
                                                   train_ds: tf.data.Dataset,
                                                   val_ds: tf.data.Dataset,
                                                   learning_rates: 'list[float]' = [
                                                       0.1, 0.001],
                                                   task_5_history: tf.keras.callbacks.History = None,
                                                   **extra_args) -> 'tuple[list[tuple[str, tf.keras.Model]], list[tuple[str, tf.keras.callbacks.History]]]':
    """Experiment with 3 different orders of magnitude for the learning rate. Plot the results, draw conclusions.

    Args:
        model_gen_function (Callable[[], tf.keras.Model]): A function that returns a model.
        train_ds (tf.data.Dataset): The training dataset.
        val_ds (tf.data.Dataset): The validation dataset.
        learning_rates (list[float], optional): A list of learning rates to experiment with. Defaults to [0.1, 0.001].
        task_5_history (tf.keras.callbacks.History, optional): The history of the model trained with learning rate 0.01. Defaults to None.

    Returns:
        tuple[list[tuple[str, Model]], list[tuple[str, tf.keras.callbacks.History]]]: A tuple of (models, histories).
        models is a list of tuples of the form (name, model). histories is a list of tuples of the form (name, history).
    """

    histories: 'list[tuple[str, tf.keras.callbacks.History]]' = []
    models: 'list[tuple[str, tf.keras.callbacks.History]]' = []

    if task_5_history:
        histories.append(('0.01', task_5_history))
    else:
        learning_rates.append(0.01)

    for learning_rate in learning_rates:
        model = model_gen_function()
        trained_model, history = task_5_compile_and_train(
            model, train_ds, val_ds, learning_rate=learning_rate, **extra_args)
        histories.append((str(learning_rate), history))
        models.append((str(learning_rate), trained_model))

    task_6_plot_metrics(histories, 'Task 7')

    return models, histories


def task_8_expriment_with_different_momentums(model_gen_function: 'Callable[[], tf.keras.Model]',
                                              train_ds: tf.data.Dataset,
                                              val_ds: tf.data.Dataset,
                                              momentums: 'list[float]' = [
                                                  0.1, 0.5, 0.9],
                                               **extra_args) -> 'tuple[list[tuple[str, tf.keras.Model]], list[tuple[str, tf.keras.callbacks.History]]]':
    """Experiment with 3 different values for the momentum. Plot the results, draw conclusions.

    Args:
        model_gen_function (Callable[[], tf.keras.Model]): A function that returns a model.
        train_ds (tf.data.Dataset): The training dataset.
        val_ds (tf.data.Dataset): The validation dataset.
        momentums (list[float], optional): A list of momentums to experiment with. Defaults to [0.1, 0.5, 0.9].
        
    Returns:
        tuple[list[tuple[str, Model]], list[tuple[str, tf.keras.callbacks.History]]]: A tuple of (models, histories).
    """

    histories: 'list[tuple[str, tf.keras.callbacks.History]]' = []
    models: 'list[tuple[str, tf.keras.callbacks.History]]' = []

    for momentum in momentums:
        model = model_gen_function()
        trained_model, new_history = task_5_compile_and_train(
            model, train_ds, val_ds, momentum=momentum, **extra_args)
        histories.append((str(momentum), new_history))
        models.append((str(momentum), trained_model))

    task_6_plot_metrics(histories, 'Task 8')

    return models, histories


def generate_model_tensor_from_img_dataset(dataset: tf.data.Dataset) -> 'tf.data.Dataset':
    """Generate a dataset of tensors from a dataset of images, using all but last layers of MobileNet v2.

    Args:
        dataset (tf.data.Dataset): The dataset of images.

    Returns:
        tf.data.Dataset: The dataset of tensors. Each has the shape (X, y)=((None, 1280), (None)).
    """

    model: tf.keras.Model = tf.keras.models.load_model(pretrained_model_path)
    flower_input: 'list' = model.input  # (batch_size, 224, 224, 3)
    flower_output: 'list' = model.layers[-2].output  # (batch_size, 1280)
    flower_model_F = tf.keras.Model(inputs=flower_input, outputs=flower_output)
    flower_model_F.trainable = False
    # shape=(batch_size, 224, 224, 3) => (batch_size, 1280)

    Xs = []
    ys = []

    for idx, (X, y) in enumerate(dataset):
        Xs.append(flower_model_F(X))  # shape=(batch_size, 1280)
        ys.append(y)
        print(f'Preprocessing Batch {idx+1}/{len(dataset)}.')

    Xs = tf.concat(Xs, axis=0)  # shape=(dataset_length, 1280)
    ys = tf.concat(ys, axis=0)  # shape=(dataset_length)

    # shape: (X, y)=((None, 1280), (None))
    return tf.data.Dataset.from_tensor_slices((Xs, ys))


def task_9_generate_acceleated_datasets(batch_size: int = 32, **extra_args) -> 'tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]':
    """Prepare datasets based on the MobileNet v2 model with the last layer removed.

    Args:
        batch_size (int, optional): The batch size to use for the datasets. Defaults to 32.

    Returns:
        tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: A tuple of (train_ds, val_ds, test_ds).
        Each dataset has the shape (X, y)=((batch_size, 1280), (batch_size)).
    """

    img_train_ds, img_val_ds, img_test_ds = task_4_prepare_dataset(
        batch_size=batch_size, **extra_args)

    train_ds = generate_model_tensor_from_img_dataset(img_train_ds)
    val_ds = generate_model_tensor_from_img_dataset(img_val_ds)
    test_ds = generate_model_tensor_from_img_dataset(img_test_ds)

    AUTOTUNE = tf.data.AUTOTUNE

    # batch the datasets
    # enable prefetching to improve performance
    if batch_size is not None:
        train_ds = train_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
        test_ds = test_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds

def generate_accelerated_model(classes: int = 5, print_model: bool = False) -> 'tf.keras.Model':
    """Generate a model based on the MobileNet v2 model with the last layer removed.

    Args:
        classes (int, optional): The number of classes in the dataset. Defaults to 5.
        print_model (bool, optional): Whether to print the model summary. Defaults to False.

    Returns:
        tf.keras.Model: The model.
    """

    flower_model = tf.keras.Sequential([
        tf.keras.layers.Dense(classes, activation='softmax'),
    ])

    flower_model.build(input_shape=(None, 1280))

    if print_model:
        print('%' * 50)
        flower_model.summary(show_trainable=True)
        print('%' * 50)

    print('New model created.')

    return flower_model

def task_10_train_on_accelerated_datasets(classes: int = 5, **extra_args) -> 'tuple[list[tuple[str, tf.keras.Model]], list[tuple[str, tf.keras.callbacks.History]]]':
    """Train a model on the accelerated datasets.

    Args:
        classes (int, optional): The number of classes in the dataset. Defaults to 5.

    Returns:
        tuple[list[tuple[str, tf.keras.Model]], list[tuple[str, tf.keras.callbacks.History]]]: A tuple of (models, histories).
    """

    models, histories = task_8_expriment_with_different_momentums(
        generate_accelerated_model, **extra_args
    )

    return models, histories

if __name__ == "__main__":
    pass

    train_configuration = {
        'max_epoch': 500,
        'min_delta': 0.001,
        'patience': 5,
        'tensorboard': False,
        'save_model': True,
        'early_stopping': True,
        'checkpoint_best': True,
        'seed': 42,
    }

    # Put this to True to use accelerated model for all tasks
    accelerated = True

    # autopep8: off
    print("*** Environment Check ***"); env_check()
    print("*** Task 1 ***"); task_1_dataset_sanity_check()
    print("*** Task 2 ***"); task_2_download_pretrained_model()
    print("*** Task 3 ***"); model_gen_function = task_3_replace_last_layer
    if accelerated: print("Accelerate Experiments"); model_gen_function = generate_accelerated_model
    print("*** Task 4 ***"); train_ds, val_ds, test_ds = task_4_prepare_dataset()
    if accelerated: print("Accelerate Experiments"); train_ds, val_ds, test_ds = task_9_generate_acceleated_datasets()
    train_configuration = {
        'max_epoch': 500,
        'min_delta': 0.001,
        'patience': 5,
        'tensorboard': False,
        'save_model': True,
        'early_stopping': True,
        'checkpoint_best': True,
        'seed': 42,
    }
    print("*** Task 5 ***"); task_5_model, task_5_history = task_5_compile_and_train(model=model_gen_function(), train_ds=train_ds, val_ds=val_ds, **train_configuration)
    print("*** Task 6 ***"); task_6_plot_metrics([('Task 5', task_5_history)], 'Task 6')
    train_configuration = {
        'max_epoch': 500,
        'min_delta': 0.001,
        'patience': 5,
        'tensorboard': False,
        'save_model': True,
        'early_stopping': False,
        'checkpoint_best': True,
        'seed': 42,
    }
    print("*** Task 7 ***"); task_7_models, task_7_histories = task_7_expriment_with_different_learning_rates(model_gen_function=model_gen_function, train_ds=train_ds, val_ds=val_ds,  **train_configuration)
    train_configuration = {
        'max_epoch': 150,
        'min_delta': 0.001,
        'patience': 5,
        'tensorboard': False,
        'save_model': True,
        'early_stopping': False,
        'checkpoint_best': True,
        'seed': 42,
    }
    
    print("*** Task 8 ***"); best_model_str_task_7, best_model_task_7 = select_best_model(models=task_7_models, dataset=test_ds); print(f'BEST MODEL: learning_rate = {best_model_str_task_7}'); task_8_expriment_with_different_momentums(model_gen_function=model_gen_function, train_ds=train_ds, val_ds=val_ds, learning_rate=float(best_model_str_task_7),  **train_configuration)
    print("*** Task 9 ***"); accelerated_train_ds, accelerated_val_ds, accelerated_test_ds = task_9_generate_acceleated_datasets()
    print("*** Task 10 ***"); task_10_models, task_10_histories = task_10_train_on_accelerated_datasets(train_ds=accelerated_train_ds, val_ds=accelerated_val_ds, learning_rate=float(best_model_str_task_7), **train_configuration)
    # autopep8: on

# TODO
# analysis
# confusion matrix