import tensorflow as tf
from data_reader import DataReader
from utils import determine_iterations_per_epoch

#policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
#tf.keras.mixed_precision.experimental.set_policy(policy) 

@tf.function
def train_step_regression(model, images, labels):
    """Perform one training step using mean squared error. Currently not used.

    Args:
        model: a tensorflow keras model
        images: First augmented image batch [B, H, W, CH]
        labels: Corresponding labels for input images, can be either images or integers
    Returns:
        loss: sum of training loss and regularization loss
        gradients: gradients of model weights
    """

    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss = tf.reduce_mean(tf.nn.l2_loss(labels - logits))
        reg_loss = tf.add_n(model.losses) if model.losses else 0
        loss = loss + reg_loss

    gradients = tape.gradient(loss, model.trainable_variables)

    return loss, gradients

@tf.function
def train_step(model, images, labels, num_classes):
    """Perform one training step using standard cross-entropy loss.

    Args:
        model: a tensorflow keras model
        images: Input images batch [B, H, W, CH]
        labels: Corresponding labels for input images, can be either images [B, H, W, CH] or integers [B, CH]
        num_classes: Number of classes
    Returns:
        loss: sum of training loss and regularization loss
        gradients: gradients of model weights
    """

    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        labels = tf.one_hot(labels, num_classes)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits), axis=0)
        loss = tf.reduce_sum(loss)
        reg_loss = tf.add_n(model.losses) if model.losses else 0
        loss = loss + reg_loss

    gradients = tape.gradient(loss, model.trainable_variables)

    return loss, gradients

@tf.function
def val_step(model, images, labels, num_classes):
    """Perform one validation step using standard cross-entropy loss.

    Args:
        model: tensorflow keras model
        images: Input images
        labels: Corresponding labels for input images, can be either images or integers
        num_classes: Number of classes
    Returns:
        loss: training loss
        preds: model prediction, can be either a class label or an image labelmap
    """

    logits = model(images, training=False)
    preds =  tf.argmax(logits, -1)
    labels = tf.one_hot(labels, num_classes)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits), axis=0)
    loss = tf.reduce_sum(loss)

    return loss, preds

def finetune(model, config):
    """Finetunes the model for either image classification or segmentation

    This function first creates an instance of the DataReader class.
    If validation data is provided an instance for this is also created.
    Pretrained weights are restored. Depending on the task either a dense layer or convolutional 
    layers are added to the model. If present fine-tuned weights are restored.
    The model is trained over the entire dataset for the pre-specified number of epochs
    and the validation dataset is evaluated after each epoch.

    Args:
        model: a tensorflow keras model
        config: model configurations
    Raises:
        OSError: If pretrained save path does not contain checkpoint or is not set to None
    """


    # Restore the pretrained model first
    checkpoint, manager = restore_weights(model, config.pretrain_save_path)
    if not manager.latest_checkpoint and config.pretrain_save_path is not None:
        raise OSError("No pretrained saved checkpoints found in {}".format(config.pretrain_save_path))
    
    if config.freeze:
        model.trainable = False
        
    # attach segmentation or classification head
    resnet_output = model.output
    if config.task == "segmentation":
        layer1 = tf.keras.layers.Conv2D(filters=config.num_classes, kernel_size=1, padding='SAME', use_bias=False)(resnet_output)
        model_output = tf.keras.layers.Lambda(lambda image: tf.image.resize(image, [config.input_size[0], config.input_size[1]]))(layer1)
    elif config.task == "classification" or config.task == 'regression':
        layer1 = tf.keras.layers.GlobalAveragePooling2D()(resnet_output)
        if config.include_nonlinearity:
            layer2 = tf.keras.layers.Dense(units=256, use_bias=False, name='nonlinear')(layer1)
            model_output = tf.keras.layers.Dense(units=config.num_classes, use_bias=False, name='output')(layer2)
        else:
            model_output = tf.keras.layers.Dense(units=config.num_classes, use_bias=False, name='output')(layer1)
        
    model = tf.keras.Model(model.input, model_output)
    
    iterations_per_epoch = determine_iterations_per_epoch(config)*config.num_epochs
    total_iterations = iterations_per_epoch*config.num_epochs

    learning_rate = tf.keras.experimental.CosineDecay(config.learning_rate, total_iterations)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)    

    # Restore the finetuned model if weights exist
    checkpoint, manager = restore_weights(model, config.finetune_save_path, optimizer)
    summary_writer = tf.summary.create_file_writer(config.finetune_save_path)

    current_epoch = tf.cast(tf.floor(optimizer.iterations/iterations_per_epoch), tf.int64)
    train_data = DataReader(config, config.train_file_path)
    train_batch = train_data.read_batch(current_epoch, config.num_epochs)

    if config.val_file_path:
        compute_validation = True
        val_data = DataReader(config, config.val_file_path)
    else:
        compute_validation = False
        
    epoch_loss_train = []
    for image, label, epoch in train_batch:
        loss, grads = train_step(model, image, label, config.num_classes)
        epoch_loss_train.append(loss)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))  

        if tf.reduce_all(tf.equal(epoch, current_epoch + 1)):
            epoch_loss_train = sum(epoch_loss_train)/len(epoch_loss_train)
            print("Training loss after epoch {}: {}".format(current_epoch, epoch_loss_train))

            if compute_validation:
                epoch_loss_val = []
                val_batch = val_data.read_batch(current_epoch=0, num_epochs=1)
                val_accuracy = tf.keras.metrics.Accuracy()

                for image, label, epoch in val_batch:
                    loss, pred = val_step(model, image, label, config.num_classes)
                    epoch_loss_val.append(loss)
                    val_accuracy(pred, label)

                epoch_loss_val = sum(epoch_loss_val)/len(epoch_loss_val)
                print("Validation loss after epoch {}: {}".format(current_epoch, epoch_loss_val))
                print("Validation accuracy after epoch {}: {}".format(current_epoch, val_accuracy.result()))
                with summary_writer.as_default():
                    tf.summary.scalar('training loss', epoch_loss_train, current_epoch)
                    tf.summary.scalar('validation loss', epoch_loss_val, current_epoch)
                    tf.summary.scalar('validation accuracy', val_accuracy.result(), current_epoch)

            epoch_loss_train = []
            current_epoch += 1
            
    print("Saved checkpoint to {}".format(config.finetune_save_path))
    manager.save()

def restore_weights(model, save_path, optimizer=None):
    """Restores weights to a saved model. Optionally restores optimizer

    Args:
        model: A tensorflow keras model 
        save_path: The path to the saved weights
        optimizer: A tensorflow keras optimizer 
    Returns:
        checkpoint: tf.train.Checkpoint
        manager: tf.train.checkpointManager for saved model
    """

    if optimizer is not None:
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    else:
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), net=model)

    manager = tf.train.CheckpointManager(checkpoint, save_path, max_to_keep=3)

    if manager.latest_checkpoint:
        status = checkpoint.restore(manager.latest_checkpoint)
        status.expect_partial()
        print('Restored weights from {}'.format(manager.latest_checkpoint))
    else:
        print('No checkpoint found in {}, creating a new checkpoint manager'.format(save_path))
        if optimizer is not None:
            checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
        else:
            checkpoint = tf.train.Checkpoint(step=tf.Variable(1), net=model)

        manager = tf.train.CheckpointManager(checkpoint, save_path, max_to_keep=3)

    return checkpoint, manager
    
