import tensorflow as tf
from resnet import resnet 

def load_ResNet(version='18', pretrained=False, include_top=False, cifar10=False, weight_decay=None):
    """Reads ResNet model

    Args:
        version: ResNet version to load, either 18 or 34
        pretrained_directory: Path to load pretrained ImageNet weights
        include_top: Set to true if the global average pooling and fully connected layer are to be included
    Returns:
        model: Tensorflow keras model
    Raises:
        NotImplementedError: For any version other than resnet-18 or resnet-34
    """

    if pretrained and cifar10:
        raise ValueError("No pretrained weights exist for cifar-10")

    inputs = tf.keras.Input((None, None, 3))
    if version == '18':
        if cifar10:
            print("Loading CIFAR-10 model")
            output = resnet.ResNet18_Cifar10(inputs, weight_decay)
        else:
            output = resnet.ResNet18(inputs, weight_decay)
    elif version == '34':
        if cifar10:
            output = resnet.ResNet34_Cifar10(inputs, weight_decay)
        else:
            output = resnet.ResNet34(inputs, weight_decay)
    else:
        print("Input must be either 18 or 34")

    model = tf.keras.Model(inputs, output)
    if pretrained:
        model.load_weights('ResNet/resnet'+version)
        output = model(inputs)

    if not include_top:
        layer_name = get_last_activation_layer(model)
        last_activation = model.get_layer(layer_name).output
        model = tf.keras.Model(inputs, last_activation)

    return model


def get_last_activation_layer(model):
    """Return the name of the last activation layer"""

    layer_names = []
    for layer in model.layers:
        layer_names.append(layer.name)

    layer_names = [name for name in layer_names if 'activation' in name]
    return layer_names[-1]
