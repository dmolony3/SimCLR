import tensorflow as tf

def BasicBlock(inputs, num_channels, kernel_size, num_blocks, skip_blocks, regularizer, name):
    """Basic residual block
    
    This creates residual blocks of ConvNormRelu for num_blocks.

    Args:
        inputs: 4-D tensor [B, W, H, CH]
        num_channels: int, number of convolutional filters
        kernel_size: int, size of kernel
        num_blocks: int, number of consecutive 
        skip_blocks: int, this block will be skipped. Used for when stride is >1
        regularizer: tensorflow regularizer
        name: name of the layer
    Returns:
        x: 4-D tensor of the image activation [B, W, H, CH]
    """
    
    x = inputs

    for i in range(num_blocks):
        if i not in skip_blocks:
            x1 = ConvNormRelu(x, num_channels, kernel_size, strides=[1,1], regularizer=regularizer, name=name + '.'+str(i))
            x = tf.keras.layers.Add()([x, x1])
            x = tf.keras.layers.Activation('relu')(x)
    return x

def BasicBlockDown(inputs, num_channels, kernel_size, regularizer, name):
    """Single residual block with strided downsampling
    
    Args:
        inputs: 4-D tensor [B, W, H, CH]
        num_channels: int, number of convolutional filters
        kernel_size: int, size of kernel
        regularizer: tensorflow regularizer
        name: name of the layer
    Returns:
        x: 4-D tensor of the image activation [B, W, H, CH]
    """
    
    x = inputs
    x1 = ConvNormRelu(x, num_channels, kernel_size, strides=[2,1], regularizer=regularizer, name=name+'.0')
    x = tf.keras.layers.Conv2D(num_channels, kernel_size=1, strides=2, padding='same', activation='linear', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=regularizer, name=name+'.0.downsample.0')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name=name+'.0.downsample.1')(x)
    x = tf.keras.layers.Add()([x, x1])
    x = tf.keras.layers.Activation('relu')(x)
    return x
         
def ResNet18(inputs, weight_decay=None):
    """A keras functional model for ResNet-18 architecture
    
    Args:
        inputs: 4-D tensor for input im age [B, W, H, CH]
        weight_decay: float, value for l2 regularization
    Returns:
        x: 2-D tensor after fully connected layer [B, CH]
    """

    if weight_decay:
        regularizer = tf.keras.regularizers.l2(weight_decay) #not valid for Adam, must use  AdamW
    else:
        regularizer = None

    x = tf.keras.layers.ZeroPadding2D(padding=(3,3), name='pad')(inputs)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='valid', activation='linear', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=regularizer, name='conv1')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name='bn1')(x)
    x = tf.keras.layers.Activation('relu', name='relu')(x)
    x = tf.keras.layers.ZeroPadding2D(padding=(1,1), name='pad1')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='valid', name='maxpool')(x)

    x = BasicBlock(x, num_channels=64, kernel_size=3, num_blocks=2, skip_blocks=[], regularizer=regularizer, name='layer1')

    x = BasicBlockDown(x, num_channels=128, kernel_size=3, regularizer=regularizer, name='layer2')
    x = BasicBlock(x, num_channels=128, kernel_size=3, num_blocks=2, skip_blocks=[0], regularizer=regularizer, name='layer2')

    x = BasicBlockDown(x, num_channels=256, kernel_size=3, regularizer=regularizer, name='layer3')
    x = BasicBlock(x, num_channels=256, kernel_size=3, num_blocks=2, skip_blocks=[0], regularizer=regularizer, name='layer3')

    x = BasicBlockDown(x, num_channels=512, kernel_size=3, regularizer=regularizer, name='layer4')
    x = BasicBlock(x, num_channels=512, kernel_size=3, num_blocks=2, skip_blocks=[0], regularizer=regularizer, name='layer4')

    x = tf.keras.layers.GlobalAveragePooling2D(name='avgpool')(x)
    x = tf.keras.layers.Dense(units=1000, use_bias=True, activation='linear', kernel_regularizer=regularizer, name='fc')(x)

    return x

def ResNet34(inputs, weight_decay=None):
    """A keras functional model for ResNet-34 architecture
    
    Args:
        inputs: 4-D tensor for input im age [B, W, H, CH]
        weight_decay: float, value for l2 regularization
    Returns:
        x: 2-D tensor after fully connected layer [B, CH]
    """

    if weight_decay:
        regularizer = tf.keras.regularizers.l2(weight_decay)
    else:
        regularizer = None

    x = tf.keras.layers.ZeroPadding2D(padding=(3,3), name='pad')(inputs)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='valid', activation='linear', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=regularizer, name='conv1')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name='bn1')(x)
    x = tf.keras.layers.Activation('relu', name='relu')(x)
    x = tf.keras.layers.ZeroPadding2D(padding=(1,1), name='pad1')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='valid', name='maxpool')(x)

    x = BasicBlock(x, num_channels=64, kernel_size=3, num_blocks=3, skip_blocks=[], regularizer=regularizer, name='layer1')

    x = BasicBlockDown(x, num_channels=128, kernel_size=3, regularizer=regularizer, name='layer2')       
    x = BasicBlock(x, num_channels=128, kernel_size=3, num_blocks=4, skip_blocks=[0], regularizer=regularizer, name='layer2')       

    x = BasicBlockDown(x, num_channels=256, kernel_size=3, regularizer=regularizer, name='layer3')       
    x = BasicBlock(x, num_channels=256, kernel_size=3, num_blocks=6, skip_blocks=[0], regularizer=regularizer, name='layer3')  

    x = BasicBlockDown(x, num_channels=512, kernel_size=3, regularizer=regularizer, name='layer4')       
    x = BasicBlock(x, num_channels=512, kernel_size=3, num_blocks=3, skip_blocks=[0], regularizer=regularizer, name='layer4')  

    x = tf.keras.layers.GlobalAveragePooling2D(name='avgpool')(x)
    x = tf.keras.layers.Dense(units=1000, use_bias=True, activation='linear', kernel_regularizer=regularizer, name='fc')(x)
    return x

def ResNet18_Cifar10(inputs, weight_decay=None):
    """A keras functional model for ResNet-18 architecture.

    Specifically for cifar10 the first layer kernel size is reduced to 3 
    
    Args:
        inputs: 4-D tensor for input im age [B, W, H, CH]
        weight_decay: float, value for l2 regularization
    Returns:
        x: 2-D tensor after fully connected layer [B, CH]
    """

    if weight_decay:
        regularizer = tf.keras.regularizers.l2(weight_decay) #not valid for Adam, must use  AdamW
    else:
        regularizer = None

    x = tf.keras.layers.ZeroPadding2D(padding=(1,1), name='pad')(inputs)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='valid', activation='linear', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=regularizer, name='conv1')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name='bn1')(x)
    x = tf.keras.layers.Activation('relu', name='relu')(x)

    x = BasicBlock(x, num_channels=64, kernel_size=3, num_blocks=2, skip_blocks=[], regularizer=regularizer, name='layer1')

    x = BasicBlockDown(x, num_channels=128, kernel_size=3, regularizer=regularizer, name='layer2')
    x = BasicBlock(x, num_channels=128, kernel_size=3, num_blocks=2, skip_blocks=[0], regularizer=regularizer, name='layer2')

    x = BasicBlockDown(x, num_channels=256, kernel_size=3, regularizer=regularizer, name='layer3')
    x = BasicBlock(x, num_channels=256, kernel_size=3, num_blocks=2, skip_blocks=[0], regularizer=regularizer, name='layer3')

    x = BasicBlockDown(x, num_channels=512, kernel_size=3, regularizer=regularizer, name='layer4')
    x = BasicBlock(x, num_channels=512, kernel_size=3, num_blocks=2, skip_blocks=[0], regularizer=regularizer, name='layer4')

    x = tf.keras.layers.GlobalAveragePooling2D(name='avgpool')(x)
    x = tf.keras.layers.Dense(units=1000, use_bias=True, activation='linear', kernel_regularizer=regularizer, name='fc')(x)

    return x

def ResNet34_Cifar10(inputs, weight_decay=None):
    """A keras functional model for ResNet-34 architecture.

    Specifically for cifar10 the first layer kernel size is reduced to 3 
    
    Args:
        inputs: 4-D tensor for input im age [B, W, H, CH]
        weight_decay: float, value for l2 regularization
    Returns:
        x: 2-D tensor after fully connected layer [B, CH]
    """

    if weight_decay:
        regularizer = tf.keras.regularizers.l2(weight_decay)
    else:
        regularizer = None

    x = tf.keras.layers.ZeroPadding2D(padding=(1,1), name='pad')(inputs)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='valid', activation='linear', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=regularizer, name='conv1')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name='bn1')(x)
    x = tf.keras.layers.Activation('relu', name='relu')(x)

    x = BasicBlock(x, num_channels=64, kernel_size=3, num_blocks=3, skip_blocks=[], regularizer=regularizer, name='layer1')

    x = BasicBlockDown(x, num_channels=128, kernel_size=3, regularizer=regularizer, name='layer2')       
    x = BasicBlock(x, num_channels=128, kernel_size=3, num_blocks=4, skip_blocks=[0], regularizer=regularizer, name='layer2')       

    x = BasicBlockDown(x, num_channels=256, kernel_size=3, regularizer=regularizer, name='layer3')       
    x = BasicBlock(x, num_channels=256, kernel_size=3, num_blocks=6, skip_blocks=[0], regularizer=regularizer, name='layer3')  

    x = BasicBlockDown(x, num_channels=512, kernel_size=3, regularizer=regularizer, name='layer4')       
    x = BasicBlock(x, num_channels=512, kernel_size=3, num_blocks=3, skip_blocks=[0], regularizer=regularizer, name='layer4')  

    x = tf.keras.layers.GlobalAveragePooling2D(name='avgpool')(x)
    x = tf.keras.layers.Dense(units=1000, use_bias=True, activation='linear', kernel_regularizer=regularizer, name='fc')(x)
    return x

def ConvNormRelu(x, num_channels, kernel_size, strides, regularizer, name):
    """Layer consisting of 2 consecutive ConvNormRelus
    
    Consists of a first convolution followed by batch normalization and relu 
    activation. This is followed by a second convolution and batch normalization

    Args:
        x: 4-D tensor for image/featuremap [B, W, H, CH]
        num_channels: int, number of convolutional filters
        kernel_size: int, size of kernel
        strides: list, value for stride for each convolution
        regularizer: tensorflow regularizer
        name: name of the layer
    Returns:
        x: 4-D tensor of the image activation [B, W, H, CH]
    """
    
    if strides[0] == 2:
        x = tf.keras.layers.ZeroPadding2D(padding=(1,1), name=name+'.pad')(x)
        x = tf.keras.layers.Conv2D(num_channels, kernel_size, strides[0], padding='valid', activation='linear', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=regularizer, name=name+'.conv1')(x)
    else:
        x = tf.keras.layers.Conv2D(num_channels, kernel_size, strides[0], padding='same', activation='linear',  use_bias=False, kernel_initializer='he_normal', kernel_regularizer=regularizer, name=name+'.conv1')(x)      
    x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name=name+'.bn1')(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(num_channels, kernel_size, strides[1], padding='same', activation='linear', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=regularizer, name=name+'.conv2')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5, name=name+'.bn2')(x)
    return x

