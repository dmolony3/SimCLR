import tensorflow as tf

def gaussian_kernel(kernel_size, sigma):
    """Defines gaussian kernel

    Args:
        kernel_size: Python int, size of the Gaussian kernel
        sigma: Python int, standard deviation of the Gaussian kernel
    Returns:
        2-D Tensor of gaussian kernel
    """

    sigma = tf.cast(sigma, tf.float32)
    x = tf.linspace(-kernel_size/2, kernel_size/2, kernel_size)
    [y, x] = tf.meshgrid(x, x)
    kernel = tf.math.exp(-(tf.math.square(x) + tf.math.square(y))/(2*tf.square(sigma)))
    kernel = kernel/tf.reduce_sum(kernel)

    return kernel

def gaussian_blur(image, kernel_size=3, sigma=3):
    """Convolves a gaussian kernel with input image

    Convolution is performed depthwise

    Args:
        image: 3-D Tensor of image, should by floats
        kernel: 2-D float Tensor for the gaussian kernel
    Returns:
        3-D Tensor image convolved with gaussian kernel
    """
    
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel = tf.expand_dims(tf.stack([kernel, kernel, kernel], axis=-1), axis=-1)
    pointwise_filter = tf.eye(3, batch_shape=[1,1])
    image = tf.expand_dims(image, axis=0)
    image = tf.nn.separable_conv2d(image, kernel, pointwise_filter, strides=[1,1,1,1], padding='SAME')
    image = tf.squeeze(image, axis=0)

    return image

def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
    """Defines gaussian kernel

    Args:
        kernel_size: Python int, size of the Gaussian kernel
        sigma: Python int, standard deviation of the Gaussian kernel
    Returns:
        2-D Tensor of gaussian kernel
    """
    
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)


def apply_blur(img, kernel_size, sigma):
    """Convolves a gaussian kernel with input image

    Convolution is performed depthwise

    Args:
        img: 3-D Tensor of image, should by floats
        kernel: 2-D float Tensor for the gaussian kernel
    Returns:
        img: 3-D Tensor image convolved with gaussian kernel
    """
    
    blur = _gaussian_kernel(kernel_size, sigma, 3, img.dtype)
    img = tf.nn.depthwise_conv2d(img[None], blur, [1,1,1,1], 'SAME')
    return img[0]
