import tensorflow as tf
import tensorflow_addons as tfa
from image_utils import apply_blur

class DataReader():
    """Reads data from text files and performs data augmentation
        
    User should provide an instance of the config class and the file_path to 
    a file where each line corresponds to an image. Images are loaded by the 
    read_batch method which will perform data augmentation and return a batch 
    of pairs of augmented images. The text file can also contain label data, 
    separated by a comma from the image.

    Args:
        config: instance of config class containing simclr configurations
        file_path:path to data file containing images

    Attributes:
        batch_size: number of images in single batch
        config: instance of config class containing simclr configurations
        file_path: path to data file containing images

    Methods:
        read: reads and decodes an image given a filename as input
        read_batch: returns a batch of images
        read_cifar10: returns a batch of images from cifar10
    """

    def __init__(self, config, file_path=None):
        super(DataReader, self).__init__()
        self.batch_size = config.batch_size
        self.config = config
        self.file_path = file_path

    @classmethod
    def read(self, fname):
        """reads image, decodes and normalizes between 0 and 1
        
        Args:
            fname: path to the image file
        Returns:
            image: 3D tensor of image  
        """

        image = tf.io.read_file(fname)
        image = tf.io.decode_image(image)
        image = tf.cast(image, dtype=tf.float32)
        image = image/255 # assumes image is uint8

        return image

    def color_jitter(self, image):
        """Apply color jitter operation"""
        
        image = tf.image.random_brightness(image, max_delta=0.8*self.config.strength)
        image = tf.image.random_contrast(image, lower=1-0.8*self.config.strength, 
                                        upper=1+0.8*self.config.strength)
        image = tf.image.random_saturation(image, lower=1-0.8*self.config.strength, 
                                           upper=1+0.8*self.config.strength)
        image = tf.image.random_hue(image, max_delta=0.2*self.config.strength)

        return image

    def color_drop(self, image):
        """drop color channel by convert to grayscale"""

        image = tf.image.rgb_to_grayscale(image)
        image = tf.tile(image, [1, 1, 3])

        return image

    def cutout(self, image, random_apply):
        """Randomly applies cutout of image, cutout is 3 times smaller than image"""

        mask_size = [int(self.config.crop_size[0]/6)*2]*2 # 3 times smaller, must be divisible by 2
        crop_size = self.config.crop_size
        offset = tf.random.uniform(minval=self.config.crop_size[0]//10, 
                                   maxval=crop_size[0] - crop_size//10, 
                                   shape=[2], dtype=tf.int32)
        image = tf.cond(self.config.cutout_prob>random_apply[0], 
                        lambda: tfa.image.cutout(image, [mask_size[0], mask_size[1]], 
                        [offset[0], offset[1]]), 
                        lambda: tf.identity(image))
        
        return image

    def rotate(self, image, random_apply):
        """Applies a rotation to the image specified by a random angle"""

        image = tf.expand_dims(image, 0)
        angle = tf.random.uniform(minval = 0, 
                                  maxval = self.config.max_rotation, 
                                  shape = [1], dtype=tf.float32)
        image = tf.cond(self.config.rotate_prob>random_apply[0], 
                        lambda: tfa.image.rotate(image, angle), 
                        lambda: tf.identity(image))
        image = tf.squeeze(image, 0)

        return image

    def add_noise(self, image, random_apply):
        """Adds gaussian noise with 25% probability"""

        noise = tf.random.normal(mean=0, stddev=0.025, shape=tf.shape(image))
        image = tf.cond(0.75>random_apply[0], lambda: tf.add(image, noise), 
                        lambda: tf.identity(image))

        return image

    def random_crop_resize(self, image):
        """Crops an image and returns the cropped image"""

        bbox = tf.constant([0, 0, 1, 1], dtype=tf.float32)
        bbox = tf.expand_dims(tf.expand_dims(bbox, 0), 0)
        min_object_covered = 0.1 # value from authors
        begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
            image.shape, 
            bounding_boxes=bbox, 
            min_object_covered=min_object_covered, 
            aspect_ratio_range=[3/4, 4/3])
        image = tf.slice(image, begin, size)
        image = tf.image.resize(image, [self.config.crop_size[0], 
                                self.config.crop_size[1]])

        return image

    def random_blur(self, image, random_apply):
        """Applies gaussian blurring randomnly"""

        sigma = tf.random.uniform(minval=0.01, maxval=2.0, shape=[1])
        kernel_size = self.config.crop_size[0]//10
        image = tf.cond(self.config.blur_prob>random_apply[0], 
                        lambda: apply_blur(image, kernel_size, sigma), 
                        lambda: tf.identity(image))        

        return image

    def augment_ops(self, image):
        """Performs data augmentation based on config settings, returns augmented image
        
        Currently implements 8 data augmentation operations. Whether these operations are
        performed or not depends on the config settings. Each operation is randomly applied
        except for cropping which is always applied.

        Args:
            image: 3D tensor of single image [H, W, CH]
        Returns:
            image_aug: 3D tensor of single augmented image  [H, W, CH]
        """

        image_aug = tf.cast(tf.identity(image), tf.float32)
        image_aug.set_shape((self.config.input_size[0], self.config.input_size[1], 3))

        if self.config.crop:
            image_aug = self.random_crop_resize(image_aug)
        if self.config.flip:
            random_apply = tf.random.uniform(minval=0, maxval=1, shape=[1])
            image_aug = tf.cond(self.config.flip_prob>random_apply[0], 
                                lambda: tf.image.flip_left_right(image_aug), 
                                lambda: tf.identity(image_aug))
        if self.config.blur:
            random_apply = tf.random.uniform(minval=0, maxval=1, shape=[1])
            image_aug = self.random_blur(image_aug, random_apply)
        if self.config.noise:
            random_apply = tf.random.uniform(minval=0, maxval=1, shape=[1])
            image_aug = self.add_noise(image_aug, random_apply)
        if self.config.rotate:
            random_apply = tf.random.uniform(minval=0, maxval=1, shape=[1])
            image_aug = self.rotate(image_aug, random_apply)
        if self.config.jitter:
            random_apply = tf.random.uniform(minval=0, maxval=1, shape=[1])
            image_aug = tf.cond(self.config.jitter_prob>random_apply[0], 
                                lambda: self.color_jitter(image_aug), 
                                lambda: tf.identity(image_aug))
        if self.config.colordrop:
            random_apply = tf.random.uniform(minval=0, maxval=1, shape=[1])
            image_aug = tf.cond(self.config.color_prob>random_apply[0], 
                                lambda: self.color_drop(image_aug), 
                                lambda: tf.identity(image_aug))
        if self.config.cutout:
            random_apply = tf.random.uniform(minval=0, maxval=1, shape=[1])
            image_aug = self.cutout(image_aug, random_apply)
        
        image_aug = tf.clip_by_value(image_aug, 0, 1)
        return image_aug

    def augment(self, image):
        """Performs data augmentation, returns 2 augmented images"""

        image1 = self.augment_ops(image)
        image2 = self.augment_ops(image)

        return image1, image2

    def read_batch(self, current_epoch, num_epochs):
        """Processes data for pretraining, classification or segmentation.
        
        This will read in the text file and depending on the task will return 
        augmented images or if the task is classification or segmentation the 
        class or image labels. A counter for the epoch is also returned. This 
        increments after each epoch.
         
        Args:
            current_epoch: Epoch at which epoch counter begins
            num_epochs: Epoch at which epoch counter finishes
        Returns:
            data: tuple containing current epoch and the augmented images and labels if required
        """

        textdata = tf.data.TextLineDataset(self.file_path)
        textdata = textdata.shuffle(5000, reshuffle_each_iteration=True)

        epoch_counter = tf.data.Dataset.range(current_epoch, num_epochs)
        if self.config.task == 'pretrain':
            data = textdata.map(lambda fnames: self.read(fnames))
            data = data.map(lambda image: self.augment(image))
            data = epoch_counter.flat_map(
                lambda i: tf.data.Dataset.zip((data, 
                tf.data.Dataset.from_tensors(i).repeat())))
        else:
            textdata = textdata.map(lambda line: tf.strings.split(line, ','))
            data_image = textdata.map(lambda fnames: self.read(fnames[0]))

            if self.config.task == 'classification' or self.config.task == 'regression':
                data_label = textdata.map(lambda classes: classes[1])
                data_label = data_label.map(lambda classes: tf.strings.to_number(classes))
            elif self.config.task == 'segmentation':
                data_label = textdata.map(lambda fnames: self.read(fnames[1]))            
            data = epoch_counter.flat_map(lambda i: tf.data.Dataset.zip(
                (data_image, data_label, 
                tf.data.Dataset.from_tensors(i).repeat())))

        data = data.batch(self.batch_size, drop_remainder=True)

        return data

    def read_cifar10(self, cifar10_images, cifar10_labels=None, current_epoch=0, num_epochs=1):
        """Reads cifar10 data and and processes data either for pretraining or classification 
        
        Args:
            cifar10_images: 4D tensor/array of cifar10 images  [B, H, W, CH]
            cifar10_labels: 1D tensor/array of cifar10 labels [B]
            current_epoch: Epoch at which epoch counter begins
            num_epochs: Epoch at which epoch counter finishes
        Returns:
            data: tuple containing current epoch and the augmented images and labels if required
        """

        epoch_counter = tf.data.Dataset.range(current_epoch, num_epochs)
        if self.config.task == 'pretrain':
            data = tf.data.Dataset.from_tensor_slices(cifar10_images)
            data = data.shuffle(5000, reshuffle_each_iteration=True)
            data = data.map(lambda i: tf.cast(i, tf.float32)/255)
            data = epoch_counter.flat_map(lambda i: tf.data.Dataset.zip(
                (data.map(self.augment, 
                num_parallel_calls=tf.data.experimental.AUTOTUNE), 
                tf.data.Dataset.from_tensors(i).repeat())))
        elif self.config.task =='classification':
            data_image = tf.data.Dataset.from_tensor_slices(cifar10_images)
            data_label = tf.data.Dataset.from_tensor_slices(cifar10_labels)
            data_image = data_image.map(lambda i: tf.cast(i, tf.float32)/255)
            data = epoch_counter.flat_map(lambda i: tf.data.Dataset.zip(
                (data_image, data_label, 
                tf.data.Dataset.from_tensors(i).repeat())))

        data = data.batch(self.batch_size, drop_remainder=True)

        return data
