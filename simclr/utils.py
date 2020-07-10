import tensorflow as tf
import math
from data_reader import DataReader

class LinearWarmUpCosineDecay():
    def __init__(self, total_iterations, learning_rate):
        """Updates the learning rate with a linear warmup and a cosine decay
        
        Args:
            total_iterations: the total iterations the model will run for
            learning_rate: initial learning rate
        Attributes:
            warmup_iterations: number of iterations for linear warmup
            learning_rate_min: minimum allowed value for learning rate
            total_iterations: the total iterations the model will run for
            learning_rate: initial learning rate
        Returns:
            learning_rate: learning rate for current iteration
        """

        self.warmup_iterations = 1000
        self.learning_rate_min = 0
        self.learning_rate = learning_rate
        self.total_iterations = total_iterations

    def __call__(self, current_iteration):
        if current_iteration <= self.warmup_iterations:
            learning_rate = self.learning_rate*(current_iteration/self.warmup_iterations)
        else:
            learning_rate = self.learning_rate_min + 0.5*(self.learning_rate - self.learning_rate_min)*(1+tf.cos(current_iteration/(self.total_iterations)*math.pi))

        return learning_rate   

def add_to_summary(summary_writer, loss, learning_rate, image1, image2, iteration):
    """Adds loss, learning_rate and images to tensorflow summary"""

    with summary_writer.as_default():
        tf.summary.scalar('Loss', loss, iteration)
        tf.summary.scalar('Learning_rate', learning_rate, iteration)
        tf.summary.image('image1', image1, iteration)
        tf.summary.image('image2', image2, iteration)

def determine_iterations_per_epoch(config):
    """Determine the number of iterations per training epoch
    
    Creates an instance of the DataReader class and iterates over one epoch to 
    determine number of iterations in an epoch. Required in order to 
    accurately decay the learning rate.
    
    Args:
        config: instance of config class
    Returns:
        count: number of iterations in each epoch
    """

    if config.train_file_path:
        data = DataReader(config, config.train_file_path)

    batch = data.read_batch(current_epoch=0, num_epochs=1)
    count = 0

    if config.task == 'pretrain':
        for image, epoch in batch:
            count += 1
    else:
        for image, label, epoch in batch:
            count += 1
    return count
