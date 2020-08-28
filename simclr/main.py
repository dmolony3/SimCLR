import argparse
from config import Config
from train import pretrain
from train_cifar10 import pretrain_cifar10
from finetune import finetune
from finetune_cifar10 import finetune_cifar10
from resnet_loader import load_ResNet

def run(config):
    """Runs the model for either pretrain or segmentation/classifiction tasks
    
    Loads the ResNet backbone and based on task specification performs either
    pretrain or segmentation/classification finetuning. 

    Args:
        config: instance of config class
    Raises:
        ValueError: In segmentation/classification tasks if number of classes are not specified
    """ 

    # load the Resnet backbone
    model = load_ResNet(config.model, config.imagenet_path, include_top=False, cifar10=config.cifar10, weight_decay=config.weight_decay)

    if config.task == 'pretrain':

        print("Pretraining model")

        if config.cifar10:
            pretrain_cifar10(model, config)
        else:
            pretrain(model, config)
    
    elif config.task == 'segmentation' or 'classification':

        print("Fine-tuning model")

        if config.num_classes is None:
            raise ValueError("The number of classes must be set")

        if config.cifar10:
            finetune_cifar10(model, config)
        else:
            finetune(model, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="simCLR for self-supervised learning", epilog="For pretraining on cifar10: python main.py --task=pretrain --cifar10=True --batch_size=256 --num_epochs=1000; For linear evaluation on cifar10: python main.py --task=classification --cifar10=True --batch_size=256 --num_classes=10 --pretrain_save_path=logs/pretrain --finetune_save_path=logs/finetune --freeze=True")
    parser.add_argument('--task', type=str, default='pretrain', help="Specify the task; either pretrain, classification or segmentation")
    parser.add_argument('--model', type=str, default='18', help="Specify the ResNet model to be used; either 18 or 34")
    parser.add_argument('--imagenet_path', type=str, default='', help="Specify the path pretrained Imagenet weights")
    parser.add_argument('--input_size', type=str, default="500 500", help="Specify the input image size in the form of row column")
    parser.add_argument('--crop_size', type=str, default="200 200",help="Specify the size that the image will be cropped too")
    parser.add_argument('--batch_size', type=int, default=128 ,help="Specify the batch size")
    parser.add_argument('--weight_decay', type=float, default=1e-6 ,help="Specify the value for weight decay")
    parser.add_argument('--learning_rate', type=float, default=0.1 ,help="Specify the value for the initial learning rate")
    parser.add_argument('--train_file_path', type=str, default='', help="Specify the path to the training files containing a list of image filenames to be loaded, where each line is an image path")
    parser.add_argument('--val_file_path', type=str, default='', help="Specify the path to the validation files containing a list of image filenames to be loaded, where each line is an image path")
    parser.add_argument('--pretrain_save_path', type=str, default='logs/pretrain', help="Specify the path to the directory containing the pretrain save checkpoint")
    parser.add_argument('--finetune_save_path', type=str, default='logs/finetune', help="Specify the path to the directory containing the finetune save checkpoint")
    parser.add_argument('--cifar10', type=bool, default=False, help="Specify whether the cifar10 dataset should be used")
    parser.add_argument('--num_classes', type=int, default=None, help="Specify the number of classes for classification or segmentation")
    parser.add_argument('--num_epochs', type=int, default=100, help="Specify the number of epochs to train the model for")
    parser.add_argument('--freeze', action='store_true', help="Freezes base model weights")
    parser.add_argument('--blur', dest='apply_blur', action='store_true', help="Removes random gaussian blur image augmentation")
    parser.add_argument('--no-blur', dest='apply_blur', action='store_false', help="Applies random gaussian blur image augmentation")
    parser.add_argument('--rotate', action='store_false', help="Applies random rotation image augmentation")
    parser.add_argument('--jitter', dest='apply_jitter', action='store_true', help="Applies jitter image augmentation")
    parser.add_argument('--no-jitter', dest='apply_jitter', action='store_false', help="Removes jitter image augmentation")
    parser.add_argument('--crop', dest='apply_crop', action='store_true', help="Applies random crop image augmentation")
    parser.add_argument('--no-crop', dest='apply_crop', action='store_false', help="Removes random crop image augmentation")
    parser.add_argument('--flip', dest='apply_flip', action='store_true', help="Applies random flip image augmentation")
    parser.add_argument('--no-flip', dest='apply_flip', action='store_false', help="Removes random flip image augmentation")
    parser.add_argument('--noise', dest='apply_noise', action='store_true', help="Applies guassian noise image augmentation")
    parser.add_argument('--no-noise', dest='apply_noise', action='store_false', help="Removes guassian noise image augmentation")
    parser.add_argument('--colordrop', dest='apply_colordrop', action='store_true', help="Applies color drop image augmentation")
    parser.add_argument('--no-colordrop', dest='apply_colordrop', action='store_false', help="Removes color drop image augmentation")
    parser.add_argument('--temperature', type=float, help="Sets the temperature for cross-entropy loss function")
    parser.add_argument('--strength', type=float, help="Sets the strength of augmentations")

    args = parser.parse_args()

    config = Config(args.task)
    config.input_size = [int(val) for val in args.input_size.split(' ')]
    config.crop_size = [int(val) for val in args.crop_size.split(' ')]
    config.batch_size = args.batch_size
    config.num_classes = args.num_classes
    config.freeze = args.freeze
    config.num_epochs = args.num_epochs
    config.weight_decay = args.weight_decay
    config.learning_rate = args.learning_rate
    config.pretrain_save_path = args.pretrain_save_path
    config.finetune_save_path = args.finetune_save_path
    config.cifar10 = args.cifar10
    config.imagenet_path = args.imagenet_path
    config.train_file_path = args.train_file_path
    config.val_file_path = args.val_file_path
    config.crop = args.apply_crop
    config.blur = args.apply_blur
    config.rotate = args.rotate
    config.jitter = args.apply_jitter
    config.flip = args.apply_flip
    config.noise = args.apply_noise
    config.colordrop = args.apply_colordrop
    config.temperature = args.temperature
    config.strength = args.strength
    config.model = args.model

    if not config.cifar10 and config.train_file_path == '':
        raise ValueError("Either a train file must be provided or cifar10 must be set to True")

    run(config)
