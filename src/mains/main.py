import tensorflow as tf
import numpy as np
import cv2
from src.data_loader.data_generator import DataGenerator
from src.models.discriminator_model import DiscriminatorModel
from src.trainers.discriminator_trainer import DiscriminatorTrainer
from src.testers.discriminator_tester import DiscriminatorTester
from src.utils.config import processing_config
from src.utils.logger import  Logger
from src.utils.utils import get_args, freeze_graph


def main():
    try:
        args = get_args()
        # print(args.config)
        # config = processing_config("/media/syrix/programms/projects/GP/Intelligent_Frame_Skipping_Network/configs/config_model.json")
        config = processing_config(args.config)
    except:
        print("Missing or invalid arguments")
        exit(0)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = config.per_process_gpu_memory_fraction

    sess = tf.Session()
    data = DataGenerator(config, training=True, testing=False)
    model = DiscriminatorModel(config)
    model.load(sess)
    logger = Logger(sess, config)

    trainer = DiscriminatorTrainer(sess, model, data, config, logger)
    trainer.train()

    # tester = DiscriminatorTester(sess, model, data, config, logger)
    # tester.test()
    #
    #
    # freeze_graph(config.checkpoint_dir, "output")
if __name__ == '__main__':
    main()


  # "train_data_path":"/media/syrix/programms/projects/GP/SuperStreaming/benchmark/skipping/dataset/train/",
  # "val_data_path":"/media/syrix/programms/projects/GP/SuperStreaming/benchmark/skipping/dataset/val/",
  # "test_data_path":"/media/syrix/programms/projects/GP/SuperStreaming/benchmark/skipping/dataset/test/",
  # "summary_dir":"/media/syrix/programms/projects/GP/Intelligent_Frame_Skipping_Network/saved_models/summary/",
  # "checkpoint_dir":"/media/syrix/programms/projects/GP/Intelligent_Frame_Skipping_Network/saved_models/checkpoint/"
