import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
from math import ceil
import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from argparse import ArgumentParser
from mobilenetv2_unet import MobilenetV2_Unet, relu6
from dataloader import DataGenerator
from keras.callbacks import TensorBoard

from utils.losses import bce_dice_loss, iou_metric
from utils.cyclical_learning_rate import CyclicalLearningRateScheduler
from utils.lr_strategy import CyclicalScheduler
from keras.utils import multi_gpu_model


BATCH_SIZE = 10
LR_MAX = 5e-3
LR_MIN = 0.
EPOCHS = 300
N_CYCLE = 3
INPUT_SHAPE = (224, 224, 3)

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--initial_epoch', type=int, required=True)
    argparser.add_argument('--final_epoch', type=int, required=False, default=EPOCHS)
    argparser.add_argument('--model', type=str, required=False, default=None)
    argparser.add_argument('--lr', type=float, required=False, default=LR_MAX)
    argparser.add_argument('--loss', type=str, default='bce_dice', required=False)
    args = argparser.parse_args()

    # Get the pre_models
    cla_num = 2
    mobilenet = MobilenetV2_Unet(cla_num=cla_num)
    mobilenet.build_model(keras.layers.Input(shape=INPUT_SHAPE), alpha=1.4)
    # multi_gpu_model(mobilenet, gpus=2)
    # # Load saved pre_models if specified
    model_path = "./checkpoints/4/mobilenet-92_loss-0.3151_val_loss-0.2251.h5"
    print(args.model)
    if args.model is not None:
        mobilenet.model = keras.models.load_model(
                    model_path,
                custom_objects={'relu6': relu6},
                compile=False)

    # Freeze encoder layers which are pretrained
    for layer in mobilenet.model.layers:
        layer.trainable = True
    # print(mobilenet.model.summary())

    # Define optimizer and compile pre_models
    # opt = keras.optimizers.Adam(lr=args.lr)
    opt = keras.optimizers.SGD(lr=LR_MAX, momentum=0.9, decay=0, nesterov=False)
    # mobilenet.model.compile(optimizer=opt, loss=bce_dice_loss, metrics=[iou_metric])
    mobilenet.model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # Get data generators
    train_folder = "/home/dep_pic/AI_Data/AI_Train_Data/LIP/LIP/TrainVal_images/train_all"
    train_label_folder = "/home/dep_pic/AI_Data/AI_Train_Data/LIP/LIP/TrainVal_parsing_annotations/train_all"
    train_id_file = "/home/dep_pic/AI_Data/AI_Train_Data/LIP/LIP/TrainVal_images/train_id.txt"
    train_generator = DataGenerator(
        df=train_folder,
        lf=train_label_folder,
        id_file=train_id_file,
        resize=(224, 224),
        shuffle=True,
        augmentations=True,
        cla_num=cla_num
    )

    val_folder = "/home/dep_pic/AI_Data/AI_Train_Data/LIP/LIP/TrainVal_images/val_half"
    val_label_folder = "/home/dep_pic/AI_Data/AI_Train_Data/LIP/LIP/TrainVal_parsing_annotations/train_all"
    val_id_file = "/home/dep_pic/AI_Data/AI_Train_Data/LIP/LIP/TrainVal_images/val_id.txt"
    val_generator = DataGenerator(
        df=val_folder,
        lf=val_label_folder,
        id_file=val_id_file,
        resize=(224, 224),
        shuffle=False,
        augmentations=False,
        cla_num=cla_num
    )

    # Define callbacks
    model_checkpoint = ModelCheckpoint(
        filepath='./checkpoints/'+str(cla_num)+'/mobilenet-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=1)

    cyclical_learning_rate = CyclicalScheduler(lrate_max=LR_MAX,
                                               lrate_min=LR_MIN,
                                               total_epochs=EPOCHS,
                                               n_cycles=N_CYCLE,
                                               epoch_step_init=args.initial_epoch,
                                               save_snapshot=True,
                                               cla_num=cla_num)

    log_dir = "./log_dir/mobilev2_unet/{}".format(cla_num)
    tensorboard = TensorBoard(log_dir=log_dir)
    callbacks = [model_checkpoint, tensorboard, cyclical_learning_rate]

    print('\nTraining...')
    train_history = mobilenet.model.fit_generator(
        generator=train_generator,
        max_queue_size=10,
        workers=8,
        use_multiprocessing=True,
        steps_per_epoch=ceil(len(train_generator) / BATCH_SIZE),
        initial_epoch=args.initial_epoch,
        epochs=args.final_epoch,
        callbacks=callbacks,
        validation_data=val_generator,
        validation_steps=ceil(len(val_generator) / BATCH_SIZE))

    import matplotlib.pyplot as plt
    plt.plot(cyclical_learning_rate.learning_rates)
    plt.xlabel('Step', fontsize=20)
    plt.ylabel('lr', fontsize=20)
    plt.axis([0, args.final_epoch, 0, LR_MAX*1.1])
    plt.xticks(np.arange(0, args.final_epoch, 1))
    plt.grid()
    plt.title('Cyclical Cosine Annealing', fontsize=20)
    plt.savefig('./cyclical.png')