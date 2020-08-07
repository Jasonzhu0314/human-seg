import cv2
from keras.utils import Sequence
import numpy as np
import skimage.transform as skt
from keras.preprocessing.image import img_to_array, load_img
import os
from albumentations import (
    HorizontalFlip,
    Compose,
    ElasticTransform,
    GridDistortion,
    RandomSizedCrop,
    OneOf,
    RandomBrightnessContrast,
    RandomGamma,
    ShiftScaleRotate,
)


class DataGenerator(Sequence):
    """
    基于Sequence的自定义Keras数据生成器
    """
    def __init__(self, df, lf, id_file,
                 batch_size=8, resize=(224, 224),
                 n_channels=3, shuffle=True, augmentations=True):
        """ 初始化方法
        :param df: 存放数据路径
        :param df: 存放标签的路径
        :param id_file: 数据索引文件
        :param to_fit: 设定是否返回标签y
        :param batch_size: batch size
        :param resize: 图像大小
        :param n_channels: 图像通道
        :param n_classes: 标签类别
        :param shuffle: 每一个epoch后是否打乱数据
        """
        self.df = df
        self.lf = lf
        self.id_file = id_file
        self.batch_size = batch_size
        self.resize = resize
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augmentations = augmentations
        self.read_ids()
        self.on_epoch_end()

    def __getitem__(self, index):
        """生成每一批次训练数据
        :param index: 批次索引
        :return: 训练图像和标签
        """
        images = []
        masks = []
        # 生成批次索引
        list_image_ids = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # 生成数据
        for image_id in list_image_ids:
            # Load image
            image_filename = os.path.join(self.df, image_id + '.jpg')
            image = img_to_array(load_img(image_filename)) / 255.

            # Load masks for this image
            parsing_path = os.path.join(self.lf, image_id + '.png')
            mask = cv2.imread(parsing_path, cv2.IMREAD_GRAYSCALE)
            # Make binary mask
            # mask = mask.astype(np.uint8)
            mask = np.where(mask >= 1., 1., 0.)

            # Resize image and mask
            if self.resize:
                image = skt.resize(image, self.resize, anti_aliasing=False)
                mask = np.round(skt.resize(mask, self.resize, anti_aliasing=False))

            # Augmentations
            if self.augmentations:
                aug = Compose([
                    HorizontalFlip(p=0.5),
                    RandomSizedCrop(p=0.35, min_max_height=(10, self.resize[0] - 30), height=self.resize[0], width=self.resize[1]),
                    GridDistortion(p=0.2, border_mode=0, distort_limit=0.1),
                    ElasticTransform(p=0.25, alpha=10, sigma=120 * 0.5, alpha_affine=120 * 0.05),
                    ShiftScaleRotate(p=0.4, border_mode=0, shift_limit=0.04, scale_limit=0.03),
                    # OneOf([
                    #     RandomBrightnessContrast(p=0.8),
                    #     RandomGamma(p=0.9)
                    # ], p=0.5)
                ], p=1)
                augmented = aug(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']

            images.append(image)
            masks.append(mask)

        return np.asanyarray(images), np.expand_dims(masks, axis=-1)

    def __len__(self):
        """每个epoch下的批次数量
        """
        return int(np.floor(len(self.indexes) / self.batch_size))

    def on_epoch_end(self):
        """每个epoch之后更新索引
        """
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def read_ids(self):
        """读取id_file中的image ids
        """
        self.indexes = []
        image_ids = os.listdir(self.df)
        for i in image_ids:
            i = i.split(".")[0]
            self.indexes.append(i)
        # with open(self.id_file, 'r') as f:
        #     for i in f.readlines():
        #         self.indexes.append(i.strip())



