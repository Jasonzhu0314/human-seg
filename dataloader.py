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
    ShiftScaleRotate,
)


class DataGenerator(Sequence):
    """
    基于Sequence的自定义Keras数据生成器,
    1.Hat,2.Hair,3.Glove,4.Sunglasses,5.UpperClothes
    6.Dress,7.Coat,8.Socks,9.Pants,10.Jumpsuits,11.Scarf
    12.Skirt,13.Face,14.Left-arm,15.Right-arm,16.Left-leg
    17.Right-leg,18.Left-shoe,19.Right-shoe
    """
    def __init__(self, df, lf, id_file,
                 batch_size=8, resize=(224, 224),
                 n_channels=3, shuffle=True, augmentations=True,
                 cla_num=2):
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
        self.cla_num = cla_num

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

            if self.cla_num == 2:
                mask = np.where(mask >= 1, 1, 0)
            mask_label = np.zeros(mask.shape + (self.cla_num,))
            if self.cla_num == 4:  # 1.Hat,2.Hair,13.Face
                label_num = {1: 1, 2: 2, 13: 3}
                for i in range(20):
                    if i in label_num.keys():
                        mask_label[mask == i, label_num[i]] = 1
                    else:
                        mask_label[mask == i, 0] = 1
            else:
                for i in range(self.cla_num):
                    mask_label[mask == i, i] = 1

            # Resize image and mask
            if self.resize:
                image = skt.resize(image, self.resize, anti_aliasing=False)
                mask_label = np.round(skt.resize(mask_label, self.resize, anti_aliasing=False))

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
                augmented = aug(image=image, mask=mask_label)
                image = augmented['image']
                mask_label = augmented['mask']

            images.append(image)
            masks.append(mask_label)

        return np.asanyarray(images), np.asarray(masks)

    def __len__(self):
        """每个epoch下的批次数量
        """
        return int(np.floor(len(self.indexes) / self.batch_size))

    def on_epoch_end(self):
        """每个epoch之后更新索引
        """
        if self.shuffle:
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



