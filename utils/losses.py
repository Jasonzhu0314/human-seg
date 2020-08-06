import keras.backend as K
import tensorflow as tf

def bce_dice_loss(y_true, y_pred):
    """Calculates binary crossentropy dice loss.

    Calculates as [BCE - log(IoU)]

    y_true: tf.Tensor
        Original labels (mask).
    y_pred: tf.Tensor
        Predicted labels (mask).

    :return: tf.Tensor
    """

    return K.mean(K.binary_crossentropy(y_true, y_pred)) - K.log(iou_metric(y_true, y_pred))

def iou_metric(y_true, y_pred, smooth=1):
    """Calculates Intersection-over-Union metric across batch axis.

    Paper: http://www.cs.umanitoba.ca/~ywang/papers/isvc16.pdf

    y_true: tf.Tensor
        Original labels (mask).
    y_pred: tf.Tensor
        Predicted labels (mask).
    smooth: int, float
        Smoothing constant for boundary cases.

    :return: tf.Tensor
        Mean IoU across batch axis.
    """

    y_pred = K.cast(y_pred, dtype=tf.float32)
    y_true = K.cast(y_true, dtype=tf.float32)
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2])
    union = K.sum(y_true, axis=[1, 2]) + K.sum(y_pred, axis=[1, 2])

    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)