import numpy as np
from tensorflow import keras
from keras import backend as K


def cosine_annealing(global_step,
                             learning_rate_max,
                             learning_rate_min,
                             epoch_per_cycle):
    """
    参数：
    global_step: 记录当前执行的步数。
    learning_rate_min：最小学习率
    learning_rate_max: 最大学习率
    epoch_per_cycle: 每次余弦退火的周期
    """
    epoch_cur = global_step % epoch_per_cycle
    # 实现余弦退火的原理
    learning_rate = learning_rate_min + 0.5 * (learning_rate_max - learning_rate_min) * \
                            (1 + np.cos(np.pi * epoch_cur / float(epoch_per_cycle)))
    return learning_rate


class CyclicalScheduler(keras.callbacks.Callback):
    """
    继承Callback，实现对学习率的调度
    """
    def __init__(self, lrate_max,
                 lrate_min,
                 total_epochs,
                 n_cycles,
                 epoch_step_init=0,
                 save_snapshot=True,
                 cla_num=4):
        super(CyclicalScheduler, self).__init__()
        self.cycles = n_cycles
        self.epochs_per_cycle = total_epochs // self.cycles
        self.epoch_step = epoch_step_init
        self.lr_max = lrate_max
        self.lr_min = lrate_min
        self.learning_rates = []  # learning_rates用于记录每次更新后的学习率，方便图形化观察
        self.save_snapshot = save_snapshot
        self.cla_num = cla_num

    def on_epoch_begin(self, epoch, logs=None):
        """
        更新学习率
        :return:
        """
        lr = cosine_annealing(global_step=self.epoch_step,
                                      learning_rate_max=self.lr_max,
                                      learning_rate_min=self.lr_min,
                                      epoch_per_cycle=self.epochs_per_cycle)
        print(f'epoch {epoch + 1}, lr {lr}')
        K.set_value(self.model.optimizer.lr, lr)
        self.learning_rates.append(lr)

    def on_epoch_end(self, epoch, logs=None):
        """
        更新epoch_step，并选择保存snapshot
        :return:
        """
        self.epoch_step += 1
        if self.save_snapshot:
            if self.epoch_step != 0 and (self.epoch_step + 1) % self.epochs_per_cycle == 0:
                filename = "./checkpoints/{}/snapshot_model_{}.h5".format(self.cla_num,
                                                                          int((self.epoch_step + 1) / self.epochs_per_cycle))
                self.model.save(filename)
                print(f'>saved snapshot {filename}, epoch {self.epoch_step}')
