# TODO 去掉
import os

import numpy as np
from sklearn.metrics import mean_absolute_error
from utils.logger import Logger
from config.config import Config
from utils import draw_img
from utils import load_data
from utils import load_model_
from utils import split_data

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class Benchmark:
    def __init__(self, ratio):
        self._x_scaler, self._y_scaler, self._pre_model = load_model_.load_scaler_model()
        self._ratio = ratio
        # 读取csv数据
        self._data = load_data.load_data()

    def _process_data(self):
        """
        对读取的csv数据做初步处理
        :return:
        """
        # 对读取后的数据进行处理data
        # 读取除了最后一行的所有数据
        _x = self._data.iloc[:, :-1].values
        _y = self._data.iloc[:, -1].values

        _y = _y.reshape((_y.shape[0], 1))

        # x归一化处理
        _x = self._x_scaler.transform(_x)
        return _x, _y

    def _get_structure_data(self):
        """
        获取可以用于模型输入的三维数据
        :return:
        """
        _x, _y = self._process_data()
        _x = split_data.split_shift_data(Config.step, _x)
        return _x, _y[Config.step - 1:]

    def predict_(self):
        _x, _y = self._get_structure_data()
        pre_data = self._pre_model.predict(_x)
        pre_data = self._y_scaler.inverse_transform(pre_data)
        return _x, _y, pre_data

    def _count_pre_and_true_below_threshold_num(self, vis_threshold, true_y, pre_y):
        """
        查找需要的部分数据
        :param vis_threshold:
        :return: int, int
        """

        less_threshold_count_pre_num = 0
        less_threshold_count_true_num = 0

        # 从真实值中获取小于5k的数值
        for i in range(true_y.shape[0]):
            if true_y[i] < vis_threshold:
                less_threshold_count_true_num += 1
                if pre_y[i] < vis_threshold + vis_threshold * self._ratio:
                    less_threshold_count_pre_num += 1

        return less_threshold_count_true_num, less_threshold_count_pre_num

    @staticmethod
    def _get_threshold_data(vis_threshold, true_y, pre_y):
        """
        返回需要的数据列表
        :param vis_threshold:
        :return:
        """
        less_threshold_count_pre_list = list()
        less_threshold_count_true_list = list()

        for i in range(len(true_y)):
            if pre_y[i] < vis_threshold:
                less_threshold_count_pre_list.append(pre_y[i])
                less_threshold_count_true_list.append(true_y[i])

        return less_threshold_count_true_list, less_threshold_count_pre_list

    def predict_acc(self, vis_threshold, true_y, pre_y, draw=False, ):
        """
        打印预测准确率
        :param pre_y:
        :param true_y:
        :param draw:
        :param vis_threshold: int
        :return:
        """
        less_threshold_count_true, less_threshold_count_pre = self._count_pre_and_true_below_threshold_num(
            vis_threshold, true_y, pre_y)

        acc = (less_threshold_count_pre / less_threshold_count_true) * 100
        Logger.info("{}米以下真实值数量: {}".format(vis_threshold, less_threshold_count_true))
        Logger.info("与真实值匹配的{}(上浮动+{})m以下预测值数量: {}".format(vis_threshold, int(vis_threshold * self._ratio),
                                                     less_threshold_count_pre))
        Logger.info("Acc: {:.3f}%".format(acc))
        print('=' * 30)
        if draw:
            draw_img.draw_img(true_y, pre_y)

    def predict_mae(self, vis_threshold, true_y, pre_y):
        """
        输出mse
        :param pre_y:
        :param true_y:
        :param vis_threshold:
        :return:
        """
        less_threshold_count_true_list, less_threshold_count_pre_list = self._get_threshold_data(vis_threshold, true_y,
                                                                                                 pre_y)
        # print(less_threshold_count_true_list)
        mse = mean_absolute_error(less_threshold_count_true_list, less_threshold_count_pre_list) / 1000

        Logger.info("{}米以下数据误差: {:.3f} 公里".format(vis_threshold, mse))

    def modulation_true_value(self, vis_threshold, subtract_num, true_y, pre_y, draw=False):
        """
        修正真实值的误差
        :param draw:
        :param vis_threshold:
        :param subtract_num:
        :param true_y:
        :param pre_y:
        :return:
        """
        print('\n')
        print('=' * 10 + '修正后性能评估结果' + '=' * 10)
        _pre_y = pre_y.tolist()
        # 统计共预测出多少个低于阈值的海雾数据
        after_count = 0
        before_count = 0
        for i in range(len(_pre_y)):
            # 如果真实值大于设定的阈值就减去设定的误差值
            # true_y[i] = [1000] 单个元素是个列表
            if _pre_y[i][0] > vis_threshold:
                _pre_y[i][0] = _pre_y[i][0] - subtract_num

        for j in pre_y:
            if j < vis_threshold:
                before_count += 1
        _pre_y = np.array(_pre_y)

        for k in _pre_y:
            if k < vis_threshold:
                after_count += 1

        # pre_y是修正后的能见度预测值, true_y是真实的能见度数值
        self.predict_acc(vis_threshold, true_y, _pre_y, draw)
        self.predict_mae(vis_threshold, true_y, _pre_y)
        # 修正前数据
        _, vis_before_change = self._count_pre_and_true_below_threshold_num(
            vis_threshold, true_y, pre_y)
        # 修正后的数据
        _, vis_after_change = self._count_pre_and_true_below_threshold_num(vis_threshold, true_y, _pre_y)

        Logger.info("修正前共预测出低于{}m的海雾数据{}个".format(vis_threshold, before_count))
        Logger.info("修正后共预测出低于{}m的海雾数据{}个".format(vis_threshold, after_count))

        Logger.info("修正前共预测出{}m(上浮动{}m)以下的海雾数据{}个".format(vis_threshold,
                                                    int(vis_threshold * self._ratio), vis_before_change))
        Logger.info("修正后预测出{}m(上浮动{}m)以下的海雾数据{}个".format(vis_threshold,
                                                   int(vis_threshold * self._ratio), vis_after_change))


if __name__ == '__main__':
    _threshold = 5000
    _modulation_value = 500
    _ratio = .3

    benchmark = Benchmark(_ratio)
    _, y, pre = benchmark.predict_()
    benchmark.predict_acc(_threshold, y, pre)
    benchmark.predict_mae(_threshold, y, pre)
    benchmark.modulation_true_value(_threshold, _modulation_value, y, pre)
