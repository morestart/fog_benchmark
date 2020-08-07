import os
from numpy.random import seed
from benchmark.benchmark import Benchmark

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if __name__ == '__main__':
    kilometre = 1000
    ratio = 0.2
    modulation_value = 500

    seed(5)
    benchmark = Benchmark(ratio)
    _, y, pre = benchmark.predict_()

    # 预报准确率
    less_threshold_count_pre, less_threshold_count_true = benchmark.predict_acc(kilometre, y, pre, draw=False)
    # 预报mae损失
    # benchmark.predict_mae(kilometre, y, pre)
    # 修正数据后的准确率和损失
    vis_before_change = benchmark.modulation_true_value(kilometre, modulation_value, y, pre, draw=False)

    acc = (less_threshold_count_pre / (less_threshold_count_true + vis_before_change - less_threshold_count_pre))
    print("综合准确率: {:.3f}%".format(acc * 100))
