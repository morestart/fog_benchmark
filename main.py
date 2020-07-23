import os

from benchmark.benchmark import Benchmark

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if __name__ == '__main__':
    kilometre = 5000
    ratio = .3
    modulation_value = 2000

    benchmark = Benchmark(ratio)
    _, y, pre = benchmark.predict_()

    # t = threading.Thread(target=benchmark.predict_acc, args=(kilometre, y, pre, True))
    # t1 = threading.Thread(target=benchmark.modulation_true_value, args=(kilometre, modulation_value, y, pre, True))
    # t.start()
    # t1.start()
    # 预报准确率
    benchmark.predict_acc(kilometre, y, pre, draw=False)
    # 预报mae损失
    benchmark.predict_mae(kilometre, y, pre)
    # 修正数据后的准确率和损失
    benchmark.modulation_true_value(kilometre, modulation_value, y, pre, draw=False)
