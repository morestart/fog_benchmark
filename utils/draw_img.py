import matplotlib.pyplot as plt


def draw_img(true_data, pre_data):
    plt.plot(true_data)
    plt.plot(pre_data)
    plt.title('True And Predict')
    plt.xlabel('true')
    plt.ylabel('predict')
    plt.show()
