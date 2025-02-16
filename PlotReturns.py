import numpy as np
import matplotlib.pyplot as plt


def plot_returns(filename):
    data = np.load(filename, allow_pickle=True)
    all_returns = data['arr_0']

    plt.figure(figsize=(10, 6))
    for idx, instance in enumerate(all_returns):
        plt.plot(instance)

    plt.xlabel("Iterations")
    plt.ylabel("Return")
    plt.title("Return Over Training Iterations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


plot_returns("data_4_layers/returns_bias_layer_1_0.005.npz")
