import matplotlib.pyplot as plt


def plot_from_lists_of_coordinates(x_list, y_list, colors, centers, x_lim=None, y_lim=None, save_plot=False, dpi=100):
    if len(x_list) != len(y_list):
        print("Lists must have the same length")

    if len(x_list) != len(colors):
        print("Colors do not match")

    if len(x_list) != len(centers):
        print("Centers do not match")

    plt.figure("Distribution title", dpi=dpi)

    for list_iter in range(len(x_list)):
        tmp_x = x_list[list_iter]
        tmp_y = y_list[list_iter]

        plt.plot(tmp_x, tmp_y, colors[list_iter], label="s" + str(list_iter + 1) + " = " + str(centers[list_iter]))

    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.legend()
    plt.grid()
    if save_plot:
        plt.savefig('/Users/marcoromanelli/Desktop/distribution.pdf')
    # plt.show()
