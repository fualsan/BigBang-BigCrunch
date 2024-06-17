import matplotlib.pyplot as plt
from matplotlib import animation


def animate_solutions(data_arr):
    fig, ax = plt.subplots()

    _scatter = ax.scatter(data_arr[0, :, 0], data_arr[0, :, 1], c='blue', s=15)

    # Set the limits of the plot
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    def update_points(n):
        _data = data_arr[n, :, :]
        _scatter.set_offsets(_data)
        ax.set_title(f'Candidate Solutions (Iteration: {n})')
        return _scatter,

    animate_fig = animation.FuncAnimation(fig=fig, func=update_points, frames=100, interval=100, blit=False, repeat=False)

    plt.show()

    return animate_fig