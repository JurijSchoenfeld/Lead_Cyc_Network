# space for testing and playing with the ES implementation
import numpy as np
import matplotlib.pyplot as plt
from event_series import EventSeries


def eventmatrix_from_ts(arr, threshold):
    # get event like ts from continuous ts
    quan = np.nanquantile(arr, threshold, axis=0)
    arr[arr < quan] = 0
    arr[arr >= quan] = 1

    return arr


def test_ts(n_points, n_timesteps, threshold, plot_probe=False):
    # return two matrices with shape n_timesteps x n_points

    X, Y = np.empty((n_timesteps, n_points)), np.empty((n_timesteps, n_points))
    X[0], Y[0] = np.random.normal(0, 1, n_points), np.random.normal(0, 1, n_points)
    phi_1, phi_2 = np.random.rand(n_points), np.random.rand(n_points)
    # phi = .3
    # phi_1, phi_2 = np.repeat(phi, n_points), np.repeat(phi, n_points)
    kappa_1, kappa_2 = np.random.rand(n_points), np.random.rand(n_points)

    for t in range(1, n_timesteps):
        for p in range(1, n_points):
            X[t, p] = phi_1[p] * X[t-1, p] + kappa_1[p] * Y[t-1, p] + np.random.normal(0, 1, 1)
            Y[t, p] = phi_2[p] * Y[t - 1, p] + kappa_2[p] * X[t - 1, p] + np.random.normal(0, 1, 1)

    X_eventseries = eventmatrix_from_ts(np.copy(X), threshold)
    Y_eventseries = eventmatrix_from_ts(np.copy(Y), threshold)
    ES = EventSeries(X_eventseries)

    if plot_probe:
        fig, axs = plt.subplots(4, 2)
        time = np.arange(0, n_timesteps, 1)

        for ax in axs:
            ax1, ax2 = ax
            point = np.random.randint(0, n_points, 1)[0]
            print(point)
            syncXY, syncYX = ES.event_synchronization(X_eventseries[:, point], Y_eventseries[:, point])

            ax1.plot(time, X_eventseries[:, point])
            ax1.set_title(rf'X for $\varphi_1=${phi_1[point]}, $\kappa_1=${kappa_1[point]}, syncXY={syncXY}')
            ax2.plot(time, Y_eventseries[:, point], c='orange')
            ax2.set_title(rf'X for $\varphi_2=${phi_2[point]}, $\kappa_2=${kappa_2[point]}, syncYX={syncYX}')
        plt.tight_layout()
        plt.show()

    return X, Y, X_eventseries, Y_eventseries


def adj_from_ts(X_data, Y_data):
    adj = EventSeries(X_data).event_synchronization(X_data[:, 50], Y_data[:, 50])
    print(adj)


if __name__ == '__main__':
    t = np.arange(0, 1000, 1)
    omega = 2*np.pi / 100
    X = np.random.normal(0, .5, 1000)
    Y = np.random.normal(0, .5, 1000)
    ts1 = X
    ts2 = Y

    fig, ((ax1), (ax2)) = plt.subplots(2, 1)

    quan1, quan2 = np.quantile(ts1, .90), np.quantile(ts2, .90)
    ts1[ts1 < quan1] = 0
    ts1[ts1 >= quan1] = 1

    ts2[ts2 < quan2] = 0
    ts2[ts2 >= quan2] = 1

    ax1.plot(t, ts1)
    ax2.plot(t, ts2)
    plt.show()

    dummy = np.random.randint(0, 2, size=(10, 2))
    ES = EventSeries(dummy).event_synchronization(ts1, ts2, taumax=10)
    print(ES)




