import numpy as np
import matplotlib.pyplot as plt
import os
import re
import time
from event_series import event_synchronization
from sklearn.metrics import pairwise_distances
import cartopy.crs as ccrs
from geopy import distance
import collect_data as cd
import pickle


class Coordinates:
    def __init__(self, test=False):

        if test:
            self.clon = np.load('./test_data_nskip=10/20021101_20191231_cyc_lon.npy')
            self.clat = np.load('./test_data_nskip=10/20021101_20191231_cyc_lat.npy')
            self.llon = np.load('./test_data_nskip=10/20021101_20191231_lead_lon.npy')
            self.llat = np.load('./test_data_nskip=10/20021101_20191231_lead_lat.npy')
        else:
            self.clon = np.load('./data/20021101_20191231_cyc_lon.npy')[:60300]
            self.clat = np.load('./data/20021101_20191231_cyc_lat.npy')[:60300]
            self.llon = np.load('./data/20021101_20191231_lead_lon.npy')
            self.llat = np.load('./data/20021101_20191231_lead_lat.npy')


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


def significance_test_surrogates(n_events_x, n_events_y):
    N_surrogates = 1000  # number of surrogates
    NT = 2632  # length of time series

    n_events_x, n_events_y = int(n_events_x[0]), int(n_events_y[0])
    print(n_events_x, n_events_y)

    # generate surrogate time series
    x_surrogates, y_surrogates = np.zeros(shape=(2, N_surrogates, NT))
    x_surrogates[:, : n_events_x] = 1
    y_surrogates[:, : n_events_y] = 1
    x_surrogates = shuffle_along_axis(x_surrogates, 1)
    y_surrogates = shuffle_along_axis(y_surrogates, 1)

    # calculate ES of surrogates
    ES = np.empty(N_surrogates)
    for i, (x, y) in enumerate(zip(x_surrogates, y_surrogates)):
        ES[i] = event_synchronization(x, y)

    return np.quantile(ES, .995)


def get_bins(n_bins=10):
    X, Y = np.load('./data/20021101_20191231_Xes.npy'), np.load('./data/20021101_20191231_Yes.npy')

    countsX, binsX = np.histogram(np.sum(X, axis=1), bins=n_bins)
    countsY, binsY = np.histogram(np.sum(Y, axis=1), bins=n_bins)

    mean_binX = np.rint(binsX[:-1] + .5 * (binsX[1:] - binsX[:-1]))
    mean_binY = np.rint(binsY[:-1] + .5 * (binsY[1:] - binsY[:-1]))
    print(n_bins)

    np.save('./significant/binsX', binsX)
    np.save('./significant/binsY', binsY)
    np.save('./significant/mean_binsX', mean_binX)
    np.save('./significant/mean_binsY', mean_binY)

    return binsX, binsY, mean_binX, mean_binY


def get_threshold_from_bins(mean_binX, mean_binY):
    print(mean_binX.shape)
    ES = pairwise_distances(mean_binX.reshape(-1, 1), mean_binY.reshape(-1, 1), metric=significance_test_surrogates,
                            n_jobs=-1)
    np.save('./significant/ES_threshold_995_1000.npy', ES)


def visualize_sig_threshold(n_bins):
    binsX, binsY, mean_binsX, mean_binsY = get_bins(n_bins)
    xx, yy = np.meshgrid(mean_binsX, mean_binsY)
    zz = np.load('./significant/ES_threshold_995_1000.npy')

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('significance threshold')
    ax.plot_surface(xx, yy, zz, cmap='viridis')
    ax.view_init(30, 40)
    plt.tight_layout()
    plt.savefig('./nevent_sig_dependance')


def build_adj(sig_threshold=.2):
    threshold_str = str(sig_threshold).replace('.', '_')
    print(threshold_str)
    path = './partial_adj'
    file_names = sorted(os.listdir(path))[1:]

    ES = np.load(path + '/' + file_names[0])
    Adj = (ES >= sig_threshold)
    for i, file in enumerate(file_names[1:]):
        print(i)
        ES = np.load(path + '/' + file)
        Adj = np.vstack((Adj, ES >= sig_threshold))

    np.save(f'adjacency_matrix_{threshold_str}', Adj)
    print(Adj.shape)


def plot_degrees(sig):
    C = Coordinates()
    Adj = np.load(f'adjacency_matrix_{sig}.npy')
    n_links = np.sum(Adj, axis=1)

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.NorthPolarStereo(-45)}, figsize=(15, 10))
    ax.coastlines(resolution='50m')
    # ax.set_extent(cd.arctic_extent, crs=ccrs.PlateCarree())

    im = ax.scatter(C.clon, C.clat, c=n_links, transform=ccrs.PlateCarree(), s=2)
    fig.colorbar(im)
    ax.set_title(
        'max: ' + np.max(n_links) + 'number of edges: ' + np.sum(Adj) + 'edge density: ' + np.sum(Adj) / np.size(Adj))
    plt.tight_layout()
    plt.savefig(f'./plots/plotdegrees_{sig}')


def avg_edge_length(sig='0_15'):
    Adj = np.load(f'adjacency_matrix_{sig}.npy')
    C = Coordinates()

    # degrees = np.sum(Adj, axis=1)

    # c_ind = np.where(degrees > 200)[0][70]
    print()
    c_ind = 4000
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.NorthPolarStereo(-45)}, figsize=(15, 10))
    ax.coastlines(resolution='50m')
    ax.scatter(C.clon[c_ind], C.clat[c_ind], c='red', s=30, transform=ccrs.PlateCarree())
    ax.set_extent(cd.arctic_extent, crs=ccrs.PlateCarree())
    for l_ind in np.where(Adj[c_ind] == 1)[0]:
        ax.scatter(C.llon[l_ind], C.llat[l_ind], c='black', s=2, transform=ccrs.PlateCarree())

    plt.show()




if __name__ == '__main__':
    '''for sig in ['0_10', '0_15', '0_20']:
        plot_degrees(sig)'''
    _, _, X, Y = get_bins(20)
    get_threshold_from_bins(X, Y)
    visualize_sig_threshold(20)
    # generate_surrogates(123, 100)
    # plot_number_links()
    # avg_edge_length()


    pass
