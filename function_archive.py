# in this module old functions are archived that might be needed in the future
import numpy as np
import matplotlib.pyplot as plt
import usefull_functions as uf
import collect_data as cd
import cartopy.crs as ccrs


def plot_example_ts():
    Data1 = cd.Data('20021101', '20051231')
    time, cyc, lead, lon, lat = Data1.dates_dt, Data1.cyc_data, Data1.lead_data, Data1.lon, Data1.lat

    fig, axs = plt.subplots(4, 2, sharex=True, figsize=(15, 10))
    print(axs.shape)

    for ax1, ax2 in zip(axs[:, 0], axs[:, 1]):
        n, m = np.random.randint(200, 300, size=2)
        ax1.bar(time, cyc[:, n, m])
        ax2.bar(time, lead[:, n, m])
        ax1.set_title(rf'$i({uf.round_sig(float(lon[n, m]), 5)}, {uf.round_sig(float(lat[n, m]), 4)})$')
        ax2.set_title(rf'$j({uf.round_sig(float(lon[n, m]), 5)}, {uf.round_sig(float(lat[n, m]), 4)})$')
    ax1.tick_params(axis='x', labelrotation=45)
    ax2.tick_params(axis='x', labelrotation=45)
    plt.tight_layout
    plt.savefig('example_timeseries_1')


def plot_example_lead_cyc_data():
    D = cd.Data('20181101', '20181130')

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.NorthPolarStereo(-45)}, figsize=(15, 10))
    ax.coastlines(resolution='50m')
    ax.set_extent(cd.arctic_extent, crs=ccrs.PlateCarree())
    print(D.lat.shape, D.lon.shape, D.lead_data[29].shape)

    # replace lead_data with cyc data if you want to display cycs
    ax.pcolormesh(D.lon.reshape(440, 480), D.lat.reshape(440, 480), D.lead_data[0].reshape(440, 480),
                  transform=ccrs.PlateCarree())
    plt.show()


def plot_nan_percentage():
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.NorthPolarStereo(-45)}, figsize=(15, 10))
    ax.coastlines(resolution='50m')
    ax.set_extent(cd.arctic_extent, crs=ccrs.PlateCarree())
    ax.set_title('Lead data percentage of nan 20021101-20191231')

    print(np.nanmax(D.lead_data_nonan[0]))
    im = ax.scatter(D.lead_lon, D.lead_lat, s=1, marker='s', c=np.nanmean(D.lead_data_nonan, axis=0), cmap='viridis',
                    transform=ccrs.PlateCarree())
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig('nan_removed-lead-data')

def rename_files():
    file_names = os.listdir('./partial_adj')
    start = 10

    for file in sorted(file_names)[1:]:
        id, slice_start, slice_end = re.findall('[0-9]+', file)
        slice_start, slice_end = int(slice_start), int(slice_end)
        # print(str(id).zfill(3), slice_start, slice_end, file)
        os.rename('./partial_adj/' + file,
                  './partial_adj/' + f'adj_id{str(int(slice_start / 100)).zfill(3)}_{slice_start}_{slice_end}.npy')


def plot_nevents_skip():
    C = na.Coordinates(True)

    X = np.load('./test_data_nskip=10/20021101_20191231_Xes.npy')
    Y = np.load('./test_data_nskip=10/20021101_20191231_Yes.npy')

    fig, ((ax1), (ax2)) = plt.subplots(1, 2, subplot_kw={"projection": ccrs.NorthPolarStereo(-45)}, figsize=(15, 10))
    ax1.coastlines(resolution='50m')
    ax2.coastlines(resolution='50m')

    ax1.set_title('number of cyclone events')
    im1 = ax1.scatter(C.clon, C.clat, c=np.sum(X, axis=1), s=5, cmap='viridis', transform=ccrs.PlateCarree())
    fig.colorbar(im1, ax=ax1, orientation='horizontal')

    ax2.set_title('number of lead events')
    im2 = ax2.scatter(C.llon, C.llat, c=np.sum(Y, axis=1), s=5, cmap='viridis', transform=ccrs.PlateCarree())
    fig.colorbar(im2, ax=ax2, orientation='horizontal')

    plt.savefig('plots/nevents_nskip10.png')


    def es_test():
        X = np.load('./test_data_nskip=10/20021101_20191231_Xes.npy')
        Y = np.load('./test_data_nskip=10/20021101_20191231_Yes.npy')
        T = np.arange(0, X.shape[1], 1)

        ncols = 2
        arr = np.random.randint(0, 10, size=(30, ncols))
        randX, randY = np.random.randint(0, 6000, size=ncols), np.random.randint(0, 6000, size=ncols)

        fig, axs = plt.subplots(2, ncols, figsize=(15, 10))

        for i, (ax, rX, rY) in enumerate(zip(axs.flatten(), randX, randX)):
            Xs, Ys = X[rX], Y[rY]
            norm = np.sqrt((np.sum(Xs) - 2) * (np.sum(Ys) - 2))
            es1, hits = eventsync_1D(Ys, Xs, identify_events=True, tau_max=30)
            print(es1, '\n', hits)
            es2 = event_synchronization(Xs, Ys)
            ax.bar(T[0:500], Xs[0:500])
            ax.scatter(T[hits[:, 0]], np.repeat(.5, es1), c='red')
            ax.set_title(f'cyc, sync1,2={es1 / norm, es2}')
            axs[1, i].bar(T[0:500], Ys[0:500])
            axs[1, i].set_title('lead')
            axs[1, i].scatter(T[hits[:, 1]], np.repeat(.5, es1), c='red')

            axs[1, i].set_xlim(0, 500)
            ax.set_xlim(0, 500)

            if i == ncols - 1:
                break

        plt.tight_layout()
        plt.savefig('./plots/es_check.png')


def ts_test():
    print('load data')
    ts_Y, ts_X = test_ts()
    print('interpolate nan')
    nans, y = np.isnan(ts_Y), lambda z: z.nonzero()[0]
    ts_Y[nans] = np.interp(y(nans), y(~nans), ts_Y[~nans])

    print('calculate events')
    np.random.seed(42)
    for point in np.random.randint(0, 60000, size=10):
        print(point)
        T = np.arange(0, ts_Y.shape[1], 1)

        n_smooth_Y, n_smooth_X = 5, 5
        ts_Y_smooth = moving_average(ts_Y[point], n_smooth_Y)
        ts_X_smooth = moving_average(ts_X[point], n_smooth_X)

        fig, axs = plt.subplots(2, 3, figsize=(20, 10))
        axs = axs.flatten()

        axs[0].plot(T, ts_X[point])
        axs[0].set_title('cyc ts')

        axs[1].plot(T[n_smooth_X - 1:], ts_X_smooth)
        axs[1].set_title('cyc ts smooth')

        t_range = 200
        ts_X_max = select_max(np.copy(ts_X_smooth), np.quantile(ts_X_smooth, .95))
        axs[2].scatter(T[n_smooth_X: - 1][:t_range], ts_X_max[:t_range])
        axs[2].bar(T[n_smooth_X: - 1][:t_range], central_mass(ts_X_max, 5)[:t_range])
        axs[2].plot(T[n_smooth_X - 1:][:t_range], ts_X_smooth[:t_range])
        axs[2].set_title('cyc ts maxima')

        ts_Y_max = select_max(np.copy(ts_Y_smooth), np.quantile(ts_Y_smooth, .95))

        axs[5].scatter(T[n_smooth_Y: - 1][:t_range], ts_Y_max[:t_range])
        axs[5].plot(T[n_smooth_Y - 1:][:t_range], ts_Y_smooth[:t_range])
        axs[5].bar(T[n_smooth_Y: - 1][:t_range], central_mass(ts_Y_max, 5)[:t_range])
        axs[5].set_title('lead ts maxima')

        axs[4].plot(T[n_smooth_Y - 1:], ts_Y_smooth)
        axs[4].set_title('lead ts smooth')

        axs[3].plot(T, ts_Y[point])
        axs[3].set_title('lead ts')

        # central_mass(np.copy(ts_Y_max), np.copy(ts_Y_smooth), 5)

        plt.savefig(f'./plots/test_ts_id{point}.png')


def plot_nevents():
    Data('20021101', '20191231')

    X, Y = np.load('./data/Xes_event_detection2.npy'), np.load('./data/Yes_event_detection2.npy')
    print(X.shape)
    print(np.sum(X, axis=1))
    print(np.sum(Y, axis=1))
    lon, lat = np.load('./data/lon_event_detection2.npy'), np.load('./data/lat_event_detection2.npy')

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": ccrs.NorthPolarStereo(-45)}, figsize=(15, 10))
    ax1.coastlines(resolution='50m')
    ax1.set_extent(arctic_extent, crs=ccrs.PlateCarree())
    im1 = ax1.scatter(lon, lat, s=1, marker='s', c=np.sum(Y, axis=1), cmap='viridis',
                      transform=ccrs.PlateCarree())
    fig.colorbar(im1, orientation='horizontal')

    ax2.coastlines(resolution='50m')
    ax2.set_extent(arctic_extent, crs=ccrs.PlateCarree())
    im2 = ax2.scatter(lon, lat, s=1, marker='s', c=np.sum(X, axis=1), cmap='viridis',
                      transform=ccrs.PlateCarree())
    fig.colorbar(im2, orientation='horizontal')

    '''fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.hist(np.sum(X, axis=1), bins=20)
    ax2.hist(np.sum(Y, axis=1), bins=20)'''

    plt.savefig('nevents_detection2.png')

if __name__ == '__main__':
    arr = np.random.randint(0, 100, size=(20, 5))
    print(arr)

    quan = np.quantile(arr, .5, axis=1)
    quan = np.repeat(quan, 5).reshape(arr.shape)
    event_arr = np.copy(arr)
    event_arr[arr < quan] = 0
    event_arr[arr >= quan] = 1
    print(event_arr)
    pass
