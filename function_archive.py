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

if __name__ == '__main__':
    pass
