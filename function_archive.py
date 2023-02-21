# in this module old functions are archived that might be needed in the future
import numpy as np
import matplotlib.pyplot as plt
import usefull_functions as uf
import collect_data as cd


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


if __name__ == '__main__':
    pass
