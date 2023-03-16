import numpy as np
import netCDF4 as nc
import usefull_functions as uf
import cftime
import os
import network_analysis as na
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from similarity_measures import eventsync_1D
from event_series import event_synchronization


DATA_DIRECTORY = '/Users/jurij/PycharmProjects/Meereisrinnen'
arctic_extent = (180, -180, 90, 60)


def lead_ds_from_date(date):
    date = date[:4] + '_' + date[4:]
    path = DATA_DIRECTORY + f'/data/DailyArcticLeadFraction_12p5km_Rheinlaender/data/LeadFraction_12p5km_{date}.nc'

    return nc.Dataset(path)


def lead_coordinates():
    path = DATA_DIRECTORY + '/data/DailyArcticLeadFraction_12p5km_Rheinlaender/LeadFraction_12p5km_LatLonGrid_subset.nc'

    return nc.Dataset(path)


def cyclone_occurrence_ds():
    path = DATA_DIRECTORY + '/data/CO_2_remapbil.nc'

    return nc.Dataset(path)


def eventmatrix_from_ts(arr, threshold):
    # get event like ts from continuous ts
    quan = np.quantile(arr, threshold, axis=0)
    arr[arr <= quan] = 0
    arr[arr > quan] = 1

    return arr.astype('int8')  # return as 8-bit int matrix


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def select_max(arr):
    rgl = (arr[:-1] < arr[1:])[1:]
    lgr = (arr[1:] < arr[:-1])[:-1]
    arr = arr[1:-1]

    arr[(rgl + lgr)] = np.nan
    arr[arr < .1] = np.nan

    return arr


def central_mass(arr_max, max_event_spacing, threshold):
    non_consecutive_max, non_consecutive_max_value = [], []

    max_inds = np.where(~np.isnan(arr_max))[0]
    if np.size(max_inds) < 2:  # too few events
        return np.zeros(shape=np.size(arr_max), dtype='int8')

    skip = 0
    for i, max_ind in enumerate(max_inds):
        j = 1
        consecutive_events = [max_ind]

        if skip > 0:  # skip consecutive events if the first one was already detected
            skip = skip - 1
        else:
            while True and i + j < np.size(max_inds):
                # find all consecutive events
                if max_inds[i + j] - max_inds[i + j - 1] <= max_event_spacing:
                    consecutive_events.append(max_inds[i + j])

                else:
                    if len(consecutive_events) == 1:
                        non_consecutive_max.append(consecutive_events[0])
                        non_consecutive_max_value.append(arr_max[consecutive_events[0]])
                    else:
                        # calculate "central mass"
                        cm, M = 0, 0
                        for event in consecutive_events:
                            cm += event * arr_max[event]
                            M += arr_max[event]
                        skip = len(consecutive_events) - 1
                        non_consecutive_max.append(round(cm / M))
                        non_consecutive_max_value.append(np.max(arr_max[consecutive_events]))
                    break
                j += 1

    # append last max if it wasn't consecutive
    if max_inds[-1] - max_inds[-2] > max_event_spacing:
        non_consecutive_max.append(max_inds[-1])
        non_consecutive_max_value.append(arr_max[max_inds[-1]])
    # calculate central mass if last max where consecutive
    else:
        # calculate "central mass"
        cm, M = 0, 0
        for event in consecutive_events:
            cm += event * arr_max[event]
            M += arr_max[event]
        non_consecutive_max.append(round(cm / M))
        non_consecutive_max_value.append(np.max(arr_max[consecutive_events]))

    eventlike = np.zeros(shape=np.size(arr_max))

    for ind, value in zip(non_consecutive_max, non_consecutive_max_value):
        eventlike[ind] = value

    quan = np.quantile(eventlike, threshold)
    #print(quan)
    # quan = np.repeat(quan, 5).reshape(eventlike.shape)
    event_arr = np.copy(eventlike)
    event_arr[eventlike <= quan] = 0
    event_arr[eventlike > quan] = 1

    return event_arr


def central_mass_detect_events(arr_max, max_event_spacing):
    non_consecutive_max = []

    max_inds = np.where(~np.isnan(arr_max))[0]
    if np.size(max_inds) < 2:  # too few events
        return np.zeros(shape=np.size(arr_max), dtype='int8')

    skip = 0
    for i, max_ind in enumerate(max_inds):
        j = 1
        consecutive_events = [max_ind]

        if skip > 0:  # skip consecutive events if the first one was already detected
            skip = skip - 1
        else:
            while True and i + j < np.size(max_inds):
                # find all consecutive events
                if max_inds[i + j] - max_inds[i + j - 1] <= max_event_spacing:
                    consecutive_events.append(max_inds[i + j])

                else:
                    if len(consecutive_events) == 1:
                        non_consecutive_max.append(consecutive_events[0])
                    else:
                        # calculate "central mass"
                        cm, M = 0, 0
                        for event in consecutive_events:
                            cm += event * arr_max[event]
                            M += arr_max[event]
                        skip = len(consecutive_events) - 1
                        non_consecutive_max.append(round(cm / M))
                    break
                j += 1

    # append last max if it wasn't consecutive
    if max_inds[-1] - max_inds[-2] > max_event_spacing:
        non_consecutive_max.append(max_inds[-1])
    # calculate central mass if last max where consecutive
    else:
        # calculate "central mass"
        cm, M = 0, 0
        for event in consecutive_events:
            cm += event * arr_max[event]
            M += arr_max[event]
        non_consecutive_max.append(round(cm / M))

    eventlike = np.zeros(shape=np.size(arr_max), dtype='int8')
    eventlike[non_consecutive_max] = 1

    return eventlike


class Data:
    # This will be our main container that holds our data, given a certain time range
    def __init__(self, date1, date2, n_smooth_X=5, n_smooth_Y=5, max_space_X=5, max_space_Y=5, threshold_X=.99, threshold_Y=.99):
        # get cyclone occurrence data set
        ds_cyc = cyclone_occurrence_ds()

        # get coordinates
        ds_coordinates = lead_coordinates()
        lon, lat = ds_coordinates['lon'][:], ds_coordinates['lat'][:]

        # get time range for data
        dates = uf.time_delta(date1, date2)
        dates_dt = uf.string_time_to_datetime(dates)

        # find indices of respective time range in cyclone data set
        date_ind = cftime.date2index(dates_dt, ds_cyc['time'])

        # assign lead data
        lead_data = np.array([lead_ds_from_date(date)['Lead Fraction'][:] for date in dates])
        lead_data[lead_data == 1] = np.nan

        # flatten coordinate axis
        lead_data = lead_data.reshape(lead_data.shape[0], -1)
        lon = lon.flatten()
        lat = lat.flatten()

        # assign cyclone occurrence data
        cyc_data = ds_cyc['cyclone_occurence'][date_ind]

        # remove time series that contains more than x% nan data, between 50% and 60% seems interesting
        nan_threshold = .55
        no_leads = np.sum(np.isnan(lead_data), axis=0) / lead_data.shape[0]
        lead_data_nonan = lead_data[:, no_leads <= nan_threshold]
        cyc_data_nonan = cyc_data[:, no_leads <= nan_threshold]
        self.lead_lon = lon.data[no_leads <= nan_threshold]
        self.lead_lat = lat.data[no_leads <= nan_threshold]

        # remove time series that have less than .1% avg lead fraction
        min_lead_avg = .01
        no_leads = np.nanmean(lead_data_nonan, axis=0)
        lead_data_nonan = lead_data_nonan[:, no_leads > min_lead_avg]
        cyc_data_nonan = cyc_data_nonan[:, no_leads > min_lead_avg]
        self.lead_lon = self.lead_lon[no_leads > min_lead_avg]
        self.lead_lat = self.lead_lat[no_leads > min_lead_avg]

        np.save('./data/lon_event_detection2', self.lead_lon)
        np.save('./data/lat_event_detection2', self.lead_lat)

        # interpolate NaN values
        ts_Y, ts_X = lead_data_nonan.T, cyc_data_nonan.T
        nans, y = np.isnan(ts_Y), lambda z: z.nonzero()[0]
        ts_Y[nans] = np.interp(y(nans), y(~nans), ts_Y[~nans])

        # detect events
        Xs, Ys, Xs_smooth, Ys_smooth = [], [], [], []

        print('smoothen')
        for i, (X, Y) in enumerate(zip(ts_X, ts_Y)):
            Xs_smooth.append(moving_average(X, n_smooth_X))
            Ys_smooth.append(moving_average(Y, n_smooth_Y))

        # X_threshold = np.quantile(Xs_smooth, .9)
        Xs_smooth, Ys_smooth = np.array(Xs_smooth), np.array(Ys_smooth)
        print('select maxima')
        for i, (X_smooth, Y_smooth) in enumerate(zip(Xs_smooth, Ys_smooth)):
            # select maxima
            # X_max = select_max(np.copy(X_smooth), X_threshold)
            # Y_max = select_max(np.copy(Y_smooth), np.quantile(Y_smooth, .90))
            X_max = select_max(np.copy(X_smooth))
            Y_max = select_max(np.copy(Y_smooth))

            # calculate central mass for consecutive maxima
            X_eventlike = central_mass(X_max, max_space_X, threshold_X)
            Y_eventlike = central_mass(Y_max, max_space_Y, threshold_Y)

            # append to list
            Xs.append(X_eventlike)
            Ys.append(Y_eventlike)

        np.save(f'./data/Xes_events_nsmooth={n_smooth_X}_maxspace={max_space_X}_threshold={threshold_X}', np.array(Xs, dtype='int8'))
        np.save(f'./data/Yes_events_nsmooth={n_smooth_Y}_maxspace={max_space_Y}_threshold={threshold_Y}', np.array(Ys, dtype='int8'))


def export_smooth(date1, date2, n_smooth_X=5, n_smooth_Y=5):
    # get cyclone occurrence data set
    ds_cyc = cyclone_occurrence_ds()

    # get coordinates
    ds_coordinates = lead_coordinates()
    lon, lat = ds_coordinates['lon'][:], ds_coordinates['lat'][:]

    # get time range for data
    dates = uf.time_delta(date1, date2)
    dates_dt = uf.string_time_to_datetime(dates)

    # find indices of respective time range in cyclone data set
    date_ind = cftime.date2index(dates_dt, ds_cyc['time'])

    # assign lead data
    lead_data = np.array([lead_ds_from_date(date)['Lead Fraction'][:] for date in dates])
    lead_data[lead_data == 1] = np.nan

    # flatten coordinate axis
    lead_data = lead_data.reshape(lead_data.shape[0], -1)
    lon = lon.flatten()
    lat = lat.flatten()

    # assign cyclone occurrence data
    cyc_data = ds_cyc['cyclone_occurence'][date_ind]

    # remove time series that contains more than x% nan data, between 50% and 60% seems interesting
    nan_threshold = .55
    no_leads = np.sum(np.isnan(lead_data), axis=0) / lead_data.shape[0]
    lead_data_nonan = lead_data[:, no_leads <= nan_threshold]
    cyc_data_nonan = cyc_data[:, no_leads <= nan_threshold]
    lead_lon = lon.data[no_leads <= nan_threshold]
    lead_lat = lat.data[no_leads <= nan_threshold]

    # remove time series that have less than .1% avg lead fraction
    min_lead_avg = .01
    no_leads = np.nanmean(lead_data_nonan, axis=0)
    lead_data_nonan = lead_data_nonan[:, no_leads > min_lead_avg]
    cyc_data_nonan = cyc_data_nonan[:, no_leads > min_lead_avg]
    lead_lon = lead_lon[no_leads > min_lead_avg]
    lead_lat = lead_lat[no_leads > min_lead_avg]

    # interpolate NaN values
    ts_Y, ts_X = lead_data_nonan.T, cyc_data_nonan.T
    nans, y = np.isnan(ts_Y), lambda z: z.nonzero()[0]
    ts_Y[nans] = np.interp(y(nans), y(~nans), ts_Y[~nans])

    # detect events
    Xs, Ys, Xs_smooth, Ys_smooth = [], [], [], []

    print('smoothen')
    for i, (X, Y) in enumerate(zip(ts_X, ts_Y)):
        Xs_smooth.append(moving_average(X, n_smooth_X))
        Ys_smooth.append(moving_average(Y, n_smooth_Y))

    Xs_smooth, Ys_smooth = np.array(Xs_smooth), np.array(Ys_smooth)
    np.save(f'./data/Xs_smooth_nsmooth={n_smooth_X}', Xs_smooth)
    np.save(f'./data/Ys_smooth_nsmooth={n_smooth_Y}', Ys_smooth)


def export_test_data(date1, date2, n_skip=10):
    # export not the complete data set, but only every n_skip entry
    print('process data')
    D = Data(date1, date2)
    lead_lon, lead_lat = D.lead_lon[::n_skip], D.lead_lat[::n_skip]
    cyc_lon, cyc_lat = D.cyc_lon[::n_skip], D.cyc_lat[::n_skip]
    X, Y = D.X[::n_skip], D.Y[::n_skip]

    date_str = f'./test_data_nskip={n_skip}/{date1}_{date2}_'

    print('save data')
    np.save(date_str + 'cyc_lon', cyc_lon)
    np.save(date_str + 'cyc_lat', cyc_lat)
    np.save(date_str + 'lead_lon', lead_lon)
    np.save(date_str + 'lead_lat', lead_lat)
    np.save(date_str + 'Xes', X)
    np.save(date_str + 'Yes', Y)

    print('done')


def export_processed_data(date1, date2):
    print('process data')
    D = Data(date1, date2)
    lead_lon, lead_lat = D.lead_lon, D.lead_lat
    cyc_lon, cyc_lat = D.cyc_lon, D.cyc_lat
    X, Y = D.X, D.Y

    date_str = f'./data/{date1}_{date2}_'

    print('save data')
    np.save(date_str + 'cyc_lon', cyc_lon)
    np.save(date_str + 'cyc_lat', cyc_lat)
    np.save(date_str + 'lead_lon', lead_lon)
    np.save(date_str + 'lead_lat', lead_lat)
    np.save(date_str + 'Xes', X)
    np.save(date_str + 'Yes', Y)

    print('done')


def test_ts():
    date1, date2 = '20021101', '20191231'
    # get cyclone occurrence data set
    ds_cyc = cyclone_occurrence_ds()

    # get coordinates
    ds_coordinates = lead_coordinates()
    lon, lat = ds_coordinates['lon'][:], ds_coordinates['lat'][:]

    # get time range for data
    dates = uf.time_delta(date1, date2)
    dates_dt = uf.string_time_to_datetime(dates)

    # find indices of respective time range in cyclone data set
    date_ind = cftime.date2index(dates_dt, ds_cyc['time'])

    # assign cyclone occurrence data
    cyc_data = ds_cyc['cyclone_occurence'][date_ind]

    # assign lead data
    lead_data = np.array([lead_ds_from_date(date)['Lead Fraction'][:] for date in dates])
    lead_data[lead_data == 1] = np.nan
    lead_data = lead_data.reshape(lead_data.shape[0], -1)

    # remove time series that contains more than x% nan data, between 50% and 60% seems interesting
    nan_threshold = .55
    no_leads = np.sum(np.isnan(lead_data), axis=0) / lead_data.shape[0]
    lead_data_nonan = lead_data[:, no_leads <= nan_threshold]
    cyc_data_nonan = cyc_data[:, no_leads <= nan_threshold]
    # lead_lon = lon.data[no_leads <= nan_threshold]
    # lead_lat = lat.data[no_leads <= nan_threshold]

    return lead_data_nonan.T, cyc_data_nonan.T


def plot_random_ts(n_smooth_X=5, n_smooth_Y=5, max_space_X=5, max_space_Y=5, threshold_X=.99, threshold_Y=.99):
    dir = f'./plots/sanity_plots/nsmooth={n_smooth_X}_maxspace={max_space_X}_threshold={threshold_X}/'
    os.makedirs(os.path.dirname(dir), exist_ok=True)

    try:
        print('try to load data')
        X_events = np.load(f'./data/Xes_events_nsmooth={n_smooth_X}_maxspace={max_space_X}_threshold={threshold_X}.npy')
        Y_events = np.load(f'./data/Yes_events_nsmooth={n_smooth_Y}_maxspace={max_space_Y}_threshold={threshold_Y}.npy')
        X_smooth, Y_smooth = np.load('./data/Xs_smooth.npy'), np.load('./data/Ys_smooth.npy')
        lon, lat = np.load('./data/lon_event_detection2.npy'), np.load('./data/lat_event_detection2.npy')
    except FileNotFoundError:
        print('failed\n', 'process data')
        Data('20021101', '20191231', n_smooth_X, n_smooth_Y, max_space_X, max_space_Y, threshold_X, threshold_Y)
        export_smooth('20021101', '20191231', n_smooth_X, n_smooth_Y)

        X_events = np.load(f'./data/Xes_events_nsmooth={n_smooth_X}_maxspace={max_space_X}_threshold={threshold_X}.npy')
        Y_events = np.load(f'./data/Yes_events_nsmooth={n_smooth_Y}_maxspace={max_space_Y}_threshold={threshold_Y}.npy')
        X_smooth, Y_smooth = np.load('./data/Xs_smooth.npy'), np.load('./data/Ys_smooth.npy')
        lon, lat = np.load('./data/lon_event_detection2.npy'), np.load('./data/lat_event_detection2.npy')

    time_range = 60
    np.random.seed(1)
    points = np.random.randint(0, X_events.shape[0], size=100)
    times = np.random.randint(0, X_events.shape[1] - time_range, size=100)
    T = np.arange(0, X_events.shape[1], 1)

    for i, (point, t0) in enumerate(zip(points, times)):
        count, sync_events = eventsync_1D(Y_events[point], X_events[point], identify_events=True, tau_max=10)
        print(sync_events)
        fig, ((ax1), (ax2)) = plt.subplots(2, 1, figsize=(20, 10))
        ax1.plot(T[t0 + n_smooth_X: t0 + time_range + n_smooth_X], X_smooth[point, t0: t0+time_range])
        ax1.bar(T[t0 + n_smooth_X: t0 + time_range + n_smooth_X], X_events[point, t0: t0+time_range], color='orange')
        ax1.set_title(f'cyc events ({round(lon[point], 3)}, {round(lat[point], 3)}), synchronized events: {count}')
        ax1.scatter(sync_events[:, 0] + n_smooth_X, np.repeat(.5, sync_events.shape[0]), c='green')
        ax1.set_xlim(t0 + n_smooth_X, t0 + time_range + n_smooth_X)
        ax1.set_ylim(0, 1)

        ax2.plot(T[t0 + n_smooth_X: t0 + time_range + n_smooth_X], Y_smooth[point, t0: t0+time_range])
        ax2.bar(T[t0 + n_smooth_X: t0 + time_range + n_smooth_X], Y_events[point, t0: t0+time_range], color='orange')
        ax2.set_title(f'lead events ({round(lon[point], 3)}, {round(lat[point], 3)})')
        ax2.scatter(sync_events[:, 1] + n_smooth_X, np.repeat(.5, sync_events.shape[0]), c='green')
        ax2.set_ylim(0, 1)
        ax2.set_xlim(t0 + n_smooth_X, t0 + time_range + n_smooth_X)

        plt.tight_layout()
        plt.savefig(dir + f'point_id={point}')
        plt.close(fig)


def eventlike_to_t(n_smooth_X=5, n_smooth_Y=5, max_space_X=5, max_space_Y=5, threshold_X=.99, threshold_Y=.99):
    try:
        print('try to load data')
        X_events = np.load(f'./data/Xes_events_nsmooth={n_smooth_X}_maxspace={max_space_X}_threshold={threshold_X}.npy')
        Y_events = np.load(f'./data/Yes_events_nsmooth={n_smooth_Y}_maxspace={max_space_Y}_threshold={threshold_Y}.npy')
    except FileNotFoundError:
        print('failed\n', 'process data')
        Data('20021101', '20191231', n_smooth_X, n_smooth_Y, max_space_X, max_space_Y, threshold_X, threshold_Y)

        X_events = np.load(f'./data/Xes_events_nsmooth={n_smooth_X}_maxspace={max_space_X}_threshold={threshold_X}.npy')
        Y_events = np.load(f'./data/Yes_events_nsmooth={n_smooth_Y}_maxspace={max_space_Y}_threshold={threshold_Y}.npy')



if __name__ == '__main__':
    # export_processed_data('20021101', '20191231')
    # export_test_data('20021101', '20191231')

    # Data('20021101', '20191231')

    # export_smooth('20021101', '20191231')

    plot_random_ts(n_smooth_X=5, max_space_X=3, n_smooth_Y=5, max_space_Y=3, threshold_X=.95, threshold_Y=.95)

    '''X, Y = np.load('./data/Xes_event_detection2.npy'), np.load('./data/Yes_event_detection2.npy')
    print(X.shape)
    lon, lat = np.load('./data/lon_event_detection2.npy'), np.load('./data/lat_event_detection2.npy')

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": ccrs.NorthPolarStereo(-45)}, figsize=(15, 10))
    ax1.coastlines(resolution='50m')
    ax1.set_extent(arctic_extent, crs=ccrs.PlateCarree())
    im1 = ax1.scatter(lon, lat, s=1, marker='s', c=np.sum(Y, axis=1), cmap='viridis',
                      transform=ccrs.PlateCarree())
    fig.colorbar(im1, orientation='horizontal')

    ax2.coastlines(resolution='50m')
    ax2.set_extent(arctic_extent, crs=ccrs.PlateCarree())
    im2 = ax2.scatter(lon, lat, s=.5, marker='s', c=np.sum(X, axis=1), cmap='viridis',
                      transform=ccrs.PlateCarree())
    fig.colorbar(im2, orientation='horizontal')'''











    pass
