import numpy as np
import netCDF4 as nc
import usefull_functions as uf
import cftime
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


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


class Data:
    # This will be our main container that holds our data, given a certain time range
    def __init__(self, date1, date2):
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

        # handle NaN values
        lead_data_nonan[np.isnan(lead_data_nonan)] = 0
        cyc_data_nonan[np.isnan(lead_data_nonan)] = 0

        # remove land mask from cyclone occurence data
        # land mask for lead grid is contained in an old lead data set, that has a different lon lat box
        # may be added later

        # convert to event series
        lead_eventlike = eventmatrix_from_ts(lead_data_nonan, .95)
        cyc_eventlike = np.copy(cyc_data_nonan)
        cyc_eventlike[cyc_eventlike <= .75] = 0
        cyc_eventlike[cyc_eventlike > .75] = 1
        cyc_eventlike = cyc_eventlike.astype('int8')

        # remove locations with less than 10 cyclone events
        cyc_events_ppixel = np.sum(cyc_eventlike, axis=0)
        cyc_eventlike = cyc_eventlike[:, cyc_events_ppixel >= 10]
        self.cyc_lon = self.lead_lon[cyc_events_ppixel >= 10]
        self.cyc_lat = self.lead_lat[cyc_events_ppixel >= 10]

        # remove locations with less than 100 lead events
        lead_events_ppixel = np.sum(lead_eventlike, axis=0)
        lead_eventlike = lead_eventlike[:, lead_events_ppixel >= 100]
        self.lead_lon = self.lead_lon[lead_events_ppixel >= 100]
        self.lead_lat = self.lead_lat[lead_events_ppixel >= 100]

        # save transpose, this will parse to es-method
        # first axis locations, second axis time
        self.X = cyc_eventlike.T
        self.Y = lead_eventlike.T

        # another option to speed up calculation is to parse a precomputed version of the events
        # now this has to be done in every iteration step
        # may be added later


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


if __name__ == '__main__':
    export_processed_data('20021101', '20191231')



    pass

