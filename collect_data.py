import numpy as np
import netCDF4 as nc
import usefull_functions as uf
import cftime

DATA_DIRECTORY = '/Users/jurij/PycharmProjects/Meereisrinnen'


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


class Data:
    # This will be our main container that holds our data, given a certain time range
    def __init__(self, date1, date2):
        # get cyclone occurrence data set
        ds_cyc = cyclone_occurrence_ds()

        # get coordinates
        ds_coordinates = lead_coordinates()
        self.lon, self.lat = ds_coordinates['lon'], ds_coordinates['lat']

        # get time range for data
        self.dates = uf.time_delta(date1, date2)
        self.dates_dt = uf.string_time_to_datetime(self.dates)

        # find indices of respective time range in cyclone data set
        date_ind = cftime.date2index(self.dates_dt, ds_cyc['time'])

        # assign lead data
        self.lead_data = np.array([lead_ds_from_date(date)['Lead Fraction'][:] for date in self.dates])

        # assign cyclone occurrence data
        self.cyc_data = ds_cyc['cyclone_occurence'][date_ind]
        # due to remapping the data looses its shape and becomes flat
        self.cyc_data = self.cyc_data.reshape(-1, self.lead_data[0].shape[0], self.lead_data[0].shape[1])
        print(self.cyc_data.shape)


if __name__ == '__main__':
    pass

