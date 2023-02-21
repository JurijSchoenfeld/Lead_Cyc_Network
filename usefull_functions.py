from datetime import datetime, date, timedelta
from math import log10, floor


def time_delta(date1, date2):
    # Returns a list that contains all dates between date1 and date2 in a '20200101.nc' way.
    days = []
    start_date = date(int(date1[:4]), int(date1[4:6]), int(date1[6:]))
    end_date = date(int(date2[:4]), int(date2[4:6]), int(date2[6:]))

    delta = end_date - start_date   # returns timedelta

    for i in range(delta.days + 1):
        day = start_date + timedelta(days=i)
        if day.month in [11, 12, 1, 2, 3]:
            day = str(day)
            days.append(''.join(c for c in day if c not in '-'))

    return days


def datetime_to_string(dates):
    # return datatime object for a given date in string format
    if not isinstance(dates, list):
        return ''.join(c for c in str(dates) if c not in '-')
    else:
        for d, date in enumerate(dates):
            dates[d] = ''.join(c for c in str(date) if c not in '-')
        return [dates]


def string_time_to_datetime(dates):
    # return string version of a given datetime object
    if not isinstance(dates, list):
        return datetime(int(dates[:4]), int(dates[4:6]), int(dates[6:]), 9, 0, 0)
    else:
        return [datetime(int(d[:4]), int(d[4:6]), int(d[6:]), 9, 0, 0) for d in dates]

def round_sig(x, sig=2):
    return round(x, sig-int(floor(log10(abs(x))))-1)
