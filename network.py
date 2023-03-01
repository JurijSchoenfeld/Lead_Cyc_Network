import collect_data as cd
import numpy as np
from event_series import event_synchronization
from sklearn.metrics import pairwise_distances
import datetime
from multiprocessing import Pool
import time


class Network:
    def __init__(self, date1=None, date2=None):
        # get data for custom time range
        self.date1, self.date2 = date1, date2
        date_str = f'./data/{date1}_{date2}_'
        print('Load data')
        if date1 and date2:
            try:
                print('Try to import pre-processed data')
                self.X, self.Y = np.load(date_str + 'Xes.npy'), np.load(date_str + 'Yes.npy')
            except FileNotFoundError:
                print('Import failed\nTry to pre-process data')
                cd.export_processed_data(date1, date2)

        # if no date range is passed collect all available data
        else:
            print('Try to import pre-processed data')
            self.date1, self.date2 = '20021101', '20191231'
            date_str = f'./data/20021101_20191231_'
            self.X, self.Y = np.load(date_str + 'Xes.npy'), np.load(date_str + 'Yes.npy')
            print('Done')

        N = 10
        # self.X = self.X[20_000: 20_000 + N, :]
        # self.Y = self.Y[20_000: 20_000 + N, :]
        self.A = None
        self.slices = self.get_slices()

    def get_slices(self):
        N = self.Y.shape[0]
        slice_width = 10
        slices = []
        for start in np.arange(0, round(N, -3), slice_width):  # round to closest thousend
            if start + slice_width > N:
                slices.append(slice(round(N, -3), N))
            else:
                slices.append(slice(start, start + slice_width))
        return slices

    def pdist_wrapper(self, identifier):
        print(identifier)
        sli = self.slices[identifier]
        start_time = datetime.datetime.now()
        A = pairwise_distances(self.X[sli], self.Y, metric=event_synchronization)
        print('Finished: ', sli, 'in: ', datetime.datetime.now() - start_time)
        np.save(f'./partial_adj/{self.date1}_{self.date2}_adj_{sli.start}_{sli.stop}.npy', A)

        return 0

    def compute_adjacency(self):
        print('Start calculating adjacency matrix')
        start_time = datetime.datetime.now()

        with Pool() as pool:
            # issue tasks to the thread pool
            res = pool.imap_unordered(self.pdist_wrapper, range(len(self.slices)))
            # shutdown the thread pool
            pool.close()
            # wait for all issued task to complete
            pool.join()

        # self.A = pairwise_distances(self.X, self.Y, metric=event_synchronization)  # for n_jobs!=1 hangs up forever
        print('Execution time: ', datetime.datetime.now() - start_time)
        print('Saving results')
        np.save(f'adj_{self.date1}_{self.date2}.npy', self.A)
        print('done')


def get_slices(slice_width=100):
    date_str = f'./data/20021101_20191231_'
    Y = np.load(date_str + 'Yes.npy')
    N = Y.shape[0]
    slices = []
    for start in np.arange(0, round(N, -3), slice_width):  # round to closest thousend
        if start + slice_width > N:
            slices.append(slice(round(N, -3), N))
        else:
            slices.append(slice(start, start + slice_width))
    return slices

# N = Network()
# X, Y = N.X, N.Y
slices = get_slices()


def pdist_wrapper(identifier):
    date_str = f'./data/20021101_20191231_'

    sli = slices[identifier+22+104]
    print('Start working on: ', sli)
    X, Y = np.load(date_str + 'Xes.npy')[sli], np.load(date_str + 'Yes.npy')

    start_time = datetime.datetime.now()
    A = pairwise_distances(X, Y, metric=event_synchronization)
    print('Finished: ', sli, 'in: ', datetime.datetime.now() - start_time)
    np.save(f'./partial_adj/adj_id{21+104+identifier}_{sli.start}_{sli.stop}.npy', A)
    return None


if __name__ == '__main__':
    with Pool() as pool:
        # issue tasks to the thread pool
        res = pool.imap_unordered(pdist_wrapper, range(len(slices)))

        for _ in res:
            pass

    # Xtest, Ytest = np.random.randint(0, 100, size=(2, 40_000, 3), dtype='int8')

    # res = pairwise_distances(Xtest, Ytest, metric='euclidean', n_jobs=-1)
    # print(res.shape)


    pass


