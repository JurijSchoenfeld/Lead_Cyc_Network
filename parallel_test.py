# example of parallel imap_unordered() with the thread pool and a task that does not return a value
from random import random
from time import sleep
from multiprocessing import Pool
import numpy as np


def task(identifier):
    print(identifier)
    res = np.sqrt(np.random.randint(0, 1000, size=(10000, 10000)))

    return None


if __name__ == '__main__':
    # create and configure the thread pool
    with Pool() as pool:
        # issue tasks to the thread pool
        results = pool.imap_unordered(task, range(100))

        for r in results:
            pass
