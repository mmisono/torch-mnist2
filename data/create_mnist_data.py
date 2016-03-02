import os

import h5py
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split

def main():
    mnist = fetch_mldata("MNIST original")
    X_, y_ = mnist.data / 255., mnist.target
    X, y = dict(), dict()
    X['train'], X__, y['train'], y__ = train_test_split(X_,y_,
            train_size=50000,random_state=42)
    X['val'], X['test'], y['val'], y['test'] = train_test_split(X__,y__,
            train_size=10000,random_state=42)
    #X['train'], X['val'], X['test'] = X_[:50000], X_[50000:60000], X_[60000:]
    #y['train'], y['val'], y['test'] = y_[:50000], y_[50000:60000], y_[60000:]

    h5file = h5py.File("mnist.hdf5", "w")
    
    for split in ['train', 'val', 'test']:
        h5file.create_group('{}'.format(split))
        h5file.create_dataset('{}/input'.format(split), data = X[split])
        h5file.create_dataset('{}/target'.format(split), data = y[split])

        #h5file.create_group('{}/input'.format(split))
        #h5file.create_group('{}/target'.format(split))
        #for i,d in enumerate(zip(X[split],y[split])):
        #    h5file.create_dataset('{}/input/{}'.format(split,i), data = d[0])
        #    h5file.create_dataset('{}/target/{}'.format(split,i), data = d[1])

    h5file.flush()
    h5file.close()

if __name__ == "__main__":
    main()
