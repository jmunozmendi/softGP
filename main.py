import pandas as pd
import numpy as np

from soft_gp import softGP, softGPInfo


def main():


    gpFK = softGP("FK","DGP1")

    test_data = np.array([-122.908041422489,-81.1222294235976,204.350554253564], ndmin = 2)

    prediction = gpFK.predict(test_data)

    print(prediction.mean)

    gpIK = softGP("IK","DGP1")

    test_data = np.array([-21.8534582838264,29.8675450150361], ndmin = 2)

    prediction = gpIK.predict(test_data)

    print(prediction.mean)

if __name__ == '__main__':
    main()