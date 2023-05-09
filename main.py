import pandas as pd
import numpy as np

from soft_gp import softGP, softGPInfo


def main():

    dataNeck = pd.read_csv("dataNeck.csv")
    dataNeck = dataNeck.sample(n=6000, random_state=2)
    y = dataNeck[['Inclination','Orientation']].to_numpy()
    x = dataNeck[['M1','M2','M3']].to_numpy()

    gpNeck = softGP("FK","DGP1", "Neck")

    prediction = gpNeck.predict(x)

    print("MAE%: ", np.mean(np.abs(prediction.mean - y), axis = 0)/(np.max(y)-np.min(y)))


    dataArm = pd.read_csv("dataArm.csv")
    dataArm = dataArm.sample(n=6000, random_state=2)
    x = dataArm[['Inclination','Orientation']].to_numpy()
    y = dataArm[['M1','M2','M3']].to_numpy()

    gpArm = softGP("IK","AGP", "Arm")

    prediction = gpArm.predict(x)

    print("MAE%: ", np.mean(np.abs(prediction.mean - y), axis = 0)/(np.max(y)-np.min(y)))

if __name__ == '__main__':
    main()