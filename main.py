import pandas as pd
import numpy as np

from soft_gp import softGP, softGPInfo


def main():

    dataNeck = pd.read_csv("dataNeck.csv")
    y = dataNeck[['Inclination','Orientation']].to_numpy()
    x = dataNeck[['M1','M2','M3']].to_numpy()


    gpNeck = softGP("FK","AGP", "Neck")

    prediction = gpNeck.predict(x)

    print("MAE%: ", np.mean(np.abs(prediction.mean - y), axis = 0)/(np.max(y)-np.min(y)))


    dataArm = pd.read_csv("dataArm.csv")
    x = dataArm[['Inclination','Orientation']].to_numpy()
    y = dataArm[['M1','M2','M3']].to_numpy()

    gpArm = softGP("IK","AGP", "Arm")

    prediction = gpArm.predict(x)

    print("MAE%: ", np.mean(np.abs(prediction.mean - y), axis = 0)/(np.max(y)-np.min(y)))

if __name__ == '__main__':
    main()