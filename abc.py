import numpy as np
import pandas as pd
import sklearn


def data_clean():
    df = pd.read_csv("loans_clean_schema.csv")
    df.dropna(inplace=True)
    df.replace("NA", np.NaN)
    df
