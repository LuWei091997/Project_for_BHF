import pandas as pd


def save_data(path, data):
    test = pd.DataFrame(data=data)
    test.to_csv(path, header=False, index=False)
