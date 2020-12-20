import pandas as pd
import csv
import numpy
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch


# Read and load the dataset
def loader(path, target=None):
    # read csv
    file = pd.read_csv(path)
    file = file.iloc[:, 1:]

    # record the columns that require modification
    columns = list(file.columns)
    filter_list = []
    for i, col in enumerate(columns):
        if file[col].dtypes == 'object':
            filter_list.append(i)

    # remove the double quotes in the values
    for i in filter_list:
        file.iloc[:, i] = file.iloc[:, i].str.replace(r"[\"\',]", '')

    # remove nan values and 'NaN' values
    for i, col in enumerate(columns):
        file[col] = file[col].fillna(value=int(file[col].median()))
        elements = set(file.iloc[:, i])
        if 'NaN' in elements:
            replace_num = int(file[col].median())
            file.iloc[:, i] = file.iloc[:, i].str.replace('NaN', str(replace_num))
            file[col] = file[col].fillna(value=int(file[col].median()))

    # encode the strings to integers
    le = LabelEncoder()
    file.iloc[:, -2] = le.fit_transform(file.iloc[:, -2])
    file.iloc[:, -1] = le.fit_transform(file.iloc[:, -1])

    # convert all the values to float
    for i, col in enumerate(columns):
        file[col] = file[col].astype(np.float)

    # move the target columns to the last
    columns = list(file.columns)
    columns.remove(target)
    columns.append(target)
    file = file[columns]

    training = file.sample(frac=0.7)
    testing = (file.merge(training, how='outer', indicator=True).loc[lambda x: x['_merge'] == 'left_only']).iloc[:, :-1]

    return file, training, testing


# define a customise torch dataset
class DataFrameDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.data_tensor = torch.Tensor(df.values)

    # a function to get items by index
    def __getitem__(self, index):
        obj = self.data_tensor[index]
        input = self.data_tensor[index][:-1]
        target = self.data_tensor[index][-1]

        return input, target

    # a function to count samples
    def __len__(self):
        n, _ = self.data_tensor.shape
        return n


# if __name__ == '__main__':
#     data = loader('3425_data.csv')
#     print(data)
