import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
# Loading the dataset
car_details = pd.read_csv("D:/Edge/CAR DETAILS.csv",encoding = 'latin1')
car_details.head()
# showing the sahpe
print(car_details.shape)
# Getting some information about the dataframe
car_details.info()
car_details.isnull().sum()
