import pandas as pd
import numpy as np

# ----- #

# # INTRODUCTION TO PANDAS DATAFRAME
# obj = pd.Series([1, 2, 3, 4, 5])
# print(obj)
# print(obj.values)
# print(obj.index)

# obj2 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
# print(obj2)
# print(obj2['a'])
# obj2['b'] = 26
# print(obj2)
# print(obj2[['b', 'a', 'c']])

# obj3 = pd.Series([5, 3, 7, 1], index=['a', 'b', 'c', 'd'])
# print(obj3[obj3 > 3])
# print(obj3 * 2)
# print(5 in obj3)
# print('a' in obj3)  # imagine the indexes as keys and objects as their respective values

# data1 = {'India': 'New Delhi', 'USA': 'Washington DC', 'Russia': 'Moscow', 'Italy': 'Rome', 'Norway': 'Oslo'}
# print(data1)
# obj4 = pd.Series(data1)
# print(obj4)
# rank = np.array(['USA', 'Italy', 'Russia', 'Norway', 'India'])
# obj4 = pd.Series(data1, index=rank)
# print(obj4)
# obj4.index.name = 'Country'
# print(obj4)
# obj4.name = 'Countries and their Capitals'
# print(obj4)