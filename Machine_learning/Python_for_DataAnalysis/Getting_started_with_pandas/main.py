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

# ----- #

# # DATAFRAME
# data = {
#     'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
#     'year': [2000, 2001, 2002, 2001, 2002],
#     'pop': [1.5, 1.7, 3.6, 2.4, 2.9]
# }
# frame1 = pd.DataFrame(data)
# print(frame1)
# frame1 = pd.DataFrame(data, columns=['year', 'state', 'pop'])
# print(frame1)

# frame2 = pd.DataFrame(
#     data, columns=['year', 'state', 'pop', 'debt'],
#     index=['one', 'two', 'three', 'four', 'five']
# )
# print(frame2)
# print(frame2.columns)
# print(frame2['state'])
# print(frame2.state)
# print(frame2.loc['one'])
# frame2.debt = np.arange(5)
# print(frame2)
# frame2['eastern'] = frame2.state == 'Nevada'
# print(frame2)
# del frame2['eastern']
# print(frame2)
# frame2 = frame2.T  # transposed dataframe
# print(frame2)

# ----- #

# # INDEX OBJECTS
# obj = pd.Series(range(3), index=['a', 'b', 'c'])
# print(obj)
#
# obj.index[0] = 'd'  # index object are immutable

# ----- #

# ESSENTIAL FUNCTIONALITY
