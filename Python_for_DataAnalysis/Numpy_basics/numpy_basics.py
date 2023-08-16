import numpy as np
import matplotlib.pyplot as plt

# ----- #

# # CREATING NDARRAYS
# data1 = [6.1, 7, 9, 1, 3]
# arr1 = np.array(data1)
# print(arr1)
#
# data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
# arr2 = np.array(data2)
# print(arr2)
#
# print(arr1.shape, arr2.shape)
# print(arr1.dtype)
#
# print(np.zeros((10, 10)))
# print(np.ones(7))
# print(np.empty((2, 3, 2)))
#
# print(np.arange(10))

# ----- #

# # DATA TYPES FOR NDARRAYS
# arr1 = np.array([32, 34, 5], dtype=np.float64)
# print(arr1)
# arr2 = np.array([1, 2, 3], dtype=np.int32)
# print(arr2)
#
# arr3 = np.array([1, 2, 3])
# arr3 = arr3.astype(np.float64)
# print(arr3)
# print(arr3.dtype)
#
# int_arr = np.arange(10)
# calibers = np.array([.22, .33, .8, .5], dtype=np.float64)
# int_arr = int_arr.astype(calibers.dtype)
# print(int_arr)

# ----- #

# # OPERATIONS BETWEEN ARRAYS AND SCALARS
# arr1 = np.array([[1, 2, 3], [4, 5, 6]])
# print(arr1)
# print(arr1+arr1)
# print(arr1*arr1)
# print(arr1-arr1)
# print(arr1/arr1)
#
# arr2 = np.arange(10)
# print(arr2[5])
# print(arr2[:])
# print(arr2[::-1])
# print(arr2[2:-1])
#
# arr3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(arr3)
# print(arr3[2])
# print(arr3[0][0])
#
# arr4 = np.array([[[1, 2, 3], [3, 4, 5]], [[5, 6, 7], [7, 8, 9]]])
# print(arr4)
# print(arr4[0])
# arr4[0] = 1
# print(arr4)
# print(arr4[1][0])
#
# arr5 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(arr5)
# print(arr5[::-1])
# print(arr5[:-1])
# print(arr5[1:, 1:2])
# print(arr5[:, :1])

# ----- #

# # BOOLEAN INDEXING
# names = np.array(['bob', 'joe', 'will'])
# print(names)
# data = np.random.randn(7, 4)
# print(data)
#
# print(names=='bob')
# print(names!='bob')

# ----- #

# # FANCY INDEXING
# arr1 = np.empty([8, 4])
# print(arr1)
#
# for i in range(8):
#     arr1[i] = i
# print(arr1)
#
# arr1[0] = 93
# print(arr1)
# print(arr1[[0, 1, 3]])
#
# arr2 = np.arange(32).reshape(8, 4)
# print(arr2)
# print(arr2[np.ix_([4, 5, 7, 2], [0, 3, 1])])

# ----- #

# # TRANSPOSING ARRAYS AND SWAPPING AXES
# arr1 = np.arange(15).reshape(3, 5)
# print(arr1)
# print(arr1.T)
#
# arr2 = np.random.randn(6, 3)
# print(arr2)
# print(np.dot(arr2.T, arr2))
#
# arr3 = np.arange(16).reshape((2, 2, 4))
# print(arr3)
# print(arr3.transpose((1, 0, 2)))
# print(arr3.swapaxes(1, 2))

# ----- #

# # UNIVERSAL FUNCTIONS: FAST ELEMENT-WISE ARRAY FUNCTIONS
# arr1 = np.arange(16)
# print(np.sqrt(arr1))
# print(np.exp(arr1))
# print(np.random.randn(8))
#
# x = np.array([1, 2, 3, 4, 5, 6, 7])
# y = np.array([2, 1, 3, 4, 3, 8, 9])
# print(np.maximum(x, y))

# ----- #

# # DATA PROCESSING USING ARRAYS
# print(points)
# xs, ys = np.meshgrid(points, points)
# print(xs)
# print(ys)
#
# z = np.sqrt(xs ** 2 + ys ** 2) # calculating distance from the origin
# print(z)
# plt.imshow(z, cmap=plt.cm.gray)
# plt.colorbar()
# plt.title('Image plot of $\sqrt{x^2 + y^2}$ for a grid of values')
# plt.show()

# ----- #

# # EXPRESSING CONDITIONAL LOGIC AS ARRAY OPERATIONS
# xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
# yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
# cond = np.array([True, False, True, True, False])
# result = [(x if c else y) for x, y, c in zip(xarr, yarr, cond)]
# print(result)
#
# result_effective = np.where(cond, xarr, yarr)
# print(result_effective)
#
# arr1 = np.random.randn(4, 4)
# print(arr1)
# print(np.where(arr1 > 0, 2, -2))
# print(np.where(arr1 > 0, 2, arr1))
#
# # -- using for loop
# cond1 = np.array([True, False, True, True, False])
# cond2 = np.array([False, True, True, False, False])
# result = []
# for i in range(len(cond1)):
#     if cond1[i] and cond2[i]:
#         result.append(0)
#     elif cond1[i]:
#         result.append(1)
#     elif cond2[i]:
#         result.append(2)
#     else:
#         result.append(3)
# print(result)
#
# # -- using np.where()
# print(
#     np.where(
#         cond1 & cond2, 0,
#         np.where(
#             cond1, 1,
#             np.where(
#                 cond2, 2, 3
#             )
#         )
#     )
# )

# ----- #

# # MATHEMATICAL AND STATISTICAL METHODS
# arr1 = np.random.randn(5, 4)
# print(arr1)
#
# print(arr1.mean())
# print(arr1.sum())
# print(arr1.mean(axis=1))  # row wise mean
# print(arr1.sum(0))  # column wise sum
#
# arr2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(arr2)
# print(arr2.cumsum(0))  # r1 = r1 + r0 and r2 = r2 + r1 + r0
# print(arr2.cumprod(1))  # c1 = c1 * c0 and c2 = c2 * c1 * c0

# ----- #

# # METHODS FOR BOOLEAN ARRAYS
# arr1 = np.random.randn(100)
# print((arr1 > 0).sum())  # sum of all positive values
#
# bool_vals = np.array([True, False, False, True, True])
# print(bool_vals.any())  # checks if any value is True
# print(bool_vals.all())  # checks if all values are True

# ----- #

# # SORTING
# arr1 = np.random.randn(10)
# print(arr1)
# arr1.sort()
# print(arr1)  # sorted in ascending order
#
# arr2 = np.random.randn(4, 4)
# print(arr2)
# arr2.sort()
# print(arr2.ndim)
# print(arr2)  # sorting the 2-d array

# ----- #

# # UNIQUE AND OTHER SET LOGIC
# names = np.array(['bob', 'joe', 'will', 'chris', 'alex', 'alex', 'bob', 'william'])
# print(np.unique(names))  # prints unique sorted names

# ----- #

# # LINEAR ALGEBRA
# x = np.array([[1, 2, 3], [4, 5, 6]])
# y = np.array([[6, 23], [-1, 7], [8, 9]])
# print(np.dot(x, y))  # prints the dot product of x and y
# print(np.dot(x, np.ones(3)))
#
# x = np.random.randn(5, 5)
# b = np.random.randn(5)
# print(x)
# print(np.linalg.inv(x))  # prints the inverse of 5 x 5 matrix
# print(np.dot(x, np.linalg.inv(x)))  # prints the dot product of x and x's inverse
# print(np.linalg.det(x))  # prints the determinant of matrix x
# print(np.linalg.solve(x, b))  # solves the matrix x