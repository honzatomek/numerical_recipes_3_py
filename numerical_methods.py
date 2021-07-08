#!/usr/bin/python3

from copy import deepcopy

class Vector:
  def __init_(self, size=0, value=0):
    self.__size = size
    self.__items = list()
    if self.__size > 0:
      if type(self.value) == list:
        self.__items = [value[i] for i in range(self.__size)]
      else:
        self.__items == [value for i in range(self.__size)]

  @property
  def size(self):
    return self.__size

  @size.setter
  def size(self, new_size):
    self.__size = new_size
    self.__items = [0.0 for i in range(self.__size)]

  def __getitem__(self, index):
    return self.__items[index]

  def __setitem__(self, index, new_value):
    if type(index) == slice:
      if type(new_value) == list:
        for i in range(index):
          self.__items[i] = new_value[i]
      else:
        if type(new_value) == list:
          j = 0
          for i in range(index):
            self.__items[i] = new_value[j]
            j += 1
        else:
          for i in range(index):
            self.__items[i] = new_value
    else:
      self.__items[index] = new_value


class Matrix:
  @classmethod
  def eye(cls, size=0, type=float):
    values = list()
    for i in range(size):
      values.append(list())
      for j in range(size):
        if i == j:
          values[i].append(type(1.0))
        else:
          values[i].append(type(0.0))
    return cls(values)

  def __init__(self, size=0, value=None):
    self.__items = list()
    if type(size) == list and value is None:
      if type(size[0]) != list:
        self.__n = 1
        self.__m = len(size)
        self.__items = deepcopy(size)
      else:
        self.__n = len(size)
        self.__m = len(size[0])
        for i in range(self.__n):
          self.__items.extend([size[i][j] for j in range(self.__m)])
    else:
      if type(size) == int:
        self.__n = size
        self.__m = size
      elif type(size) == tuple or type(size) == list:
        self.__n = size[0]
        self.__m = size[1]
      else:
        raise ValueError('size must be an int or a tuple oe list of two ints')
      if value is None:
        value = 0.0
      for i in range(self.__n):
        self.__items.extend([value for j in range(self.__m)])

  @property
  def size(self):
    return (self.__n, self.__m)

  @size.setter
  def size(self, new_size):
    if type(new_size) == int:
      self.__n = new_size
      self.__m = new_size
    elif type(new_size) == tuple or type(new_size) == list:
      self.__n = new_size[0]
      self.__m = new_size[1]
    else:
      raise ValueError('newsize must bw an int or a tuple oe list of two ints')

    self.__items = list()
    for i in range(self.__n):
      self.__items.extend([0.0 for j in range(self.__m)])

  @property
  def nrows(self):
    return self.__n

  @property
  def ncols(self):
    return self.__m

  def __getitem__(self, index):
    if type(index) == list or type(index) == tuple:
      return self.__items[index[0] * self.__m + index[1]]
    elif self.__n == 1:
      return self.__items[index]
    else:
      raise ValueError('wrong index')

  def __setitem__(self, index, new_value):
    if type(index) == list or type(index) == tuple:
      self.__items[index[0] * self.__m + index[1]] = new_value
    elif self.__n == 1:
      self.__items[index] = new_value
    else:
      raise ValueError('wrong index')

  def __add__(self, B):
    if self.nrows == B.nrows and self.ncols == B.ncols:
      C = self.__class__(size=(self.nrows, self.ncols), value=0.0)
      for i in range(self.nrows):
        for j in range(self.ncols):
          C[i,j] = self[i,j] + B[i,j]
      return C
    else:
      raise ValueError('matrix B has to have the same amount of rows and columns')

  def __sub__(self, B):
    if self.nrows == B.nrows and self.ncols == B.ncols:
      C = self.__class__(size=(self.nrows, self.ncols), value=0.0)
      for i in range(self.nrows):
        for j in range(self.ncols):
          C[i,j] = self[i,j] - B[i,j]
      return C
    else:
      raise ValueError('matrix B has to have the same amount of rows and columns')

  def __mul__(self, B):
    A = self
    if type(B) == int or type(B) == double:
      C = self.__class__(size=(A.nrows,A.ncols), value=0.0)
      for i in range(A.nrows):
        for j in range(A.ncols):
          C[i,j] = A[i,j] * B
      return C
    if A.ncols == B.nrows:
      C = self.__class__(size=(A.nrows,B.ncols), value=0.0)
      for i in range(A.rows):
        for j in range(A.ncols):
          for k in range(B.ncols):
            C[i,k] += A[i,j] * B[j,k]
      return C
    else:
      raise ValueError('matrix B has to have the same amount of rows as A has columns')

  def __iadd__(self, B):
    if self.nrows == B.nrows and self.ncols == B.ncols:
      for i in range(self.nrows):
        for j in range(self.ncols):
          self[i,j] += B[i,j]
      return self
    else:
      raise ValueError('matrix B has to have the same amount of rows and columns')

  def __isub__(self, B):
    if self.nrows == B.nrows and self.ncols == B.ncols:
      for i in range(self.nrows):
        for j in range(self.ncols):
          self[i,j] -= B[i,j]
      return self
    else:
      raise ValueError('matrix B has to have the same amount of rows and columns')

  def __imul__(self, B):
    return self.__mul__(B)

  def swap_rows(self, i, j):
    for k in range(self.__m):
      tmp = self.__items[i * self.__m + k]
      self.__items[i * self.__m + k] = self.__items[j * self.__m + k]
      self.__items[j * self.__m + k] = tmp

  def swap_cols(self, i, j):
    for k in range(self.__n):
      tmp = self.__items[k * self.__m + i]
      self.__items[k * self.__m + i] = self.__items[k * self.__m + j]
      self.__items[k * self.__m + j] = tmp

  def swap(self, first, second):
    tmp = self.__items[first[0] * self.__m + first[1]]
    self.__items[first[0] * self.__m + first[1]] = self.__items[second[0] * self.__m + second[1]]
    self.__items[second[0] * self.__m + second[1]] = tmp

  def row(self, r):
    return self.__class__(self.__items[r * self.__m: r * self.__m + self.__n])

  def col(self, c):
    column = list()
    for i in range(self.__n):
      column.append([self.__items[i * self.__m + c]])
    return self.__class__(column)


class LinAlg:
  @staticmethod
  def swap_rows(A, r1=0, r2=0):
    R = Matrix.eye(size=(A.nrows,A.ncols))
    if type(r1) == int and type(r2) == int:
      r1 = [r1]
      r2 = [r2]
    if type(r1) == list and type(r2) == list and len(r1) == len(r2):
      for i in range(len(r1)):
        R[r1[i],r1[i]] = 0.0
        R[r2[i],r2[i]] = 0.0
        R[r1[i],r2[i]] = 1.0
        R[r2[i],r1[i]] = 1.0
      A = R * A
    else:
      raise ValueError('row indexes must be of same length')

  @staticmethod
  def gauss_jordan_full_pivot(A, b=None):
    if b is None:
      b = Matrix(size=(A.nrows,0), value=0.0)
    i = 0
    irow = 0
    icol = 0
    j = 0
    k = 0
    l = 0
    ll = 0
    n = A.nrows
    m = b.ncols

    big = 0.0
    tmp = 0.0
    pivinv = 0.0

    indxr = Matrix((1, A.nrows), 0)    # bookkeeping on the pivoting
    indxc = Matrix((1, A.nrows), 0)    # bookkeeping on the pivoting
    ipiv = Matrix((1, A.nrows), 0)    # bookkeeping on the pivoting

    for i in range(n):
      big = 0.0
      for j in range(n):
        if ipiv[j] != 1:
          for k in range(n):
            if ipiv[k] == 0:
              if abs(A[j,k]) >= big:
                big = abs(A[j,k])
                irow = j
                icol = k
      ipiv[icol] += 1
      if irow != icol:
        A.swap_rows(irow, icol)
        b.swap_rows(irow, icol)
      indxr[i] = irow
      indxc[i] = icol
      if A[icol,icol] == 0.0:
        raise ValueError('gauss_jordan_full_pivot - singular matrix')
      pivinv = 1.0 / A[icol,icol]
      A[icol,icol] = 1.0
      for l in range(n):
        A[icol,l] *= pivinv
      for l in range(m):
        b[icol,l] *= pivinv
      for ll in range(n):
        if ll != icol:
          tmp = A[ll,icol]
          A[ll,icol] = 0.0
          for l in range(n):
            A[ll,l] -= A[icol,l] * tmp
          for l in range(m):
            b[ll,l] -= b[icol,l] * tmp


if __name__ == '__main__':
  a = Matrix(3, 0.0)
  a = Matrix.eye(3)
  for i in range(a.nrows):
    for j in range(a.ncols):
      print(a[i,j])

  # gauss-jordan
  a = Matrix([[2.0, 6.0, -2.0], [1.0, 6.0, -4.0], [-1.0, 4.0, 9.0]])
  for i in range(a.nrows):
    for j in range(a.ncols):
      print(a[i,j])
  LinAlg.gauss_jordan_full_pivot(a)
  for i in range(a.nrows):
    for j in range(a.ncols):
      print(a[i,j])

