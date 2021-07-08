#!/usr/bin/python3

from copy import deepcopy

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

  @classmethod
  def permutation(size=0, index1=0, index2=0):
    R = self.__class__.eye(size=size)
    if type(index1) == int and type(index2) == int:
      index1 = [index1]
      index2 = [index2]
    if type(index1) == list and type(index2) == list and len(index1) == len(index2):
      for i in range(len(index1)):
        R[index1[i],index1[i]] = 0.0
        R[index2[i],index2[i]] = 0.0
        R[index1[i],index2[i]] = 1.0
        R[index2[i],index1[i]] = 1.0
      return R
    else:
      raise ValueError('indexes must be of the same length')

  def __init__(self, size=0, value=None, type=float):
    self.__items = list()
    self,__type = type
    if type(size) == list and value is None:
      if type(size[0]) != list:
        self.__n = 1
        self.__m = len(size)
        self.__items = deepcopy(size)
      else:
        self.__n = len(size)
        self.__m = len(size[0])
        for i in range(self.__n):
          self.__items.extend([self.__type(size[i][j]) for j in range(self.__m)])
    elif type(size) == self.__class__:
      self.__n = size.nrows
      self.__m = size.ncols
      for i in range(self.__n):
        self.__items.extend(self.__type([size[i,j]) for j in range(self.__m)])
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
        self.__items.extend([self.__type(value) for j in range(self.__m)])

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
      self.__items.extend([self.__type(0.0) for j in range(self.__m)])

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
    if type(B) == int or type(B) == float:
      C = self.__class__(size=(A.nrows,A.ncols), value=0.0)
      for i in range(A.nrows):
        for j in range(A.ncols):
          C[i,j] = A[i,j] * B
      return C
    if A.ncols == B.nrows:
      C = self.__class__(size=(A.nrows,B.ncols), value=0.0)
      for i in range(A.nrows):
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

  def __str__(self):
    retval = ''
    for i in range(self.__n):
      for j in range(self.__m):
        retval += '{0:12.5f}'.format(self[i,j])
      retval += '\n'
    retval = retval[0:-1]
    return retval

  def __repr__(self):
    return 'Matrix, rows = {0:n}, columns{1:n}'.format(self.__n, self.__m)

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
    R = Matrix.permutation(size=(A.nrows,A.ncols), index1=r1, index2=r2)
      A = R * A
    else:
      raise ValueError('row indexes must be of same length')

  @staticmethod
  def swap_cols(A, c1=0, c2=0):
    R = Matrix.permutation(size=(A.nrows,A.ncols), index1=c1, index2=c2)
      A = A * R
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

  @staticmethod
  def backsubstitution(A, b):
    x = Matrix(size=(b.nrows, b.ncols), value=0.0)
    for i in range(A.nrows - 1, -1, -1):
      for k in range(b.ncols):
        x[i,k] = b[i,k]
        for j in range(i + 1, A.ncols):
          x[i,k] -= a[i,j] * x[j,k]
        x[i,k] /= a[i,i]
    return x

  @staticmethod
  def forwardsubstitution(A, b):
    x = Matrix(size=(b.nrows, b.ncols), value=0.0)
    for i in range(A.nrows):
      for k in range(b.ncols):
        x[i,k] = b[i,k]
        for j in range(i):
          x[i,k] -= a[i,j] * x[j,k]
        x[i,k] /= a[i,i]
    return x

class LUdecomposition:
  def __init__(self, A):
    self.__n = A.nrows
    self.__lu = Matrix(size=A)
    self.__d = 0.0
    self.__indx = Matrix(size=(1,A.nrows), value=0.0)
    self.__decompose()

  @property
  def lu(self):
    return self.__lu

  @lu.setter
  def.lu(self, A):
    self.__init__(A)

  def __decompose(self):
    TINY = 1.0e-40
    imax = 0
    big = 0.0
    tmp = 0.0
    vv = Matrix(size=(1, self.__n), value=0, type=float)
    self.__d = 1.0
    for i in range(self.__n):
      big = 0.0
      for j in range(n):
        tmp = abs(self.__lu[i,j])
        if tmp > big:
          big = tmp
        if big == 0.0:
          raise ValueError('singular matrix')
        vv[i] = 1.0 / big
    for k in range(self.__n):
      big = 0.0
      for i in range(k, self.__n):
        tmp = vv[i] * abs(self.__lu[i,k])
        if tmp > big:
          big = tmp
          imax = i
      if k != imax:
        for j in range(n):
          tmp = self.__lu[imax,j]
          self.__lu[imax,j] = self.__lu[k,j]
          self.__lu[k,j] = tmp
          self.__d *= -1.0
          vv[imax] = vv[k]
      indx[k] = imax
      if self.__lu[k, k] == 0.0:
        self.__lu[k,k] = TINY
      for i in range(k+1, self.__n):
        tmp = self.__lu[i,] / self.__lu[k,k]
        for j in range(k+1, n):
          self.__lu[i,j] -= tmp * self.__lu[k,j]

  @property
  def L(self):
    l = Matrix.eye(size=self.__n)
    for i in range(self.__n):
      for j in range(i):
        l[i,j] = self.__lu[i,j]
    return l

  @property
  def U(self):
    u = Matrix(size=self.__n, value=0.0)
    for i in range(self.__n):
      for j in range(i + 1, self.__n):
        u[i,j] = self.__lu[i,j]
    return u


if __name__ == '__main__':
  a = Matrix(3, 0.0)
  a = Matrix.eye(3)
  print('a = \n{0}\n'.format(a))
  # for i in range(a.nrows):
  #   for j in range(a.ncols):
  #     print(a[i,j])

  # gauss-jordan
  a = Matrix([[2.0, 6.0, -2.0], [1.0, 6.0, -4.0], [-1.0, 4.0, 9.0]])
  b = Matrix([[2.0, 6.0, -2.0], [1.0, 6.0, -4.0], [-1.0, 4.0, 9.0]])
  print('a = \n{0}\n'.format(a))
  # for i in range(a.nrows):
  #   for j in range(a.ncols):
  #     print(a[i,j])
  LinAlg.gauss_jordan_full_pivot(a)
  print('a = \n{0}\n'.format(a))
  # for i in range(a.nrows):
  #   for j in range(a.ncols):
  #     print(a[i,j])
  print('a * a-1 =\n{0}\n'.format(b * a))


