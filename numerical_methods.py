#!/usr/bin/python3

from copy import deepcopy

class Matrix:
  @classmethod
  def eye(cls, size=0, mytype=float):
    values = list()
    for i in range(size):
      values.append(list())
      for j in range(size):
        if i == j:
          values[i].append(mytype(1.0))
        else:
          values[i].append(mytype(0.0))
    return cls(values, mytype=mytype)

  @classmethod
  def permutation(size=0, index1=0, index2=0):
    R = self.__class__.eye(size=size, mytype=float)
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

  def __init__(self, size=0, value=None, mytype=float):
    self.__items = list()
    self.__type = mytype
    if type(size) == list and value is None:
      if type(size[0]) != list:
        self.__n = int(1)
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
        self.__items.extend([self.__type(size[i,j]) for j in range(self.__m)])
    else:
      if type(size) == int or type(size) == float:
        self.__n = int(size)
        self.__m = int(size)
      elif type(size) == tuple or type(size) == list:
        self.__n = int(size[0])
        self.__m = int(size[1])
      else:
        raise ValueError('size must be an int or a tuple or list of two ints')
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
    if type(r1) == int:
      r1 = [r1]
    if type(r2) == int:
      r2 = [r2]
    if len(r1) == len(r2):
      R = Matrix.permutation(size=(A.nrows,A.ncols), index1=r1, index2=r2)
      A = R * A
    else:
      raise ValueError('row indexes must be of same length')

  @staticmethod
  def swap_cols(A, c1=0, c2=0):
    if type(c1) == int:
      c1 = [c1]
    if type(c2) == int:
      c2 = [c2]
    if len(c1) == len(c2):
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
  def backsubstitution(A, b, x=None):
    ret = False
    if x is None:
      ret = True
    x = Matrix(size=(b.nrows, b.ncols), value=0.0)
    for i in range(A.nrows - 1, -1, -1):
      for k in range(b.ncols):
        x[i,k] = b[i,k]
        for j in range(i + 1, A.ncols):
          x[i,k] -= a[i,j] * x[j,k]
        x[i,k] /= a[i,i]
    if ret:
      return x

  @staticmethod
  def forwardsubstitution(A, b, x=None):
    ret = False
    if x is None:
      ret = True
    x = Matrix(size=(b.nrows, b.ncols), value=0.0)
    for i in range(A.nrows):
      for k in range(b.ncols):
        x[i,k] = b[i,k]
        for j in range(i):
          x[i,k] -= a[i,j] * x[j,k]
        x[i,k] /= a[i,i]
    if ret:
      return x

  @staticmethod
  def tridiag(A, b, x=None):
    ret = False
    if x is None:
      ret = True
    x = Matrix(size=(b.nrows, b.ncols), value=0.0)
    n = A.nrows
    bet = 0.0
    gam = Matrix(size=(1,n), value=0.0, mytype=float)

    if A[0,1] == 0.0:
      raise ValueError('Error tridiag: 0 on diagonal A[0,0]')
    bet = A[0,1]
    for k in range(b.ncols):
      x[0,k] = b[0,k] / bet
    for j in range(1,n):
      gam[j] = A[j-1,2] / bet
      bet = A[j,1] - A[j,0] * gam[j]
      if bet == 0.0:
        raise ValueError('Error tridiag: zero pivot on row {0:n}'.format(j))
      for k in range(b.ncols):
        x[j,k] = (b[j,k] - A[j,0] * x[j-1,k]) / bet
    for j in range(j-2,-1,-1):
      for k in range(b.ncols):
        x[j,k] -= gam[j+1] * x[j+1,k]

    if ret:
      return x

  @staticmethod
  def banded_multiply(A, x, m1, m2, b=None):
    # m1 = number of subdiagonal elements
    # m2 = number of superdiagonal elements
    ret = False
    if b is None:
      ret = True
    tmploop = 0
    n = A.nrows
    b = Matrix(size=(1,n), value=0.0, mytype=float)
    for i in range(n):
      k = i - m1
      tmploop = min(m1 + m2 + 1, n - k)
      # b[i] = 0.0
      for j in range(max(0,-k),tmploop):
        b[i] += A[i,j] * x[j+k]
    if ret:
      return b


class LUdecomposition:
  def __init__(self, A):
    self.__n = A.nrows
    self.__lu = Matrix(size=A)
    self.__d = 0.0
    self.__indx = Matrix(size=(1,A.nrows), value=0.0, mytype=float)
    self.__decompose()

  @property
  def lu(self):
    return self.__lu

  @lu.setter
  def lu(self, A):
    self.__init__(A)

  def __decompose(self):
    TINY = 1.0e-40
    imax = 0
    big = 0.0
    tmp = 0.0
    vv = Matrix(size=(1, self.__n), value=0, mytype=float)
    self.__d = 1.0
    for i in range(self.__n):
      big = 0.0
      for j in range(self.__n):
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
        for j in range(self.__n):
          tmp = self.__lu[imax,j]
          self.__lu[imax,j] = self.__lu[k,j]
          self.__lu[k,j] = tmp
          self.__d *= -1.0
          vv[imax] = vv[k]
        self.__indx[k] = imax
      if self.__lu[k, k] == 0.0:
        self.__lu[k,k] = TINY
      for i in range(k+1, self.__n):
        tmp = self.__lu[i,k] / self.__lu[k,k]
        for j in range(k+1, self.__n):
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
      for j in range(i, self.__n):
        u[i,j] = self.__lu[i,j]
    return u

  @property
  def rowindx(self):
    return self.__indx

  def solve(self, b, x=None):
    ret = False
    if x is None:
      ret = True
    x = deepcopy(b)

    m = b.ncols
    if b.nrows != self.__n or x.nrows != self.__n or b.ncols != x.ncols:
      raise ValueError('LUdecomposition.solve: bad sizes')
    for k in range(m):
      ii = 0
      ip = 0
      sum = 0.0
      for i in range(self.__n):
        ip = self.__indx[i]
        sum = x[ip,k]
        x[ip,k]=x[i,k]
        if ii != 0:
          for j in range(ii-1,j):
            sum -= self.__lu[i,j] * x[j,k]
        elif sum != 0.0:
          ii = i + 1
        x[i,k] = sum
      for i in range(self.__n, -1, -1):
        sum = x[i,k]
        for j in range(i+1, self.__n):
          sum -= self.__lu[i,j] * x[j,k]
        x[i,k] = sum / self.__lu[i,i]
    if ret:
      return x

  def inverse(self, ainv=None):
    ret = False
    if ainv is None:
      ret = True
    ainv = Matrix.eye(size=self.__n, mytype=double)
    self.solve(b=ainv, x=ainv)
    if ret:
      return ainv

  def determinant(self):
    dd = self.__d
    for i in range(self.__n):
      dd *= self.__lu[i,i]
    return dd


class Banded:
  @staticmethod
  def bandwidth(A):
    sub = 0
    sup = 0
    if type(A) == list:
      n = len(A)
      for j in range(n):
        for i in range(j + 1, n):
          if A[i][j] != 0.0:
            sub = max(sub, i - j)
        for i in range(j - 1, -1, -1):
          if A[i][j] != 0.0:
            sup = max(sup, j - i)
    elif type(A) == Matrix:
      n = A.nrows
      for j in range(n):
        for i in range(j + 1, n):
          if A[i,j] != 0.0:
            sub = max(sub, i - j)
        for i in range(j - 1, -1, -1):
          if A[i,j] != 0.0:
            sup = max(sup, j - i)
    else:
      raise TypeError(f'input of type {str(type(A))} not supported, must be either Matrix or a list of lists.')
    return sub, sup

  @classmethod
  def convert(cls, A):
    sub, sup = cls.bandwidth(A)
    tmploop = 0
    if type(A) == list:
      n = len(A)
      a = Matrix(size=(len(A[0]), sub + sup + 1), value = 0.0)
      for i in range(n):
        k = i - sub
        tmploop = min(sub + sup + 1, n - k)
        m = max(k, 0)
        for j in range(max(0,-k), tmploop):
          a[i,j] = A[i][m]
          m += 1

    elif type(A) == Matrix:
      n = A.nrows
      a = Matrix(size=(A.nrows, sub + sup + 1), value = 0.0)
      for i in range(n):
        k = i - sub
        tmploop = min(sub + sup + 1, n - k)
        m = max(k, 0)
        for j in range(max(0,-k), tmploop):
          a[i,j] = A[i,m]
          m += 1

    else:
      raise TypeError(f'input of type {str(type(A))} not supported, must be either Matrix or a list of lists.')

    return cls(a, sub, sup)

  def unravel(self):
    n = self.__n
    sub = self.__sub
    sup = self.__sup
    tmploop = 0
    a = Matrix(size=(n, n), value = 0.0)
    for i in range(n):
      k = i - sub
      tmploop = min(sub + sup + 1, n - k)
      m = max(k, 0)
      for j in range(max(0,-k), tmploop):
        a[i,m] = self.__b[i,j]
        m += 1
    return a

  def __init__(self, A, sub, sup):
    self.__n = A.nrows
    self.__au = deepcopy(A)
    self.__al = Matrix(size=(A.nrows, sub), value=0.0, mytype=float)
    self.__indx = Matrix(size=(1, A.nrows), value=0, mytype=int)
    self.__sub = sub
    self.__sup = sup
    self.__decompose()

  def __decompose(self):
    TINY = 1.0e-40
    tmp = 0.0
    mm = self.__sub + self.__sup + 1
    l = self.__sub
    for i in range(self.__sub):
      for j in range(self.__sub, mm):
        self.__au[i,j - l] = self.__au[i,j]
      l -= 1
      for j in range(mm - l - 1, mm):
        self.__au[i,j] = 0.0
    d = 1.0
    l = self.__sub
    for k in range(self.__n):
      tmp = self.__au[k,0]
      i = k
      if l < self.__n:
        l += 1
      for j in range(k+1, l):
        if abs(self.__au[j,0]) > abs(tmp):
          tmp = self.__au[j,0]
          i = j
      self.__indx[k] = i + 1
      if tmp == 0.0:
        self.__au[k,0] = TINY
        if i != k:
          d *= -1.0
          self.__au.swap_rows(k, i)
        for i in range(k+1, l):
          tmp = self.__au[i,0] / self.__au[k,0]
          self.__al[k, i - k - 1] = tmp
          for j in range(mm):
            self.__au[i,j-1] = self.__au[i,j] - tmp * self.__au[k,j]
          self.__au[i,mm -1] = 0.0

  def __str__(self):
    return str(self.__b)


# Cuthill-McKee and Reverse Cuthill-McKee algorithm
# https://ciprian-zavoianu.blogspot.com/2009/01/project-bandwidth-reduction.html
class CuthillMcKee:
  @classmethod
  def __degrees(cls, A):
    d = list()
    for i in range(A.nrows):
      d.append(0)
      for j in range(A.ncols):
        if i != j and A[i,j] != 0.0:
          d[i] += 1
    return d

  @classmethod
  def __sort_by_degree(cls, indx, degrees):
    # ascending
    tmp = 0
    n = len(indx)
    for i in range(n - 1):
      for j in range(i, n):
        if degrees[indx[i]] < degrees[indx[j]]:
          tmp = indx[i]
          indx[i] = indx[j]
          indx[j] = tmp

  @classmethod
  def __min_degree(cls, degrees, exclude=list()):
    min = max(degrees)
    indx = -1
    for i in range(len(degrees)):
      if i not in exclude:
        if degrees[i] < min:
          min = degrees[i]
          indx = i
    return indx

  @classmethod
  def normal(cls, A):
    degrees = cls.__degrees(A)
    Q = list()
    R = list()
    k = -1
    while len(R) < A.nrows:
      min_degree = cls.__min_degree(degrees, exclude=R)
      Q.append(min_degree)
      while len(Q) > 0:
        i = Q.pop(0)
        if i not in R:
          R.append(i)
          indx = list()
          for j in range(A.ncols):
            if A[i,j] != 0 and j not in R:
              indx.append(j)
          cls.__sort_by_degree(indx, degrees)
          Q.extend(indx)
        k += 1
        print('{0:3n}: R=({1}) Q=({2})'.format(k, ' '.join([str(r) for r in R]), ' '.join([str(q) for q in Q])))
    print()
    return R

  @classmethod
  def reversed(cls, A):
    R = cls.normal(A)
    return R[::-1]

  @classmethod
  def print(cls, A, R=None):
    if R is None:
      R = [i for i in range(A.nrows)]
    msg = ''
    for i in range(A.nrows):
      msg += ''.join(['{0:3n}'.format(A[R[i],R[j]]) for j in range(A.ncols)])
      msg += '\n'
    msg = msg
    print(msg)


if __name__ == '__main__':
  # a = Matrix(size=int(3), value=0.0)
  # a = Matrix.eye(size=3)
  # print('a = \n{0}\n'.format(a))
  # # for i in range(a.nrows):
  # #   for j in range(a.ncols):
  # #     print(a[i,j])

  # # gauss-jordan
  # a = Matrix([[2.0, 6.0, -2.0], [1.0, 6.0, -4.0], [-1.0, 4.0, 9.0]])
  # b = Matrix([[2.0, 6.0, -2.0], [1.0, 6.0, -4.0], [-1.0, 4.0, 9.0]])
  # print('a = \n{0}\n'.format(a))
  # # for i in range(a.nrows):
  # #   for j in range(a.ncols):
  # #     print(a[i,j])
  # LinAlg.gauss_jordan_full_pivot(a)
  # print('a = \n{0}\n'.format(a))
  # # for i in range(a.nrows):
  # #   for j in range(a.ncols):
  # #     print(a[i,j])
  # print('a * a-1 =\n{0}\n'.format(b * a))

  # lu = LUdecomposition(b)
  # print('l =\n{0}\nu =\n{1}\nindx =\n{2}\n'.format(lu.L, lu.U, lu.rowindx))

  # b = Matrix(size=[[3, 1, 0, 0, 0, 0, 0],
  #                  [4, 1, 5, 0, 0, 0, 0],
  #                  [9, 2, 6, 5, 0, 0, 0],
  #                  [0, 3, 5, 8, 9, 0, 0],
  #                  [0, 0, 7, 9, 3, 2, 0],
  #                  [0, 0, 0, 3, 8, 4, 6],
  #                  [0, 0, 0, 0, 2, 4, 4]], mytype=float)
  # print('b = \n{0}\n'.format(b))
  # m1, m2 = Banded.bandwidth(b)
  # print(f'm1 = {m1:n}, m2 = {m2:n}')
  # bb = Banded.convert(b)
  # print('banded b = \n{0}\n'.format(bb))

  A = Matrix(size=[[1, 0, 0, 0, 1, 0, 0, 0],
                   [0, 1, 1, 0, 0, 1, 0, 1],
                   [0, 1, 1, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 1, 0],
                   [1, 0, 1, 0, 1, 0, 0, 0],
                   [0, 1, 0, 0, 0, 1, 0, 1],
                   [0, 0, 0, 1, 0, 0, 1, 0],
                   [0, 1, 0, 0, 0, 1, 0, 1]], mytype=int)
  CuthillMcKee.print(A)
  R = CuthillMcKee.normal(A)
  CuthillMcKee.print(A, R)
  R = CuthillMcKee.reversed(A)
  CuthillMcKee.print(A, R)

