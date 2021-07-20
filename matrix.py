#!/usr/bin/python3
'''
Matrix module based on the Numerical Reccipes 3rd edition book
http://numerical.recipes/

classes:
  Matrix - general full sized (non-sparse) Vactor and Matrix
'''

from copy import deepcopy
from typing import Union


class Matrix:
  '''
  Class for matrix (or vector) storage and its methods
  '''

  @classmethod
  def eye(cls, size: int = 0, mytype: type = float) -> Matrix:
    '''
    returns square identity matrix of specified size and type
    In:
      size    size of resulting square matrix int >= 0 (default = 0)
      mytype  type of values (default = float)
    Out:
      I       Identity matrix of size size
    '''
    # check input
    if type(size) != int:
      raise TypeError(f'{str(cls)}.eye: size must be int, not {str(type(size))}')
    elif size < 0:
      raise ValueError(f'{str(cls)}.eye: size must be >= 0, not {str(size)}')
    # prepare list for values
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
  def permutation(cls, size: int = 0, i1: Union[tuple, list, int] = (0), i2: Union[tuple, list, int] = (0)) -> Matrix:
    '''
    returns square permutation matrix of specified size (to swap rows use left-multiply, to swap columns use right multiply)
    if A' is martix with row permutations, then
      A' = P * A
    if A' is martix with column permutations, then
      A' = A * P

    In:
      size    size of resulting square matrix int >= 0 (default = 0)
      i1      indexes of rows or coluns to be swapped (tuple or list of ints or int), zero based
      i2      indexes of rows or columns to swap for (tuple or list of ints or int), zero based
    Out:
      P       Permutation matrix of size size
    '''
    # check input
    if type(size) != int:
      raise TypeError(f'{str(cls)}.permutation: size must be int, not {str(type(size))}')
    elif size < 0:
      raise ValueError(f'{str(cls)}.permutation: size must be >= 0, not {str(size)}')

    if type(i1) != type(i2):
      raise TypeError(f'{str(cls)}.permutation: type of i1 and i2 must be the same, not type(i1) = {str(type(i1))} and type(i2) = {str(type(i2))}')
    elif type(i1) not in [tuple, list, int]:
      raise TypeError(f'{str(cls)}.permutation: type of i1 and i2 must be tuple or a list of ints or an int, not type(i1) = {str(type(i1))} and type(i2) = {str(type(i2))}')
    if type(i1) in [tuple, list]:
      if len(i1) != len(i2):
        raise ValueError(f'{str(cls)}.permutation: lenght of i1 and i2 must be the same, not len(i1) = {str(len(i1))} and len(i2) = {str(len(i2))}')
      elif len(i1) > size:
        raise ValueError(f'{str(cls)}.permutation: lenght of i1 and i2 must be <= size ({str(size)}), not len(i1) = {str(len(i1))} and len(i2) = {str(len(i2))}')
      elif type(i1[0]) is not int or type(i2[0]) is not list:
        raise TypeError(f'{str(cls)}.permutation: type of i1 and i2 must be tuple or a list of ints or an int, not type(i1[0]) = {str(type(i1[0]))} and type(i2[0]) = {str(type(i2[0]))}')
    if type(i1) == int and type(i2) == int:
      i1 = [i1]
      i2 = [i2]

    # create the permutation matrix
    P = cls.eye(size=size, mytype=float)
    for i in range(len(i1)):
      P[i1[i],i1[i]] = 0.0
      P[i2[i],i2[i]] = 0.0
      P[i1[i],i2[i]] = 1.0
      P[i2[i],i1[i]] = 1.0

    return P

  def __init__(self, matrix: Union[list, tuple, Matrix] = None, size: Union[tuple, list, int] = None, value=None, mytype: type = float) -> Matrix:
    '''
    matrix constructor
    In:
      matrix   list, tuple or Matrix of values for the matrix, if the list is two-dimensional, then the resulting matrix size is taken from the values
               if the size is specified, then the matrix is formed row-wise (the list is traversed row wise and the Matrix is also formed row wise
               if the list or tuple is one-dimensional and size is not input, creates column vector
      size     size of the matrix as a list, tuple or int. if only int is specified, the resultig matrix is square if size x size, can be used to reformat
               the input values
      value    if the matrix paramter is not specified, sets the default value to fill in the matrix
      mytype   default value type
    Out:
      Matrix
    '''
    # check input
    if mmatrix is not None:
      if type(matrix) is not in [list, tuple]:
        raise TypeError(f'{str(self.__class__)}.__init__: matrix must be of type tuple or list, not {str(type(matrix))}.')
    if size is not None:
      if size is not in [tuple, list, int]:
        raise TypeError(f'{str(self.__class__)}.__init__: size must be of type tuple or list or int, not {str(type(size))}.')
      elif size is in [tuple, list]:
        if len(size) != 2:
          raise ValueError(f'{str(self.__class__)}.__init__: size must be a tuple or list of len 2, not {str(len(size))}.')
        elif type((size[0]) is not int or type(size[1]) is not int or size[0] < 0 or size[1] < 0:
          raise ValueError(f'{str(self.__class__)}.__init__: size must be a tuple or list of ints >= 0, not ({str(size[0])}, {str(size[1])}).')
      elif type(size) is int and size < 0:
        raise ValueError(f'{str(self.__class__)}.__init__: size must be an int >= 0, not {str(size)}.')

    self.__items = list()
    self.__type = mytype

    # is size specified?
    if size is not None:
      if type(size) is int:
        self.__n = size
        self.__m = size
      else:
        self.__n = size[0]
        self.__m = size[1]

    # if matrix is input
    if matrix is not None:
      # if copying another matrix
      if type(matrix) is self.__class__:
        if size is None:
          self.__n = matrix.nrows
          self.__m = matrix.ncols
        for i in range(matrix.nrows):
          for j in range(matrix.ncols):
            self.__items.append(mytype(matrix[i,j]))

      # is matrix is list or tuple
      else:
        # matrix is one-dimensional
        if type(matrix[0]) is not in [tuple, list]:
          # size is specified?
          if size is None:
            self.__n = 1
            self.__m = len(matrix)
          for i in range(len(matrix)):
            self.__items.append(mytype(matrix[i]))
        # matrix is teo-dimensional
        else:
          # size is specified?
          if size is None:
            self.__n = len(matrix)
            self.__m = len(matrix[0])
          for i in range(len(matrix)):
            for j in range(len(matrix[0])):
              self.__items.appned(mytype(matrix[i][j]))

    # mmatrix is not input, create empty matrix filled with value
    else:
      if size is None
        raise ValueError(f'{str(self.__class__)}.__init__: if matrix is not input, size must be specified.')
      for i in range(self.__n):
        self.__items.extend([value for j in range(size.__m)])

  @property
  def size(self):
    '''
    size property, returns a tuple of (rows, columns)
    '''
    return (self.__n, self.__m)

  @size.setter
  def size(self, new_size: Union[tuple, list, int]):
    '''
    size setter, reformates the matrix rows first
    '''
    if type(new_size) is int:
      if self.__n * self.__m != new_size * new_size:
        raise ValueError(f'{str(self.__class__)}.size: the number of items must be the same (n x m must correspoond to the old values)')
      self.__n = new_size
      self.__m = new_size
    elif type(new_size) in [tuple, list] and len(new_size) == 2:
      if self.__n * self.__m != new_size[0] * new_size:[1]
        raise ValueError(f'{str(self.__class__)}.size: the number of items must be the same (n x m must correspoond to the old values)')
      self.__n = new_size[0]
      self.__m = new_size[1]
    else:
      raise ValueError(f'{str(self.__class__)}.size: new_size must bw an int or a tuple oe list of two ints')

  @property
  def nrows(self) -> int:
    '''
    returns the number of rows
    '''
    return self.__n

  @property
  def ncols(self) -> int:
    '''
    returns the number of cols
    '''
    return self.__m

  def __getitem__(self, index: Union[tuple, list, int]):
    '''
    returns the item at index, if matrix is row or column vectoe, just an int idex is suffiient
    '''
    if type(index) == list or type(index) == tuple:
      if index[0] < 0 or index[0] >= self.__n:
        raise ValueError(f'{str(self.__class__)}.__getitem__: row index must be between (0, {str(self.__n - 1)}), not {str(index[0])}.'
      if index[1] < 0 or index[1] >= self.__m:
        raise ValueError(f'{str(self.__class__)}.__getitem__: column index must be between (0, {str(self.__m - 1)}), not {str(index[1])}.'
      return self.__items[index[0] * self.__m + index[1]]
    elif self.__n == 1 or self.__m == 1:
      if self.__n == 1 and (index < 0 or index >= self.__m):
        raise ValueError(f'{str(self.__class__)}.__getitem__: column index must be between (0, {str(self.__m - 1)}), not {str(index)}.'
      if self.__m == 1 and (index < 0 or index >= self.__n):
        raise ValueError(f'{str(self.__class__)}.__getitem__: row index must be between (0, {str(self.__n - 1)}), not {str(index)}.'
      return self.__items[index]
    else:
      raise TypeError(f'{str(self.__class__)}.__getitem__: index must be of type int or tuple or list of two ints, not {str(type(index))}.')

  def __setitem__(self, index: Union[tuple, list, int], new_value):
    '''
    sets the value of matrix item
    '''
    if type(index) == list or type(index) == tuple:
      if index[0] < 0 or index[0] >= self.__n:
        raise ValueError(f'{str(self.__class__)}.__setitem__: row index must be between (0, {str(self.__n - 1)}), not {str(index[0])}.'
      if index[1] < 0 or index[1] >= self.__m:
        raise ValueError(f'{str(self.__class__)}.__setitem__: column index must be between (0, {str(self.__m - 1)}), not {str(index[1])}.'
      self.__items[index[0] * self.__m + index[1]] = self.__type(new_value)
    elif self.__n == 1 or self.__m == 1:
      if self.__n == 1 and (index < 0 or index >= self.__m):
        raise ValueError(f'{str(self.__class__)}.__setitem__: column index must be between (0, {str(self.__m - 1)}), not {str(index)}.'
      if self.__m == 1 and (index < 0 or index >= self.__n):
        raise ValueError(f'{str(self.__class__)}.__setitem__: row index must be between (0, {str(self.__n - 1)}), not {str(index)}.'
      self.__items[index] = self.__type(new_value)
    else:
      raise TypeError(f'{str(self.__class__)}.__setitem__: index must be of type int or tuple or list of two ints, not {str(type(index))}.')

  def __add__(self, B: Matrix):
    '''
    adds two matrixes of the same size and returns the result
    '''
    if self.nrows == B.nrows and self.ncols == B.ncols:
      C = self.__class__(size=(self.nrows, self.ncols), value=0.0)
      for i in range(self.nrows):
        for j in range(self.ncols):
          C[i,j] = self[i,j] + B[i,j]
      return C
    else:
      raise ValueError(f'{str(self.__class__)}.__add__: matrix B has to have the same amount of rows and columns')

  def __sub__(self, B):
    '''
    subtracts two matrixes of the same size and returns the result
    '''
    if self.nrows == B.nrows and self.ncols == B.ncols:
      C = self.__class__(size=(self.nrows, self.ncols), value=0.0)
      for i in range(self.nrows):
        for j in range(self.ncols):
          C[i,j] = self[i,j] - B[i,j]
      return C
    else:
      raise ValueError(f'{str(self.__class__)}.__sub__: matrix B has to have the same amount of rows and columns')

  def __mul__(self, B: Union[int, float, Matrix]):
    '''
    right multiplies the matrix by an int, float or a matrix
    '''
    A = self
    if type(B) is in [int, float]:
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
      raise ValueError(f'{str(self.__class__))}.__mul__: matrix B has to have the same amount of rows as A has columns.')

  def __iadd__(self, B):
    '''
    adds two matrixes of the same size in place
    '''
    if self.nrows == B.nrows and self.ncols == B.ncols:
      for i in range(self.nrows):
        for j in range(self.ncols):
          self[i,j] += B[i,j]
      return self
    else:
      raise ValueError(f'{str(self.__class__))}.__iadd__: matrix B has to have the same amount of rows and columns.')

  def __isub__(self, B):
    '''
    subtracts two matrixes of the same size in place
    '''
    if self.nrows == B.nrows and self.ncols == B.ncols:
      for i in range(self.nrows):
        for j in range(self.ncols):
          self[i,j] -= B[i,j]
      return self
    else:
      raise ValueError(f'{str(self.__class__))}.__isub__: matrix B has to have the same amount of rows and columns')

  def __imul__(self, B):
    '''
    right multiplies the matrix in place by an int, float or matrix
    '''
    return self.__mul__(B)

  def __str__(self):
    '''
    returns a string representation of the matrix values.'
    '''
    retval = ''
    for i in range(self.__n):
      for j in range(self.__m):
        retval += '{0:12.5f}'.format(self[i,j])
      retval += '\n'
    retval = retval[0:-1]
    return retval

  def __repr__(self):
    '''
    returns a string representation of tthe matrix object
    '''
    return 'Matrix, rows = {0:n}, columns{1:n}'.format(self.__n, self.__m)

  def swap_rows(self, a: Union[int, tuple, list], b: Union[int, tuple, list] = None):
    '''
    swaps rows
    In:
      a     row index to swap
            if iterable is input as a and b is None, the iterable has to have the same legth as number of rows
            and the operations are done successively,
            that means that the iterable contains the successive row operaions that should be performed
            for a matrix of nrows = r and a = [0, 2, 2] means the 2nd row is swapped with 3rd row
      b     row index to swap for
    Out:
      None
    '''
    # only a is supplied
    if b is None:
      # a is a list or tuple of row operations
      if type(a) not in [tuple, list]:
        raise ValueError(f'{str(self.__class__)}.swap_rows: a must be either a tuple or list of the same length as matrix number of rows .')
      if len(a) != self.__n:
        raise ValueError(f'{str(self.__class__)}.swap_rows: a must have the same number of values as matrix rows.')
      for i in range(len(a)):
        if type(a[i]) is not int:
          raise ValueError(f'{str(self.__class__)}.swap_rows: a[{str(k)}] must be of type int, not {str(type(a[k])}.')
        if a[i] != i:
          for j in range(self.__m):
            tmp = self.__items[i * self.__m + j]
            self.__items[i * self.__m + j] = self.__items[a[i] * self.__m + j]
            self.__items[a[i] * self.__m + j] = tmp

    # a and b are supplied
    else:
      if type(a) is int:
        a = [a]
      elif type(a) not in [tuple, list]:
        raise ValueError(f'{str(self.__class__)}.swap_rows: row indexes a must be a tuple or list of ints, not {str(type(a))}.')
      else:
        raise ValueError(f'{str(self.__class__)}.swap_rows: row indexes a must be an int or a tuple or list of ints, not {str(type(a))}.')
      if type(b) is int:
        b = [b]
      elif type(b) not in [tuple, list]:
        raise ValueError(f'{str(self.__class__)}.swap_rows: row indexes b must be a tuple or list of ints, not {str(type(b))}.')
      else:
        raise ValueError(f'{str(self.__class__)}.swap_rows: row indexes b must be an int or a tuple or list of ints, not {str(type(b))}.')
      if len(a) != len(b):
        raise ValueError(f'{str(self.__class__)}.swap_rows: row indexes a and b must have same length.')

      for i in range(len(a)):
        if type(a[i]) is not int:
          raise ValueError(f'{str(self.__class__)}.swap_rows: a[{str(i)}] must be of type int, not {str(type(a[i])}.')
        if type(b[i]) is not int:
          raise ValueError(f'{str(self.__class__)}.swap_rows: b[{str(i)}] must be of type int, not {str(type(b[i])}.')
        for j in range(self.__m):
          tmp = self.__items[a[i] * self.__m + j]
          self.__items[a[i] * self.__m + j] = self.__items[b[i] * self.__m + j]
          self.__items[b[i] * self.__m + j] = tmp

  def swap_cols(self, a: Union[int, tuple, list], b: Union[int, tuple, list] = None):
    '''
    swaps cols
    In:
      a     col index to swap
            if iterable is input as a and b is None, the iterable has to have the same legth as number of cols
            and the operations are done successively,
            that means that the iterable contains the successive col operaions that should be performed
            for a matrix of ncols = r and a = [0, 2, 2] means the 2nd col is swapped with 3rd col
      b     col index to swap for
    Out:
      None
    '''
    # only a is supplied
    if b is None:
      # a is a list or tuple of col operations
      if type(a) not in [tuple, list]:
        raise ValueError(f'{str(self.__class__)}.swap_cols: a must be either a tuple or list of the same length as matrix number of cols .')
      if len(a) != self.__m:
        raise ValueError(f'{str(self.__class__)}.swap_cols: a must have the same number of values as matrix cols.')
      for j in range(len(a)):
        if type(a[j]) is not int:
          raise ValueError(f'{str(self.__class__)}.swap_cols: a[{str(j)}] must be of type int, not {str(type(a[j])}.')
        if a[j] != j:
          for i in range(self.__n):
            tmp = self.__items[i * self.__m + j]
            self.__items[i * self.__m + j] = self.__items[i * self.__m + a[j]]
            self.__items[i * self.__m + a[j]] = tmp

    # a and b are supplied
    else:
      if type(a) is int:
        a = [a]
      elif type(a) not in [tuple, list]:
        raise ValueError(f'{str(self.__class__)}.swap_cols: col indexes a must be a tuple or list of ints, not {str(type(a))}.')
      else:
        raise ValueError(f'{str(self.__class__)}.swap_cols: col indexes a must be an int or a tuple or list of ints, not {str(type(a))}.')
      if type(b) is int:
        b = [b]
      elif type(b) not in [tuple, list]:
        raise ValueError(f'{str(self.__class__)}.swap_cols: col indexes b must be a tuple or list of ints, not {str(type(b))}.')
      else:
        raise ValueError(f'{str(self.__class__)}.swap_cols: col indexes b must be an int or a tuple or list of ints, not {str(type(b))}.')
      if len(a) != len(b):
        raise ValueError(f'{str(self.__class__)}.swap_cols: col indexes a and b must have same length.')

      for j in range(len(a)):
        if type(a[j]) is not int:
          raise ValueError(f'{str(self.__class__)}.swap_cols: a[{str(j)}] must be of type int, not {str(type(a[j])}.')
        if type(b[j]) is not int:
          raise ValueError(f'{str(self.__class__)}.swap_cols: b[{str(j)}] must be of type int, not {str(type(b[j])}.')
        for i in range(self.__n):
          tmp = self.__items[i * self.__m + a[j]]
          self.__items[i * self.__m + a[j]] = self.__items[i * self.__m + b[j]]
          self.__items[i * self.__m + a[j]] = tmp

  def swap(self, first: Union[tuple, list, int], second: Union[tuple, list, int]):
    '''
    Swaps two values in the matrix
    In:
      first     a list or tuple of the first item (for vector an int is enough)
      second    a list or tuple of the second item (for vector an int is enough)
    '''
    # check for ints
    if type(first) is int:
      if self.__n == 1:
        first = [1, first]
      elif self.__m == 1:
        first = [first, 1]
      else:
        raise TypeError(f'{str(self.__class__)}.swap: supplied first index is an int and Matrix is not a vector.')
    if type(second) is int:
      if self.__n == 1:
        second = [1, second]
      elif self.__m == 1:
        second = [second, 1]
      else:
        raise TypeError(f'{str(self.__class__)}.swap: supplied second index is an int and Matrix is not a vector.')
    # check bounds
    if first[0] < 0 or first[0] >= self.__n:
      raise ValueError(f'{str(self.__class__)}.swap: first row index ({str(first[0])}) is out of bounds (0, {str(self.__n - 1)}).')
    if first[1] < 0 or first[1] >= self.__m:
      raise ValueError(f'{str(self.__class__)}.swap: first col index ({str(first[1])}) is out of bounds (0, {str(self.__m - 1)}).')
    if second[0] < 0 or second[0] >= self.__n:
      raise ValueError(f'{str(self.__class__)}.swap: second row index ({str(second[0])}) is out of bounds (0, {str(self.__n - 1)}).')
    if second[1] < 0 or second[1] >= self.__m:
      raise ValueError(f'{str(self.__class__)}.swap: second col index ({str(second[1])}) is out of bounds (0, {str(self.__m - 1)}).')

    # swap the values
    tmp = self.__items[first[0] * self.__m + first[1]]
    self.__items[first[0] * self.__m + first[1]] = self.__items[second[0] * self.__m + second[1]]
    self.__items[second[0] * self.__m + second[1]] = tmp

  def row(self, r: int) -> Matrix:
    '''
    Returns a row as a new Matrix
    In:
      r    row index
    Out:
      row  a row Matrix
    '''
    # check bounds
    if type(r) is not int:
      raise TypeError(f'{str(self.__class__)}.row: row index must be of type int, not {str(type(r))}.')
    if r < 0 or r >= self.__n:
      raise ValueError(f'{str(self.__class__)}.row: row index ({str(r)}) out of bounds (0, {str(self.__n - 1)}).')
    return self.__class__(self.__items[r * self.__m: r * self.__m + self.__n], size=(1, self.__m), mytype=self.__type)

  def col(self, c: int) -> Matrix:
    '''
    Returns a column as a new Matrix
    In:
      c    column index
    Out:
      col  a column Matrix
    '''
    # check bounds
    if type(c) is not int:
      raise TypeError(f'{str(self.__class__)}.col: col index must be of type int, not {str(type(c))}.')
    if c < 0 or c >= self.__m:
      raise ValueError(f'{str(self.__class__)}.col: col index ({str(c)}) out of bounds (0, {str(self.__m - 1)}).')

    column = list()
    for i in range(self.__n):
      column.append([self.__items[i * self.__m + c]])
    return self.__class__(column, size=(self.__n, 1), mytype=self.__type)

  def transpose(self):
    '''
    Transposes the Matrix in place
    '''
    items = list()
    for j in range(self.__m):
      for i in range(self.__n):
        items.append(self[i,j])
    self.__items = items
    tmp = self.__n
    self.__n = self.__m
    self.__m = tmp

  @property
  def T(self) -> Matrix:
    '''
    returns the transpose of the matrix
    '''
    B = deepcopy(self)
    B.transpose()
    return B

