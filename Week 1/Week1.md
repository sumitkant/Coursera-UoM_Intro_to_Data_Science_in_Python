
---

_You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._

---

# The Python Programming Language: Functions


```python
x = 1
y = 3
x + y
```




    4




```python
x
```

<br>
`add_numbers` is a function that takes two numbers and adds them together.


```python
def add_numbers(x, y):
    return x + y

add_numbers(1, 2)
```




    3



<br>
`add_numbers` updated to take an optional 3rd parameter. Using `print` allows printing of multiple expressions within a single cell.


```python
def add_numbers(x,y,z=None):
    if (z==None):
        return x+y
    else:
        return x+y+z

print(add_numbers(1, 2))
print(add_numbers(1, 2, 3))
```

    3
    6


<br>
`add_numbers` updated to take an optional flag parameter.


```python
def add_numbers(x, y, z=None, flag=False):
    if (flag):
        print('Flag is true!')
    if (z==None):
        return x + y
    else:
        return x + y + z
    
print(add_numbers(1, 2, flag=True))
```

<br>
Assign function `add_numbers` to variable `a`.


```python
def add_numbers(x,y):
    return x+y

a = add_numbers
a(1,2)
```

<br>
# The Python Programming Language: Types and Sequences

<br>
Use `type` to return the object's type.


```python
type('This is a string')
```


```python
type(None)
```


```python
type(1)
```


```python
type(1.0)
```


```python
type(add_numbers)
```

<br>
Tuples are an immutable data structure (cannot be altered).


```python
x = (1, 'a', 2, 'b')
type(x)
```




    tuple



<br>
Lists are a mutable data structure.


```python
x = [1, 'a', 2, 'b']
type(x)
```




    list



<br>
Use `append` to append an object to a list.


```python
x.append(3.3)
print(x)
```

    [1, 'a', 2, 'b', 3.3]


<br>
This is an example of how to loop through each item in the list.


```python
for item in x:
    print(item)
```

    1
    a
    2
    b
    3.3


<br>
Or using the indexing operator:


```python
i=0
while( i != len(x) ):
    print(x[i])
    i = i + 1
```

    1
    a
    2
    b
    3.3


<br>
Use `+` to concatenate lists.


```python
[1,2] + [3,4]
```




    [1, 2, 3, 4]



<br>
Use `*` to repeat lists.


```python
[1]*3
```




    [1, 1, 1]



<br>
Use the `in` operator to check if something is inside a list.


```python
1 in [1, 2, 3]
```




    True



<br>
Now let's look at strings. Use bracket notation to slice a string.


```python
x = 'This is a string'
print(x[0]) #first character
print(x[0:1]) #first character, but we have explicitly set the end character
print(x[0:2]) #first two characters

```

    T
    T
    Th


<br>
This will return the last element of the string.


```python
x[-1]
```




    'g'



<br>
This will return the slice starting from the 4th element from the end and stopping before the 2nd element from the end.


```python
x[-4:-2]
```




    'ri'



<br>
This is a slice from the beginning of the string and stopping before the 3rd element.


```python
x[:3]
```




    'Thi'



<br>
And this is a slice starting from the 3rd element of the string and going all the way to the end.


```python
x[3:]
```




    's is a string'




```python
firstname = 'Christopher'
lastname = 'Brooks'

print(firstname + ' ' + lastname)
print(firstname*3)
print('Chris' in firstname)

```

<br>
`split` returns a list of all the words in a string, or a list split on a specific character.


```python
firstname = 'Christopher Arthur Hansen Brooks'.split(' ')[0] # [0] selects the first element of the list
lastname = 'Christopher Arthur Hansen Brooks'.split(' ')[-1] # [-1] selects the last element of the list
print(firstname)
print(lastname)
```

    Christopher
    Brooks


<br>
Make sure you convert objects to strings before concatenating.


```python
'Chris' + 2
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-16-1623ac76de6e> in <module>()
    ----> 1 'Chris' + 2
    

    TypeError: Can't convert 'int' object to str implicitly



```python
'Chris' + str(2)
```




    'Chris2'



<br>
Dictionaries associate keys with values.


```python
x = {'Christopher Brooks': 'brooksch@umich.edu', 'Bill Gates': 'billg@microsoft.com'}
x['Christopher Brooks'] # Retrieve a value by using the indexing operator

```




    'brooksch@umich.edu'




```python
x['Kevyn Collins-Thompson'] = None
x['Kevyn Collins-Thompson']
```

<br>
Iterate over all of the keys:


```python
for name in x:
    print(x[name])
```

    brooksch@umich.edu
    billg@microsoft.com
    None


<br>
Iterate over all of the values:


```python
for email in x.values():
    print(email)
```

    brooksch@umich.edu
    billg@microsoft.com
    None


<br>
Iterate over all of the items in the list:


```python
for name, email in x.items():
    print(name)
    print(email)
```

    Christopher Brooks
    brooksch@umich.edu
    Bill Gates
    billg@microsoft.com
    Kevyn Collins-Thompson
    None


<br>
You can unpack a sequence into different variables:


```python
x = ('Christopher', 'Brooks', 'brooksch@umich.edu')
fname, lname, email = x
```


```python
fname
```




    'Christopher'




```python
lname
```




    'Brooks'



<br>
Make sure the number of values you are unpacking matches the number of variables being assigned.


```python
x = ('Christopher', 'Brooks', 'brooksch@umich.edu', 'Ann Arbor')
fname, lname, email = x
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-30-9ce70064f53e> in <module>()
          1 x = ('Christopher', 'Brooks', 'brooksch@umich.edu', 'Ann Arbor')
    ----> 2 fname, lname, email = x
    

    ValueError: too many values to unpack (expected 3)


<br>
# The Python Programming Language: More on Strings


```python
print('Chris' + 2)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-31-82ccfdd3d5d3> in <module>()
    ----> 1 print('Chris' + 2)
    

    TypeError: Can't convert 'int' object to str implicitly



```python
print('Chris' + str(2))
```

    Chris2


<br>
Python has a built in method for convenient string formatting.


```python
sales_record = {
'price': 3.24,
'num_items': 4,
'person': 'Chris'}

sales_statement = '{} bought {} item(s) at a price of {} each for a total of {}'

print(sales_statement.format(sales_record['person'],
                             sales_record['num_items'],
                             sales_record['price'],
                             sales_record['num_items']*sales_record['price']))

```

    Chris bought 4 item(s) at a price of 3.24 each for a total of 12.96


<br>
# Reading and Writing CSV files

<br>
Let's import our datafile mpg.csv, which contains fuel economy data for 234 cars.

* mpg : miles per gallon
* class : car classification
* cty : city mpg
* cyl : # of cylinders
* displ : engine displacement in liters
* drv : f = front-wheel drive, r = rear wheel drive, 4 = 4wd
* fl : fuel (e = ethanol E85, d = diesel, r = regular, p = premium, c = CNG)
* hwy : highway mpg
* manufacturer : automobile manufacturer
* model : model of car
* trans : type of transmission
* year : model year


```python
import csv

%precision 2 

with open('mpg.csv') as csvfile:
    mpg = list(csv.DictReader(csvfile))
    
mpg[:3] # The first three dictionaries in our list.
```




    [{'': '1',
      'class': 'compact',
      'cty': '18',
      'cyl': '4',
      'displ': '1.8',
      'drv': 'f',
      'fl': 'p',
      'hwy': '29',
      'manufacturer': 'audi',
      'model': 'a4',
      'trans': 'auto(l5)',
      'year': '1999'},
     {'': '2',
      'class': 'compact',
      'cty': '21',
      'cyl': '4',
      'displ': '1.8',
      'drv': 'f',
      'fl': 'p',
      'hwy': '29',
      'manufacturer': 'audi',
      'model': 'a4',
      'trans': 'manual(m5)',
      'year': '1999'},
     {'': '3',
      'class': 'compact',
      'cty': '20',
      'cyl': '4',
      'displ': '2',
      'drv': 'f',
      'fl': 'p',
      'hwy': '31',
      'manufacturer': 'audi',
      'model': 'a4',
      'trans': 'manual(m6)',
      'year': '2008'}]



<br>
`csv.Dictreader` has read in each row of our csv file as a dictionary. `len` shows that our list is comprised of 234 dictionaries.


```python
len(mpg)
```

<br>
`keys` gives us the column names of our csv.


```python
mpg[0].keys()
```




    dict_keys(['', 'cyl', 'fl', 'class', 'hwy', 'model', 'trans', 'drv', 'year', 'displ', 'manufacturer', 'cty'])



<br>
This is how to find the average cty fuel economy across all cars. All values in the dictionaries are strings, so we need to convert to float.


```python
sum(float(d['cty']) for d in mpg) / len(mpg)
```




    16.86



<br>
Similarly this is how to find the average hwy fuel economy across all cars.


```python
sum(float(d['hwy']) for d in mpg) / len(mpg)
```




    23.44



<br>
Use `set` to return the unique values for the number of cylinders the cars in our dataset have.


```python
cylinders = set(d['cyl'] for d in mpg)
cylinders
```




    {'4', '5', '6', '8'}



<br>
Here's a more complex example where we are grouping the cars by number of cylinder, and finding the average cty mpg for each group.


```python
CtyMpgByCyl = []

for c in cylinders: # iterate over all the cylinder levels
    summpg = 0
    cyltypecount = 0
    for d in mpg: # iterate over all dictionaries
        if d['cyl'] == c: # if the cylinder level type matches,
            summpg += float(d['cty']) # add the cty mpg
            cyltypecount += 1 # increment the count
    CtyMpgByCyl.append((c, summpg / cyltypecount)) # append the tuple ('cylinder', 'avg mpg')

CtyMpgByCyl.sort(key=lambda x: x[0])
CtyMpgByCyl
```

<br>
Use `set` to return the unique values for the class types in our dataset.


```python
vehicleclass = set(d['class'] for d in mpg) # what are the class types
vehicleclass
```

<br>
And here's an example of how to find the average hwy mpg for each class of vehicle in our dataset.


```python
HwyMpgByClass = []

for t in vehicleclass: # iterate over all the vehicle classes
    summpg = 0
    vclasscount = 0
    for d in mpg: # iterate over all dictionaries
        if d['class'] == t: # if the cylinder amount type matches,
            summpg += float(d['hwy']) # add the hwy mpg
            vclasscount += 1 # increment the count
    HwyMpgByClass.append((t, summpg / vclasscount)) # append the tuple ('class', 'avg mpg')

HwyMpgByClass.sort(key=lambda x: x[1])
HwyMpgByClass
```

<br>
# The Python Programming Language: Dates and Times


```python
import datetime as dt
import time as tm
```

<br>
`time` returns the current time in seconds since the Epoch. (January 1st, 1970)


```python
tm.time()
```




    1497492906.57



<br>
Convert the timestamp to datetime.


```python
dtnow = dt.datetime.fromtimestamp(tm.time())
dtnow
```




    datetime.datetime(2017, 6, 15, 2, 15, 13, 821108)



<br>
Handy datetime attributes:


```python
dtnow.year, dtnow.month, dtnow.day, dtnow.hour, dtnow.minute, dtnow.second # get year, month, day, etc.from a datetime
```




    (2017, 6, 15, 2, 15, 13)



<br>
`timedelta` is a duration expressing the difference between two dates.


```python
delta = dt.timedelta(days = 100) # create a timedelta of 100 days
delta
```




    datetime.timedelta(100)



<br>
`date.today` returns the current local date.


```python
today = dt.date.today()
```


```python
today - delta # the date 100 days ago
```




    datetime.date(2017, 3, 7)




```python
today > today-delta # compare dates
```




    True



<br>
# The Python Programming Language: Objects and map()

<br>
An example of a class in python:


```python
class Person:
    department = 'School of Information' #a class variable

    def set_name(self, new_name): #a method
        self.name = new_name
    def set_location(self, new_location):
        self.location = new_location
```


```python
person = Person()
person.set_name('Christopher Brooks')
person.set_location('Ann Arbor, MI, USA')
print('{} live in {} and works in the department {}'.format(person.name, person.location, person.department))
```

<br>
Here's an example of mapping the `min` function between two lists.


```python
store1 = [10.00, 11.00, 12.34, 2.34]
store2 = [9.00, 11.10, 12.34, 2.01]
cheapest = map(min, store1, store2)
cheapest
```

<br>
Now let's iterate through the map object to see the values.


```python
for item in cheapest:
    print(item)
```


```python
type(lambda x: x+1)
```




    function



<br>
# The Python Programming Language: Lambda and List Comprehensions

<br>
Here's an example of lambda that takes in three parameters and adds the first two.


```python
my_function = lambda a, b, c : a + b
```


```python
my_function(1, 2, 3)
```

<br>
Let's iterate from 0 to 999 and return the even numbers.


```python
my_list = []
for number in range(0, 1000):
    if number % 2 == 0:
        my_list.append(number)
my_list
```

<br>
Now the same thing but with list comprehension.


```python
my_list = [number for number in range(0,1000) if number % 2 == 0]
my_list
```

<br>
# The Python Programming Language: Numerical Python (NumPy)


```python
import numpy as np
```

<br>
## Creating Arrays

Create a list and convert it to a numpy array


```python
mylist = [1, 2, 3]
x = np.array(mylist)
x
```




    array([1, 2, 3])



<br>
Or just pass in a list directly


```python
y = np.array([4, 5, 6])
y
```




    array([4, 5, 6])



<br>
Pass in a list of lists to create a multidimensional array.


```python
m = np.array([[7, 8, 9], [10, 11, 12]])
m
```




    array([[ 7,  8,  9],
           [10, 11, 12]])



<br>
Use the shape method to find the dimensions of the array. (rows, columns)


```python
m.shape
```




    (2, 3)



<br>
`arange` returns evenly spaced values within a given interval.


```python
n = np.arange(0, 30, 2) # start at 0 count up by 2, stop before 30
n
```




    array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])



<br>
`reshape` returns an array with the same data with a new shape.


```python
n = n.reshape(3, 5) # reshape array to be 3x5
n
```




    array([[ 0,  2,  4,  6,  8],
           [10, 12, 14, 16, 18],
           [20, 22, 24, 26, 28]])



<br>
`linspace` returns evenly spaced numbers over a specified interval.


```python
o = np.linspace(0, 4, 9) # return 9 evenly spaced values from 0 to 4
o
```




    array([ 0. ,  0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ])



<br>
`resize` changes the shape and size of array in-place.


```python
o.resize(3, 3)
o
```




    array([[ 0. ,  0.5,  1. ],
           [ 1.5,  2. ,  2.5],
           [ 3. ,  3.5,  4. ]])



<br>
`ones` returns a new array of given shape and type, filled with ones.


```python
np.ones((3, 2))
```




    array([[ 1.,  1.],
           [ 1.,  1.],
           [ 1.,  1.]])



<br>
`zeros` returns a new array of given shape and type, filled with zeros.


```python
np.zeros((2, 3))
```




    array([[ 0.,  0.,  0.],
           [ 0.,  0.,  0.]])



<br>
`eye` returns a 2-D array with ones on the diagonal and zeros elsewhere.


```python
np.eye(3)
```




    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])



<br>
`diag` extracts a diagonal or constructs a diagonal array.


```python
np.diag(y)
```




    array([[4, 0, 0],
           [0, 5, 0],
           [0, 0, 6]])



<br>
Create an array using repeating list (or see `np.tile`)


```python
np.array([1, 2, 3] * 3)
```




    array([1, 2, 3, 1, 2, 3, 1, 2, 3])



<br>
Repeat elements of an array using `repeat`.


```python
np.repeat([1, 2, 3], 3)
```




    array([1, 1, 1, 2, 2, 2, 3, 3, 3])



<br>
#### Combining Arrays


```python
p = np.ones([2, 3], int)
p
```




    array([[1, 1, 1],
           [1, 1, 1]])



<br>
Use `vstack` to stack arrays in sequence vertically (row wise).


```python
np.vstack([p, 2*p])
```




    array([[1, 1, 1],
           [1, 1, 1],
           [2, 2, 2],
           [2, 2, 2]])



<br>
Use `hstack` to stack arrays in sequence horizontally (column wise).


```python
np.hstack([p, 2*p])
```




    array([[1, 1, 1, 2, 2, 2],
           [1, 1, 1, 2, 2, 2]])



<br>
## Operations

Use `+`, `-`, `*`, `/` and `**` to perform element wise addition, subtraction, multiplication, division and power.


```python
print(x + y) # elementwise addition     [1 2 3] + [4 5 6] = [5  7  9]
print(x - y) # elementwise subtraction  [1 2 3] - [4 5 6] = [-3 -3 -3]
```

    [5 7 9]
    [-3 -3 -3]



```python
print(x * y) # elementwise multiplication  [1 2 3] * [4 5 6] = [4  10  18]
print(x / y) # elementwise divison         [1 2 3] / [4 5 6] = [0.25  0.4  0.5]
```

    [ 4 10 18]
    [ 0.25  0.4   0.5 ]



```python
print(x**2) # elementwise power  [1 2 3] ^2 =  [1 4 9]
```

    [1 4 9]


<br>
**Dot Product:**  

$ \begin{bmatrix}x_1 \ x_2 \ x_3\end{bmatrix}
\cdot
\begin{bmatrix}y_1 \\ y_2 \\ y_3\end{bmatrix}
= x_1 y_1 + x_2 y_2 + x_3 y_3$


```python
x.dot(y) # dot product  1*4 + 2*5 + 3*6
```




    32




```python
z = np.array([y, y**2])
print(len(z)) # number of rows of array
```

    2


<br>
Let's look at transposing arrays. Transposing permutes the dimensions of the array.


```python
z = np.array([y, y**2])
z
```




    array([[ 4,  5,  6],
           [16, 25, 36]])



<br>
The shape of array `z` is `(2,3)` before transposing.


```python
z.shape
```




    (2, 3)



<br>
Use `.T` to get the transpose.


```python
z.T
```




    array([[ 4, 16],
           [ 5, 25],
           [ 6, 36]])



<br>
The number of rows has swapped with the number of columns.


```python
z.T.shape
```




    (3, 2)



<br>
Use `.dtype` to see the data type of the elements in the array.


```python
z.dtype
```




    dtype('int64')



<br>
Use `.astype` to cast to a specific type.


```python
z = z.astype('f')
z.dtype
```




    dtype('float32')



<br>
## Math Functions

Numpy has many built in math functions that can be performed on arrays.


```python
a = np.array([-4, -2, 1, 3, 5])
```


```python
a.sum()
```




    3




```python
a.max()
```




    5




```python
a.min()
```




    -4




```python
a.mean()
```




    0.60




```python
a.std()
```




    3.26



<br>
`argmax` and `argmin` return the index of the maximum and minimum values in the array.


```python
a.argmax()
```




    4




```python
a.argmin()
```




    0



<br>
## Indexing / Slicing


```python
s = np.arange(13)**2
s
```




    array([  0,   1,   4,   9,  16,  25,  36,  49,  64,  81, 100, 121, 144])



<br>
Use bracket notation to get the value at a specific index. Remember that indexing starts at 0.


```python
s[0], s[4], s[-1]
```




    (0, 16, 144)



<br>
Use `:` to indicate a range. `array[start:stop]`


Leaving `start` or `stop` empty will default to the beginning/end of the array.


```python
s[1:5]
```




    array([ 1,  4,  9, 16])



<br>
Use negatives to count from the back.


```python
s[-4:]
```




    array([ 81, 100, 121, 144])



<br>
A second `:` can be used to indicate step-size. `array[start:stop:stepsize]`

Here we are starting 5th element from the end, and counting backwards by 2 until the beginning of the array is reached.


```python
s[-5::-2]
```




    array([64, 36, 16,  4,  0])



<br>
Let's look at a multidimensional array.


```python
r = np.arange(36)
r.resize((6, 6))
r
```




    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17],
           [18, 19, 20, 21, 22, 23],
           [24, 25, 26, 27, 28, 29],
           [30, 31, 32, 33, 34, 35]])



<br>
Use bracket notation to slice: `array[row, column]`


```python
r[2, 2]
```




    14



<br>
And use : to select a range of rows or columns


```python
r[3, 3:6]
```

<br>
Here we are selecting all the rows up to (and not including) row 2, and all the columns up to (and not including) the last column.


```python
r[:2, :-1]
```

<br>
This is a slice of the last row, and only every other element.


```python
r[-1, ::2]
```

<br>
We can also perform conditional indexing. Here we are selecting values from the array that are greater than 30. (Also see `np.where`)


```python
r[r > 30]
```

<br>
Here we are assigning all values in the array that are greater than 30 to the value of 30.


```python
r[r > 30] = 30
r
```

<br>
## Copying Data

Be careful with copying and modifying arrays in NumPy!


`r2` is a slice of `r`


```python
r2 = r[:3,:3]
r2
```

<br>
Set this slice's values to zero ([:] selects the entire array)


```python
r2[:] = 0
r2
```

<br>
`r` has also been changed!


```python
r
```

<br>
To avoid this, use `r.copy` to create a copy that will not affect the original array


```python
r_copy = r.copy()
r_copy
```

<br>
Now when r_copy is modified, r will not be changed.


```python
r_copy[:] = 10
print(r_copy, '\n')
print(r)
```

<br>
### Iterating Over Arrays

Let's create a new 4 by 3 array of random numbers 0-9.


```python
test = np.random.randint(0, 10, (4,3))
test
```

<br>
Iterate by row:


```python
for row in test:
    print(row)
```

<br>
Iterate by index:


```python
for i in range(len(test)):
    print(test[i])
```

<br>
Iterate by row and index:


```python
for i, row in enumerate(test):
    print('row', i, 'is', row)
```

<br>
Use `zip` to iterate over multiple iterables.


```python
test2 = test**2
test2
```


```python
for i, j in zip(test, test2):
    print(i,'+',j,'=',i+j)
```


```python
[x**2 for x in range(10)]
```




    [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

