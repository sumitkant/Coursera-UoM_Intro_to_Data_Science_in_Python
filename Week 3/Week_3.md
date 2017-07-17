
---

_You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._

---

# Merging Dataframes



```python
import pandas as pd

df = pd.DataFrame([{'Name': 'Chris', 'Item Purchased': 'Sponge', 'Cost': 22.50},
                   {'Name': 'Kevyn', 'Item Purchased': 'Kitty Litter', 'Cost': 2.50},
                   {'Name': 'Filip', 'Item Purchased': 'Spoon', 'Cost': 5.00}],
                  index=['Store 1', 'Store 1', 'Store 2'])
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cost</th>
      <th>Item Purchased</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Store 1</th>
      <td>22.5</td>
      <td>Sponge</td>
      <td>Chris</td>
    </tr>
    <tr>
      <th>Store 1</th>
      <td>2.5</td>
      <td>Kitty Litter</td>
      <td>Kevyn</td>
    </tr>
    <tr>
      <th>Store 2</th>
      <td>5.0</td>
      <td>Spoon</td>
      <td>Filip</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Date'] = ['December 1', 'January 1', 'mid-May']
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cost</th>
      <th>Item Purchased</th>
      <th>Name</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Store 1</th>
      <td>22.5</td>
      <td>Sponge</td>
      <td>Chris</td>
      <td>December 1</td>
    </tr>
    <tr>
      <th>Store 1</th>
      <td>2.5</td>
      <td>Kitty Litter</td>
      <td>Kevyn</td>
      <td>January 1</td>
    </tr>
    <tr>
      <th>Store 2</th>
      <td>5.0</td>
      <td>Spoon</td>
      <td>Filip</td>
      <td>mid-May</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Delivered'] = True
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cost</th>
      <th>Item Purchased</th>
      <th>Name</th>
      <th>Date</th>
      <th>Delivered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Store 1</th>
      <td>22.5</td>
      <td>Sponge</td>
      <td>Chris</td>
      <td>December 1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>Store 1</th>
      <td>2.5</td>
      <td>Kitty Litter</td>
      <td>Kevyn</td>
      <td>January 1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>Store 2</th>
      <td>5.0</td>
      <td>Spoon</td>
      <td>Filip</td>
      <td>mid-May</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Feedback'] = ['Positive', None, 'Negative']
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cost</th>
      <th>Item Purchased</th>
      <th>Name</th>
      <th>Date</th>
      <th>Delivered</th>
      <th>Feedback</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Store 1</th>
      <td>22.5</td>
      <td>Sponge</td>
      <td>Chris</td>
      <td>December 1</td>
      <td>True</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>Store 1</th>
      <td>2.5</td>
      <td>Kitty Litter</td>
      <td>Kevyn</td>
      <td>January 1</td>
      <td>True</td>
      <td>None</td>
    </tr>
    <tr>
      <th>Store 2</th>
      <td>5.0</td>
      <td>Spoon</td>
      <td>Filip</td>
      <td>mid-May</td>
      <td>True</td>
      <td>Negative</td>
    </tr>
  </tbody>
</table>
</div>




```python
adf = df.reset_index()
adf['Date'] = pd.Series({0: 'December 1', 2: 'mid-May'})
adf
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Cost</th>
      <th>Item Purchased</th>
      <th>Name</th>
      <th>Date</th>
      <th>Delivered</th>
      <th>Feedback</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Store 1</td>
      <td>22.5</td>
      <td>Sponge</td>
      <td>Chris</td>
      <td>December 1</td>
      <td>True</td>
      <td>Positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Store 1</td>
      <td>2.5</td>
      <td>Kitty Litter</td>
      <td>Kevyn</td>
      <td>NaN</td>
      <td>True</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Store 2</td>
      <td>5.0</td>
      <td>Spoon</td>
      <td>Filip</td>
      <td>mid-May</td>
      <td>True</td>
      <td>Negative</td>
    </tr>
  </tbody>
</table>
</div>




```python
staff_df = pd.DataFrame([{'Name': 'Kelly', 'Role': 'Director of HR'},
                         {'Name': 'Sally', 'Role': 'Course liasion'},
                         {'Name': 'James', 'Role': 'Grader'}])
staff_df = staff_df.set_index('Name')
student_df = pd.DataFrame([{'Name': 'James', 'School': 'Business'},
                           {'Name': 'Mike', 'School': 'Law'},
                           {'Name': 'Sally', 'School': 'Engineering'}])
student_df = student_df.set_index('Name')
print(staff_df.head())
print()
print(student_df.head())
```

                     Role
    Name                 
    Kelly  Director of HR
    Sally  Course liasion
    James          Grader
    
                School
    Name              
    James     Business
    Mike           Law
    Sally  Engineering



```python
pd.merge(staff_df, student_df, how='outer', left_index=True, right_index=True)
```


```python
pd.merge(staff_df, student_df, how='inner', left_index=True, right_index=True)
```


```python
pd.merge(staff_df, student_df, how='left', left_index=True, right_index=True)
```


```python
pd.merge(staff_df, student_df, how='right', left_index=True, right_index=True)
```


```python
staff_df = staff_df.reset_index()
student_df = student_df.reset_index()
pd.merge(staff_df, student_df, how='left', left_on='Name', right_on='Name')
```


```python
staff_df = pd.DataFrame([{'Name': 'Kelly', 'Role': 'Director of HR', 'Location': 'State Street'},
                         {'Name': 'Sally', 'Role': 'Course liasion', 'Location': 'Washington Avenue'},
                         {'Name': 'James', 'Role': 'Grader', 'Location': 'Washington Avenue'}])
student_df = pd.DataFrame([{'Name': 'James', 'School': 'Business', 'Location': '1024 Billiard Avenue'},
                           {'Name': 'Mike', 'School': 'Law', 'Location': 'Fraternity House #22'},
                           {'Name': 'Sally', 'School': 'Engineering', 'Location': '512 Wilson Crescent'}])
pd.merge(staff_df, student_df, how='left', left_on='Name', right_on='Name')
```


```python
staff_df = pd.DataFrame([{'First Name': 'Kelly', 'Last Name': 'Desjardins', 'Role': 'Director of HR'},
                         {'First Name': 'Sally', 'Last Name': 'Brooks', 'Role': 'Course liasion'},
                         {'First Name': 'James', 'Last Name': 'Wilde', 'Role': 'Grader'}])
student_df = pd.DataFrame([{'First Name': 'James', 'Last Name': 'Hammond', 'School': 'Business'},
                           {'First Name': 'Mike', 'Last Name': 'Smith', 'School': 'Law'},
                           {'First Name': 'Sally', 'Last Name': 'Brooks', 'School': 'Engineering'}])
staff_df
student_df
pd.merge(staff_df, student_df, how='inner', left_on=['First Name','Last Name'], right_on=['First Name','Last Name'])
```

# Idiomatic Pandas: Making Code Pandorable


```python
import pandas as pd
df = pd.read_csv('census.csv')
df
```


```python
(df.where(df['SUMLEV']==50)
    .dropna()
    .set_index(['STNAME','CTYNAME'])
    .rename(columns={'ESTIMATESBASE2010': 'Estimates Base 2010'}))
```


```python
df = df[df['SUMLEV']==50]
df.set_index(['STNAME','CTYNAME'], inplace=True)
df.rename(columns={'ESTIMATESBASE2010': 'Estimates Base 2010'})
```


```python
import numpy as np
def min_max(row):
    data = row[['POPESTIMATE2010',
                'POPESTIMATE2011',
                'POPESTIMATE2012',
                'POPESTIMATE2013',
                'POPESTIMATE2014',
                'POPESTIMATE2015']]
    return pd.Series({'min': np.min(data), 'max': np.max(data)})
```


```python
df.apply(min_max, axis=1)
```


```python
import numpy as np
def min_max(row):
    data = row[['POPESTIMATE2010',
                'POPESTIMATE2011',
                'POPESTIMATE2012',
                'POPESTIMATE2013',
                'POPESTIMATE2014',
                'POPESTIMATE2015']]
    row['max'] = np.max(data)
    row['min'] = np.min(data)
    return row
df.apply(min_max, axis=1)
```


```python
rows = ['POPESTIMATE2010',
        'POPESTIMATE2011',
        'POPESTIMATE2012',
        'POPESTIMATE2013',
        'POPESTIMATE2014',
        'POPESTIMATE2015']
df.apply(lambda x: np.max(x[rows]), axis=1)
```

# Group by


```python
import pandas as pd
import numpy as np
df = pd.read_csv('census.csv')
df = df[df['SUMLEV']==50]
df
```


```python
%%timeit -n 10
for state in df['STNAME'].unique():
    avg = np.average(df.where(df['STNAME']==state).dropna()['CENSUS2010POP'])
    print('Counties in state ' + state + ' have an average population of ' + str(avg))
```


```python
%%timeit -n 10
for group, frame in df.groupby('STNAME'):
    avg = np.average(frame['CENSUS2010POP'])
    print('Counties in state ' + group + ' have an average population of ' + str(avg))
```


```python
df.head()
```


```python
df = df.set_index('STNAME')

def fun(item):
    if item[0]<'M':
        return 0
    if item[0]<'Q':
        return 1
    return 2

for group, frame in df.groupby(fun):
    print('There are ' + str(len(frame)) + ' records in group ' + str(group) + ' for processing.')

```


```python
df = pd.read_csv('census.csv')
df = df[df['SUMLEV']==50]
```


```python
df.groupby('STNAME').agg({'CENSUS2010POP': np.average})
```


```python
print(type(df.groupby(level=0)['POPESTIMATE2010','POPESTIMATE2011']))
print(type(df.groupby(level=0)['POPESTIMATE2010']))
```


```python
(df.set_index('STNAME').groupby(level=0)['CENSUS2010POP']
    .agg({'avg': np.average, 'sum': np.sum}))
```


```python
(df.set_index('STNAME').groupby(level=0)['POPESTIMATE2010','POPESTIMATE2011']
    .agg({'avg': np.average, 'sum': np.sum}))
```


```python
(df.set_index('STNAME').groupby(level=0)['POPESTIMATE2010','POPESTIMATE2011']
    .agg({'POPESTIMATE2010': np.average, 'POPESTIMATE2011': np.sum}))
```

# Scales


```python
df = pd.DataFrame(['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D'],
                  index=['excellent', 'excellent', 'excellent', 'good', 'good', 'good', 'ok', 'ok', 'ok', 'poor', 'poor'])
df.rename(columns={0: 'Grades'}, inplace=True)
df
```


```python
df['Grades'].astype('category').head()
```


```python
grades = df['Grades'].astype('category',
                             categories=['D', 'D+', 'C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+'],
                             ordered=True)
grades.head()
```


```python
grades > 'C'
```


```python
df = pd.read_csv('census.csv')
df = df[df['SUMLEV']==50]
df = df.set_index('STNAME').groupby(level=0)['CENSUS2010POP'].agg({'avg': np.average})
pd.cut(df['avg'],10)
```

# Pivot Tables


```python
#http://open.canada.ca/data/en/dataset/98f1a129-f628-4ce4-b24d-6f16bf24dd64
df = pd.read_csv('cars.csv')
```


```python
df.head()
```


```python
df.pivot_table(values='(kW)', index='YEAR', columns='Make', aggfunc=np.mean)
```


```python
df.pivot_table(values='(kW)', index='YEAR', columns='Make', aggfunc=[np.mean,np.min], margins=True)
```

# Date Functionality in Pandas


```python
import pandas as pd
import numpy as np
```

### Timestamp


```python
pd.Timestamp('9/1/2016 10:05AM')
```




    Timestamp('2016-09-01 10:05:00')



### Period


```python
pd.Period('1/2016')
```




    Period('2016-01', 'M')




```python
pd.Period('3/5/2016')
```




    Period('2016-03-05', 'D')



### DatetimeIndex


```python
t1 = pd.Series(list('abc'), [pd.Timestamp('2016-09-01'), pd.Timestamp('2016-09-02'), pd.Timestamp('2016-09-03')])
t1
```




    2016-09-01    a
    2016-09-02    b
    2016-09-03    c
    dtype: object




```python
type(t1.index)
```




    pandas.tseries.index.DatetimeIndex



### PeriodIndex


```python
t2 = pd.Series(list('def'), [pd.Period('2016-09'), pd.Period('2016-10'), pd.Period('2016-11')])
t2
```




    2016-09    d
    2016-10    e
    2016-11    f
    Freq: M, dtype: object




```python
type(t2.index)
```




    pandas.tseries.period.PeriodIndex



### Converting to Datetime


```python
d1 = ['2 June 2013', 'Aug 29, 2014', '2015-06-26', '7/12/16']
ts3 = pd.DataFrame(np.random.randint(10, 100, (4,2)), index=d1, columns=list('ab'))
ts3
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2 June 2013</th>
      <td>16</td>
      <td>46</td>
    </tr>
    <tr>
      <th>Aug 29, 2014</th>
      <td>14</td>
      <td>66</td>
    </tr>
    <tr>
      <th>2015-06-26</th>
      <td>59</td>
      <td>99</td>
    </tr>
    <tr>
      <th>7/12/16</th>
      <td>27</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>




```python
ts3.index = pd.to_datetime(ts3.index)
ts3
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a</th>
      <th>b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-06-02</th>
      <td>16</td>
      <td>46</td>
    </tr>
    <tr>
      <th>2014-08-29</th>
      <td>14</td>
      <td>66</td>
    </tr>
    <tr>
      <th>2015-06-26</th>
      <td>59</td>
      <td>99</td>
    </tr>
    <tr>
      <th>2016-07-12</th>
      <td>27</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.to_datetime('4.7.12', dayfirst=True)
```




    Timestamp('2012-07-04 00:00:00')



### Timedeltas


```python
pd.Timestamp('9/3/2016')-pd.Timestamp('9/1/2016')
```




    Timedelta('2 days 00:00:00')




```python
pd.Timestamp('9/2/2016 8:10AM') + pd.Timedelta('12D 3H')
```




    Timestamp('2016-09-14 11:10:00')



### Working with Dates in a Dataframe


```python
dates = pd.date_range('10-01-2016', periods=9, freq='2W-SUN')
dates
```




    DatetimeIndex(['2016-10-02', '2016-10-16', '2016-10-30', '2016-11-13',
                   '2016-11-27', '2016-12-11', '2016-12-25', '2017-01-08',
                   '2017-01-22'],
                  dtype='datetime64[ns]', freq='2W-SUN')




```python
df = pd.DataFrame({'Count 1': 100 + np.random.randint(-5, 10, 9).cumsum(),
                  'Count 2': 120 + np.random.randint(-5, 10, 9)}, index=dates)
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Count 1</th>
      <th>Count 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-10-02</th>
      <td>104</td>
      <td>125</td>
    </tr>
    <tr>
      <th>2016-10-16</th>
      <td>109</td>
      <td>122</td>
    </tr>
    <tr>
      <th>2016-10-30</th>
      <td>111</td>
      <td>127</td>
    </tr>
    <tr>
      <th>2016-11-13</th>
      <td>117</td>
      <td>126</td>
    </tr>
    <tr>
      <th>2016-11-27</th>
      <td>114</td>
      <td>126</td>
    </tr>
    <tr>
      <th>2016-12-11</th>
      <td>109</td>
      <td>121</td>
    </tr>
    <tr>
      <th>2016-12-25</th>
      <td>105</td>
      <td>126</td>
    </tr>
    <tr>
      <th>2017-01-08</th>
      <td>105</td>
      <td>125</td>
    </tr>
    <tr>
      <th>2017-01-22</th>
      <td>101</td>
      <td>123</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.index.weekday_name
```




    array(['Sunday', 'Sunday', 'Sunday', 'Sunday', 'Sunday', 'Sunday',
           'Sunday', 'Sunday', 'Sunday'], dtype=object)




```python
df.diff()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Count 1</th>
      <th>Count 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-10-02</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2016-10-16</th>
      <td>5.0</td>
      <td>-3.0</td>
    </tr>
    <tr>
      <th>2016-10-30</th>
      <td>2.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2016-11-13</th>
      <td>6.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>2016-11-27</th>
      <td>-3.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2016-12-11</th>
      <td>-5.0</td>
      <td>-5.0</td>
    </tr>
    <tr>
      <th>2016-12-25</th>
      <td>-4.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2017-01-08</th>
      <td>0.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>2017-01-22</th>
      <td>-4.0</td>
      <td>-2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.resample('M').mean()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Count 1</th>
      <th>Count 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-10-31</th>
      <td>108.0</td>
      <td>124.666667</td>
    </tr>
    <tr>
      <th>2016-11-30</th>
      <td>115.5</td>
      <td>126.000000</td>
    </tr>
    <tr>
      <th>2016-12-31</th>
      <td>107.0</td>
      <td>123.500000</td>
    </tr>
    <tr>
      <th>2017-01-31</th>
      <td>103.0</td>
      <td>124.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['2017']
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Count 1</th>
      <th>Count 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-01-08</th>
      <td>105</td>
      <td>125</td>
    </tr>
    <tr>
      <th>2017-01-22</th>
      <td>101</td>
      <td>123</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['2016-12']
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Count 1</th>
      <th>Count 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-12-11</th>
      <td>109</td>
      <td>121</td>
    </tr>
    <tr>
      <th>2016-12-25</th>
      <td>105</td>
      <td>126</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['2016-12':]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Count 1</th>
      <th>Count 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-12-11</th>
      <td>109</td>
      <td>121</td>
    </tr>
    <tr>
      <th>2016-12-25</th>
      <td>105</td>
      <td>126</td>
    </tr>
    <tr>
      <th>2017-01-08</th>
      <td>105</td>
      <td>125</td>
    </tr>
    <tr>
      <th>2017-01-22</th>
      <td>101</td>
      <td>123</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.asfreq('W', method='ffill')
```


```python
import matplotlib.pyplot as plt
%matplotlib inline

df.plot()
```
