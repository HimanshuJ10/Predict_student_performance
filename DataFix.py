import pandas as pd
df1 = pd.read_csv('./DATA/student-mat.csv',sep=';')
df2 = pd.read_csv('./DATA/student-por.csv',sep=';')
m = pd.merge(df1,df2,on=("school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"))
# m.to_csv('./DATA/students.csv')
# df2.to_csv('./DATA/studentpor.csv')
print(df1.columns)