# library
import pandas as pd
import numpy as np
from scipy import stats
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
#read dataset
df_student_mat = pd.read_parquet('./DATA/studentMat.parquet')
df_student_por = pd.read_parquet('./DATA/studentPor.parquet')
#dataset in streamlit
st.header('Student Performance Streamlit Dashboard')
st.divider()
st.title('DataSet')
labelMat, labelPor = st.columns(2)
labelMat.write('### Mat')
labelPor.write('### Por')
column_df_mat, column_df_por = st.columns(2)
column_df_mat.dataframe(df_student_mat)
column_df_por.dataframe(df_student_mat)
# mean and std and min and max for mat
st.divider()
st.title('Mean, STD, MIN, MAX (MAT)')
des_label_m, des_label_p = st.columns(2)
des_label_m.write('### Mat')
des_label_p.write('### Por')
des_m, des_p = st.columns(2)
desc_m = df_student_mat.describe()
des_m.dataframe(desc_m)
des_p.dataframe(desc_m)

st.divider()
# we want just work with int or float not obj, Str
fea_mat_num = df_student_mat.select_dtypes(include='number')
fea_por_num = df_student_por.select_dtypes(include='number')
#Var
st.title("Var")
var_label_m, var_label_p = st.columns(2)
var_label_m.write('### Mat')
var_label_p.write('### Por')
var_m, var_p = st.columns(2)
var_m.dataframe(fea_mat_num.var())
var_p.dataframe(fea_por_num.var())
# range
st.divider()
st.title('Range')
range_label_m, range_label_p = st.columns(2)
range_label_m.write('### Mat')
range_label_p.write('### Por')
range_m, range_p = st.columns(2)
       #range Mat
_range = {}
for i in fea_mat_num.columns:
       _range.update({i:float(np.max(df_student_mat[i]) - np.min(df_student_mat[i]))})
range_student_mat = pd.DataFrame(_range,index=[0])
range_m.dataframe(range_student_mat)
       #range Por
_range = {}
for i in fea_por_num.columns:
       _range.update({i:float(np.max(df_student_por[i]) - np.min(df_student_por[i]))})
range_student_por = pd.DataFrame(_range,index=[0])
range_p.dataframe(range_student_por)

st.divider()
st.title('IQR')
iqr_label_m, iqr_label_p = st.columns(2)
iqr_label_m.write('### Mat')
iqr_label_p.write('### Por')
iqr_m, iqr_p = st.columns(2)
       # IQR Mat
_range = {}
for i in fea_mat_num.columns:
       _range.update({i:stats.iqr(df_student_mat[i])})
iqr_student_mat = pd.DataFrame(_range,index=[0])
iqr_m.dataframe(iqr_student_mat)
       # IQR Por
_range = {}
for i in fea_por_num.columns:
       _range.update({i:stats.iqr(df_student_por[i])})
iqr_student_por = pd.DataFrame(_range,index=[0])
iqr_p.dataframe(iqr_student_por)
st.divider()
st.title('Median')
median_label_m, median_label_p = st.columns(2)
median_label_m.write('### Mat')
median_label_p.write('### Por')
median_m, median_p = st.columns(2)
       #Median Mat
median_student_mat = fea_mat_num.median()
median_m.dataframe(median_student_mat)
       #Median Por
median_student_por = fea_por_num.median()
median_p.dataframe(median_student_por)
st.divider()
st.title('Mod')
mod_label_m, mod_label_p = st.columns(2)
mod_label_m.write('### Mat')
mod_label_p.write('### Por')
mod_m, mod_p = st.columns(2)
       # mode mat
mode_result = stats.mode(fea_mat_num)
mode_student_mat = pd.DataFrame(mode_result,columns=fea_mat_num.columns,index=['mode','count'])
mod_m.dataframe(mode_student_mat)
       # mode por
mode_result = stats.mode(fea_por_num)
mode_student_por = pd.DataFrame(mode_result,columns=fea_por_num.columns,index=['mode','count'])
mod_p.dataframe(mode_student_por)
st.divider()
# correlation
st.title('Correlation')
cor_label_m, cor_label_p = st.columns(2)
cor_label_m.write('### Mat')
cor_label_p.write('### Por')
cor_m, cor_p = st.columns(2)
cor_m.dataframe(fea_mat_num.corr())
cor_p.dataframe(fea_por_num.corr())
st.divider()
st.title('Chart')
st.title('gender grade Mat')
st.scatter_chart(data=df_student_mat,x='sex',y='g3')
mat_cor = fea_mat_num.corr()
st.write('### Mat Correlation chart')
st.line_chart(mat_cor)
st.title('gender grade Por')
st.scatter_chart(data=df_student_por,x='sex',y='g3')
por_cor = fea_por_num.corr()
st.write('### por Correlation chart')
st.line_chart(por_cor)
# Ml linearRegression
       #LinearRegression Mat 
st.divider()
st.title('ML model LinearRegression and Classification for Mat and Por')
st.divider()
st.write('## User Input')
       # Label and input
inp_label_mat_school,inp_label_mat_sex,inp_label_mat_age,inp_label_mat_studytime,inp_label_mat_failures,inp_label_mat_schoolsup= st.columns(6)
inp_mat_school,inp_mat_sex,inp_mat_age,inp_mat_studytime,inp_mat_failures,inp_mat_schoolsup = st.columns(6)
# student's school (GP=0,MS=1)
inp_label_mat_school.write("##### student's school (GP=0,MS=1)")
school = inp_mat_school.number_input('school',key='mat school',min_value=0,max_value=1,step=1)
# student's gender (F=0,M=1)
inp_label_mat_sex.write("##### student's gender (F=0,M=1)")
sex = inp_mat_sex.number_input('Gender',key='mat sex',min_value=0,max_value=1,step=1)
# age [16, 18, 17, 19, 15, 22, 20, 21]
inp_label_mat_age.write("##### age [16, 18, 17, 19, 15, 22, 20, 21]")
age = inp_mat_age.number_input('age',key='mat age',step=1)
# weekly study time [2, 1, 3, 4]
inp_label_mat_studytime.write("##### weekly study time [2, 1, 3, 4]")
studytime = inp_mat_studytime.number_input('studytime',key='mat studytime',step=1)

# number of past class failures [0, 3, 1, 2]
inp_label_mat_failures.write("##### number of past class failures [0, 3, 1, 2]")
failures = inp_mat_failures.number_input('failures',key='mat failures',step=1)
# extra educational support (0=No,1=Yes)
inp_label_mat_schoolsup.write("##### extra educational support (0=No,1=Yes)")
schoolsup = inp_mat_schoolsup.number_input('schoolsup',key='mat schoolsup',min_value=0,max_value=1,step=1)
       # Label and input for Mat
inp_label_mat_famsup,inp_label_mat_paid,inp_label_mat_activities,inp_label_mat_internet,inp_label_mat_romantic,inp_label_mat_freetime= st.columns(6)
inp_mat_famsup,inp_mat_paid,inp_mat_activities,inp_mat_internet,inp_mat_romantic,inp_mat_freetime = st.columns(6)
# family support (0=No,1=Yes)
inp_label_mat_famsup.write("##### family support (0=No,1=Yes)")
famsup = inp_mat_famsup.number_input('famsup',key='mat famsup',min_value=0,max_value=1,step=1)

# extra paid classes within the course subject (0=No,1=Yes)
inp_label_mat_paid.write("##### extra paid classes within the course subject (0=No,1=Yes)")
paid = inp_mat_paid.number_input('paid',key='mat paid',min_value=0,max_value=1,step=1)

# extra-curricular activities (0=No,1=Yes)
inp_label_mat_activities.write("##### extra-curricular activities (0=No,1=Yes)")
activities = inp_mat_activities.number_input('activities',key='mat activities',min_value=0,max_value=1,step=1)

# Internet access at home (0=No,1=Yes)
inp_label_mat_internet.write("##### Internet access at home (0=No,1=Yes)")
internet = inp_mat_internet.number_input('internet',key='mat internet',min_value=0,max_value=1,step=1)

# with a romantic relationship (0=No,1=Yes)
inp_label_mat_romantic.write("##### with a romantic relationship (0=No,1=Yes)")
romantic = inp_mat_romantic.number_input('romantic',key='mat romantic',min_value=0,max_value=1,step=1)

# free time after school [2, 3, 4, 1, 5] (numeric: from 1 -> very low to 5 -> very high)
inp_label_mat_freetime.write("##### free time after school [2, 3, 4, 1, 5] (numeric: from 1 -> very low to 5 -> very high)")
freetime = inp_mat_freetime.number_input('freetime',key='mat freetime',min_value=1,max_value=5,step=1)

       # Label and input
inp_label_mat_goout,inp_label_mat_health,inp_label_mat_absences,inp_label_mat_g1,inp_label_mat_g2= st.columns(5)
inp_mat_goout,inp_mat_health,inp_mat_absences,inp_mat_g1,inp_mat_g2 = st.columns(5)

#going out with friends [3, 5, 4, 2, 1] (numeric: from 1 - very low to 5 - very high)

inp_label_mat_goout.write("##### going out with friends [3, 5, 4, 2, 1] (numeric: from 1 - very low to 5 - very high)")
goout = inp_mat_goout.number_input('goout',key='mat goout',min_value=1,max_value=5,step=1)

# current health status [3, 4, 5, 1, 2](numeric: from 1 - very bad to 5 - very good)
inp_label_mat_health.write("##### current health status [3, 4, 5, 1, 2](numeric: from 1 - very bad to 5 - very good)")
health = inp_mat_health.number_input('health',key='mat health',min_value=1,max_value=5,step=1)

# number of school absences (numeric: from 0 to 93)[ 2,  0, 16,  7, 11, 23,  4, 10,  6, 13,  8, 20, 12,  5, 14, 19,  3,9,  1, 24, 26, 18, 30, 56, 15, 22, 17, 54, 21, 25, 28, 40, 75]
inp_label_mat_absences.write("##### number of school absences (numeric: from 0 to 93)[ 2,  0, 16,  7, 11, 23,  4, 10,  6, 13,  8, 20, 12,  5, 14, 19,  3,9,  1, 24, 26, 18, 30, 56, 15, 22, 17, 54, 21, 25, 28, 40, 75]")
absences = inp_mat_absences.number_input('absences',key='mat absences',min_value=0,max_value=20,step=1)

# first period grade (numeric: from 0 to 20)
inp_label_mat_g1.write("##### first period grade (numeric: from 0 to 20)")
g1 = inp_mat_g1.number_input('g1',key='mat g1',min_value=0,max_value=20,step=1)

# second period grade (numeric: from 0 to 20)
inp_label_mat_g2.write("##### second period grade (numeric: from 0 to 20)")
g2 = inp_mat_g2.number_input('g2',key='mat g2',step=1)
st.divider()
st.write('## Mat Linear Regression')
# train test split
df_st_mat = df_student_mat.copy()
# Encoding
encoder = LabelEncoder()
for i in df_st_mat.select_dtypes(include='object'):
       df_st_mat[i] = encoder.fit_transform(df_st_mat[i])
       df_st_mat_encoded = df_st_mat.drop(i,axis=1)
X = df_st_mat[[
       'school',
       'sex',
       'age',
       'studytime',
       'failures',
       'schoolsup', 
       'famsup', 
       'paid', 
       'activities',
       'internet', 
       'romantic',
       'freetime', 
       'goout',  
       'health', 
       'absences',
       'g1',
       'g2'    
]]
y = df_st_mat['g3']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
# Our ML model (LinearRegression)
model_lin_mat = LinearRegression()
model_lin_mat.fit(X_train,y_train)
acc_lin_mat = model_lin_mat.score(X_test,y_test)
st.write(f'### Accuracy LinearRegression for Mat: {acc_lin_mat:.2f}')

       #predict
pre_lin_mat = model_lin_mat.predict(np.array([school,sex,age,studytime,failures,schoolsup,famsup,paid,activities,internet,romantic,freetime,goout,health,absences,g1,g2]).reshape(1,-1))
st.write(f'### the predicted g3 is {pre_lin_mat[0]:.2f}')
       # Por linearRegression
st.divider()
st.write('## Por LinearRegression')
# For Por
df_st_por = df_student_por.copy()
# Encoding
encoder = LabelEncoder()
for i in df_st_por.select_dtypes(include='object'):
       df_st_por[i] = encoder.fit_transform(df_st_por[i])
       df_st_por_encoded = df_st_por.drop(i,axis=1)
X = df_st_por[[
       'school',
       'sex',
       'age',
       'studytime',
       'failures',
       'schoolsup', 
       'famsup', 
       'paid', 
       'activities',
       'internet', 
       'romantic',
       'freetime', 
       'goout',  
       'health', 
       'absences',
       'g1',
       'g2'    
]]
y = df_st_por['g3']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
       # Our ML model (LinearRegression)
model_lin_por = LinearRegression()
model_lin_por.fit(X_train,y_train)
acc_lin_por = model_lin_por.score(X_test,y_test)
st.write(f'### Accuracy LinearRegression por: {acc_lin_por:.2f}')
       #predict
pre_lin_por = model_lin_por.predict(np.array([school,sex,age,studytime,failures,schoolsup,famsup,paid,activities,internet,romantic,freetime,goout,health,absences,g1,g2]).reshape(1,-1))
st.write(f'### the predicted g3 is {pre_lin_por[0]:.2f}')
st.divider()
st.write('## Classification For Mat')
       # classsification for mat
df_st_mat_fail_pas = df_st_mat.copy()
df_st_mat_fail_pas["performance"] = 'Pass'
df_st_mat_fail_pas.loc[df_st_mat_fail_pas['g3']<=9,'performance'] = 'Fail'
# Encoding
encoder = LabelEncoder()
df_st_mat_fail_pas['performance'] = df_st_mat_fail_pas['performance']
df_st_mat_fail_pas['performance'] = encoder.fit_transform(df_st_mat_fail_pas['performance'])
df_encoded_mat = df_st_mat_fail_pas.drop('g3',axis=1)
X = df_st_mat_fail_pas[[
       'school',
       'sex',
       'age',
       'studytime',
       'failures',
       'schoolsup', 
       'famsup', 
       'paid', 
       'activities',
       'internet', 
       'romantic',
       'freetime', 
       'goout',  
       'health', 
       'absences',
       'g1',
       'g2'    
]]
y = df_st_mat_fail_pas['performance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Our ML model
model_class_mat = LogisticRegression()
# fit model to data
model_class_mat.fit(X_train,y_train)
acc_mat_classification = model_class_mat.score(X_test,y_test)
st.write(f"### Accuracy: {acc_mat_classification:.2f}")
pre_class_mat = model_class_mat.predict(np.array([school,sex,age,studytime,failures,schoolsup,famsup,paid,activities,internet,romantic,freetime,goout,health,absences,g1,g2]).reshape(1,-1))
classes = {0:'Fail',1:'Pass'}
st.write(f"### The predicted student for mat is: {classes[pre_class_mat[0]]}")
st.divider()
   #classification for por
st.write('## Classification For Por')
df_st_por_fail_pas = df_st_por.copy()
df_st_por_fail_pas["performance"] = 'Pass'
df_st_por_fail_pas.loc[df_st_por_fail_pas['g3']<=9,'performance'] = 'Fail'
# Encoding
encoder = LabelEncoder()
df_st_por_fail_pas['performance'] = df_st_por_fail_pas['performance']
df_st_por_fail_pas['performance'] = encoder.fit_transform(df_st_por_fail_pas['performance'])
df_encoded_por = df_st_por_fail_pas.drop('g3',axis=1)
# student Classes encoded
classes = {0:'Fail',1:'Pass'}
X = df_st_por_fail_pas[[
       'school',
       'sex',
       'age',
       'studytime',
       'failures',
       'schoolsup', 
       'famsup', 
       'paid', 
       'activities',
       'internet', 
       'romantic',
       'freetime', 
       'goout',  
       'health', 
       'absences',
       'g1',
       'g2'    
]]
y = df_st_por_fail_pas['performance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Our ML model
model_class_por = LogisticRegression()
# fit model to data
model_class_por.fit(X_train,y_train)
acc_por_classification = model_class_por.score(X_test,y_test)
st.write(f"### Accuracy: {acc_por_classification:.2f}")
# student's school (GP=0,MS=1)
school = 0
# student's gender (F=0,M=1)
sex = 1
# age [16, 18, 17, 19, 15, 22, 20, 21]
age = 12
# weekly study time [2, 1, 3, 4]
studytime = 2
# number of past class failures [0, 3, 1, 2]
failures = 3
# extra educational support (0=No,1=Yes)
schoolsup = 0
# extra educational support (0=No,1=Yes)
famsup = 1
# extra paid classes within the course subject (0=No,1=Yes)
paid = 1 
# extra-curricular activities (0=No,1=Yes)
activities = 1
# Internet access at home (0=No,1=Yes)
internet = 1
# with a romantic relationship (0=No,1=Yes)
romantic  = 0
# free time after school [2, 3, 4, 1, 5]
# (numeric: from 1 -> very low to 5 -> very high)
freetime = 3
#going out with friends [3, 5, 4, 2, 1] 
#(numeric: from 1 - very low to 5 - very high)
goout = 2
# current health status [3, 4, 5, 1, 2]
# (numeric: from 1 - very bad to 5 - very good)
health = 4
# number of school absences 
# (numeric: from 0 to 93)
# [ 2,  0, 16,  7, 11, 23,  4, 10,  6, 13,  8, 20, 12,  5, 14, 19,  3,9,  1, 24, 26, 18, 30, 56, 15, 22, 17, 54, 21, 25, 28, 40, 75]
absences = 2
# first period grade (numeric: from 0 to 20)
g1 = 13
# second period grade (numeric: from 0 to 20)
g2 = 20
pre_class_por = model_class_por.predict(np.array([school,sex,age,studytime,failures,schoolsup,famsup,paid,activities,internet,romantic,freetime,goout,health,absences,g1,g2]).reshape(1,-1))
st.write(f"### The predicted student for por is: {classes[pre_class_por[0]]}")