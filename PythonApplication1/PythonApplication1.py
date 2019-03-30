import pandas as pd

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

#from pandas.plotting import scatter_matrix
#import matplotlib.pyplot as plt

data_train = pd.read_csv('train_data.csv')
data_test  = pd.read_csv('test_data.csv')


#print(data.shape)
#print(data.describe())

#categorical_columns = [c for c in data.columns if data[c].dtype.name == 'object']
#numerical_columns   = [c for c in data.columns if data[c].dtype.name != 'object']

#print('categorical_columns: ',categorical_columns)

#print('numerical_columns: ',numerical_columns)

#print('categorical_columns des: ',data[categorical_columns].describe())

#for c in categorical_columns:  print(data[c].unique())
data_train.at[data_train['gamma_ray'] == 'low', 'gamma_ray'] = 0
data_train.at[data_train['gamma_ray'] == 'moderate', 'gamma_ray'] = 1
data_train.at[data_train['gamma_ray'] == 'high', 'gamma_ray'] = 2

data_test.at[data_test['gamma_ray'] == 'low', 'gamma_ray'] = 0
data_test.at[data_test['gamma_ray'] == 'moderate', 'gamma_ray'] = 1
data_test.at[data_test['gamma_ray'] == 'high', 'gamma_ray'] = 2
data_test.at[data_test['gamma_ray'] == 'very high', 'gamma_ray'] = 3


#for c in categorical_columns:  print(data[c].unique())

data_train = data_train.drop(['robot_gear_compression_diff_1',
                        'robot_gear_temperature_diff_5',
                        'robot_gear_temperature_diff_1',
                        'robot_gear_temperature_2',
                        'robot_engine_temperature_11',
                        'robot_probe_circulation_10',
                        'robot_probe_temperature_2',
                        'robot_probe_circulation_2',
                        'weapon_robot_armour_index_2',
                        'robot_gear_temperature_1'], axis=1) #Удаляем найденные корреляционные столбцы: 

data_test = data_test.drop(['robot_gear_compression_diff_1',
                        'robot_gear_temperature_diff_5',
                        'robot_gear_temperature_diff_1',
                        'robot_gear_temperature_2',
                        'robot_engine_temperature_11',
                        'robot_probe_circulation_10',
                        'robot_probe_temperature_2',
                        'robot_probe_circulation_2',
                        'weapon_robot_armour_index_2',
                        'robot_gear_temperature_1'], axis=1) #Удаляем найденные корреляционные столбцы: 
                                            
CorrKoef = data_train.corr() #Посчитал корреляцию
#CorrKoef.to_csv('CorrMatrx.csv', sep=';')


#FieldDrop = [i for i in CorrKoef if CorrKoef[i].isnull().drop_duplicates().values[0]] #Находим поля без корреляции, т.е. независимые
#print(FieldDrop) #Их нет

CorField = [] #Вычисляю коррелирующие с признаков друг с другом больше 90%
for i in CorrKoef:
    for j in CorrKoef.index[CorrKoef[i] > 0.9]:
        if i != j and j not in CorField and i not in CorField:
            CorField.append(j)
            print("%s-->%s: r^2=%f" % (i,j,CorrKoef[i][CorrKoef.index==j].values[0]))

X = data_train.drop(('target'), axis=1)  # Выбрасываем столбец 'target'.
Y = data_train['target']

Xtrn, Xtest, Ytrn, Ytest = train_test_split(X, Y, test_size=0.3)

#LinearRegression(), # метод наименьших квадратов
RFRmodel =  RandomForestRegressor(n_estimators=100, max_features ='sqrt') # случайный лес
RFRmodel.fit(Xtrn,Ytrn)
RFRprediction = RFRmodel.predict(Xtest)
RFRmodelres = r2_score(Ytest, RFRprediction)

print('RandomForestRegressor r^2 = ', RFRmodelres)

#Xprod = data_test.drop(('target'), axis=1)  # Выбрасываем столбец 'target' из тестовой выборки.
#RFRprediction_datatest = RFRmodel.predict(Xprod)

#submission = pd.DataFrame({'year':data_test['year'], 'target':RFRprediction_datatest})
#submission.to_csv('submission.csv', index=False)
#print(submission.head())


#	      KNeighborsRegressor(n_neighbors=6), # метод ближайших соседей
#	      SVR(kernel='linear'), # метод опорных векторов с линейным ядром
#	      LogisticRegression() # логистическая регрессия