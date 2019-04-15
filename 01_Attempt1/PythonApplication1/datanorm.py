import pandas as pd
from sklearn import preprocessing
import sklearn.linear_model as lm
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

train = pd.read_csv('train_data.csv')
train = train.drop('gamma_ray',axis = 1)
test = pd.read_csv('test_data.csv')
test = test.drop('gamma_ray',axis = 1)

X = train.drop(['year','target'],axis = 1)
Y = train['target']
Xtst = test.drop(['year','target'],axis = 1)

Xtrn, Xtest, Ytrn, Ytest = train_test_split(X, Y, test_size=0.3)

#print(X.head())
#=================================
#CorrKoef = X.corr() #Посчитал корреляцию
#FieldDrop = [i for i in CorrKoef if CorrKoef[i].isnull().drop_duplicates().values[0]] #Находим поля без корреляции, т.е. независимые
#print(FieldDrop) #Их нет

#CorField = [] #Вычисляю коррелирующие с признаков друг с другом больше 90%
#for i in CorrKoef:
#    for j in CorrKoef.index[CorrKoef[i] > 0.9]:
#        if i != j and j not in CorField and i not in CorField:
#            CorField.append(j)
#            print("%s-->%s: r^2=%f" % (i,j,CorrKoef[i][CorrKoef.index==j].values[0]))
#==================================

model = lm.LinearRegression().fit(Xtrn,Ytrn)
prediction = model.predict(Xtest)
res = r2_score(Ytest, prediction)
print('res : ',res)

subm_res = model.predict(Xtst)

submission = pd.DataFrame({'year':test['year'], 'target':subm_res})
submission.to_csv('submission2.csv', index=False)
print(submission.head())
