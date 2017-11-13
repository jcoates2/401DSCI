#Assignment 2
#Jacqueline Coates

import pandas as pd
import matplotlib.pyplot as plt

ahouse = pd.read_csv('./Assignment-2/AmesHousingSetA.csv')
bhouse = pd.read_csv('./Assignment-2/AmesHousingSetB.csv')
print(ahouse.head())

#Part 1
ahouse.describe()
list(ahouse)


imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
data_x = imp.fit_transform(data_x)


# Normalize all predictors to range between 0 and 1.
data_x_norm = preprocessing.normalize(data_x, axis=0)

# Alternative - standardize all features to a column-based z-score.
data_x_std = preprocessing.scale(data_x)

print(data_x_std)






#first transform is one hot coding 

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(categorical_features=cat_feature_inds(ahouse)
enc.fit(ahouse.as_matrix())

#second tried quadratic transform

from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score

data = ahouse[['Lot.Area', 'Year.Built', 'Overall.Cond', 'PID']]
quad = PolynomialFeatures(degree=2)
#setting up data x and y 
data_x = quad.fit_transform(data)
data_y = ahouse['SalePrice']
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, te
st_size = 0.2, random_state = 4)
quad_mod = linear_model.LinearRegression()
quad_mod.fit(x_train,y_train)
  
preds = quad_mod.predict(x_test)
print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(y_test, preds),
     median_absolute_error(y_test, preds), r2_score(y_test, preds), explained_variance_score(y_test, preds)])) 

#MSE, MAE, R^2, EVS: 
#[3259600082.7602925, 26791.887891156599, 0.51915028532542695, 0.52207618484561547]

data_x.mean() # 13051591389.736917

data_y.mean() # 10060.029436860068

# Transform the data into quadratic data.
from sklearn.model_selection import train_test_split
from sklearn import linear_model
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)
quad_mod = linear_model.LinearRegression()
quad_mod.fit(x_train,y_train)

preds = quad_mod.predict(x_test)
print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(y_test, preds), \
							   median_absolute_error(y_test, preds), \
							   r2_score(y_test, preds), \
							   explained_variance_score(y_test, preds)])) 
							   
#MSE 25219115.708046429
#MAE 1918.5824406429092
#R^2 0.10994233166230938
#EVS 0.11194650582836907

#Part 2
potential_y = ahouse[['Utilities', 'Year.Built', 'Neighborhood', 'Alley', 'Sale.Condition', 'Overall.Cond', 'SalePrice']]
sm = pd.plotting.scatter_matrix(potential_y)
plt.show()  

#ahouse.plot.scatter(x='SalePrice',y= '') 

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import preprocessing

#x
salePrice = ahouse['SalePrice']
#y
compare = ahouse[['PID', 'Yr.Sold', 'Lot.Area', 'Year.Built']]

#replace all nan with 0
#imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
#compare = imp.fit_transform(compare)

x_train, x_test, y_train, y_test = train_test_split(salePrice, compare, test_size = 0.2, random_state = 4)
linear_mod = linear_model.LinearRegression()
linear_mod.fit(x_train,y_train) 

prediction = linear_mod.predict(x_test)
print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(y_test, prediction), median_absolute_error(y_test, prediction), r2_score(y_test, prediction), explained_variance_score(y_test, prediction)])) 

#Part 3
#plot the potential values
potential_x = ahouse[['Utilities', 'Year.Built', 'Neighborhood', 'Alley', 'Sale.Condition', 'Overall.Cond']]
house = pd.plotting.scatter_matrix(potential_x, diagonal='kde')
plt.tight_layout()
plt.show()

#baseline model (containing all predictors) to predict the price
#change saleprice column to int
pd.to_numeric(ahouse['SalePrice'], errors='ignore')

datax = ahouse[['Lot.Area','Year.Built','PID', 'Overall.Cond']]
datay = ahouse['SalePrice']

xtrain, xtest, ytrain, ytest = train_test_split(datax, datay, test_size = 0.2, random_state = 4)
base_model = linear_model.LinearRegression()
base_model.fit(xtrain, ytrain)
preds = base_model.predict(xtest)

print('MSE '+ str(mean_squared_error(ytest, preds)))
print('MAE '+ str( median_absolute_error(ytest, preds)))
print('R^2 '+ str(r2_score(ytest, preds)))
print('EVS '+ str( explained_variance_score(ytest, preds)))

#MSE 4264807319.22
#MAE 35033.66616
#R^2 0.370864115069
#EVS 0.374617100216

#trying to do the categorical data fit
ahouse = pd.get_dummies(ahouse, columns=cat_features(ahouse))
#house_feature = list(ahouse)
#house_feature.remove('SalePrice')
#datax = ahouse[house_feature]
datax = ahouse[['Lot.Area', 'Year.Built', 'Overall.Cond', 'PID']]
datay = ahouse['SalePrice']


x_train, x_test, y_train, y_test = train_test_split(datax, datay, test_size = 0.2, random_state = 4)
linear_mod = linear_model.LinearRegression()
# Fit the model.
linear_mod.fit(x_train,y_train)
# Make predictions on test data and look at the results.
preds = linear_mod.predict(x_test)
print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(y_test, preds), \
							   median_absolute_error(y_test, preds), \
							   r2_score(y_test, preds), \
							   explained_variance_score(y_test, preds)])) 
#MSE, MAE, R^2, EVS: [
#4264807318.8967705, 35033.666242725216, 0.37086411511655626, 0.37461710026442996]


enc = OneHotEncoder(categorical_features=cat_feature_inds(ahouse))
cat_inds = cat_feature_inds(ahouse)
enc.fit(ahouse.as_matrix())


#PART 4
#baseline model
datax = bhouse[['Lot.Area','Year.Built','PID', 'Overall.Cond']]
datay = bhouse['SalePrice']

xtrain, xtest, ytrain, ytest = train_test_split(datax, datay, test_size = 0.2, random_state = 4)
base_model = linear_model.LinearRegression()
base_model.fit(xtrain, ytrain)
preds = base_model.predict(xtest)

print('MSE '+ str(mean_squared_error(ytest, preds)))
print('MAE '+ str( median_absolute_error(ytest, preds)))
print('R^2 '+ str(r2_score(ytest, preds)))
print('EVS '+ str( explained_variance_score(ytest, preds)))
#MSE 3264170428.56
#MAE 32466.0493046
#R^2 0.362824268442
#EVS 0.366819806075

#quadratic transformation best
data = bhouse[['Lot.Area', 'Year.Built', 'Overall.Cond', 'PID']]
quad = PolynomialFeatures(degree=2)
data_x = quad.fit_transform(data)
data_y = bhouse['SalePrice']
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 4)
quad_mod = linear_model.LinearRegression()
quad_mod.fit(x_train,y_train)
  
preds = quad_mod.predict(x_test)
print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(y_test, preds),
     median_absolute_error(y_test, preds), r2_score(y_test, preds), explained_variance_score(y_test, preds)])) 
#MSE, MAE, R^2, EVS: 
#[7598769575.21733, 32039.4954227095, -0.48330231800021273, -0.46223318831892812]