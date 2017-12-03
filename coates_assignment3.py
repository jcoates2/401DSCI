import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
from sklearn import naive_bayes
from sklearn import ensemble 
#from data_utils import *

churn = pd.read_csv('./churn_data.csv')


# Get a list of the categorical features for a given dataframe. M
def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return list(filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe)))
	

# Print out common error metrics for the binary classifications.
def print_binary_classif_error_report(y_test, preds):
	print('Accuracy: ' + str(accuracy_score(y_test, preds)))
	print('Precison: ' + str(precision_score(y_test, preds)))
	print('Recall: ' + str(recall_score(y_test, preds)))
	print('F1: ' + str(f1_score(y_test, preds)))
	print('ROC AUC: ' + str(roc_auc_score(y_test, preds)))
	print("Confusion Matrix:\n" + str(confusion_matrix(y_test, preds)))
	
def print_multiclass_classif_error_report(y_test, preds):
	print('Accuracy: ' + str(accuracy_score(y_test, preds)))
	print('Avg. F1 (Micro): ' + str(f1_score(y_test, preds, average='micro')))
	print('Avg. F1 (Macro): ' + str(f1_score(y_test, preds, average='macro')))
	print('Avg. F1 (Weighted): ' + str(f1_score(y_test, preds, average='weighted')))
	print(classification_report(y_test, preds))
	print("Confusion Matrix:\n" + str(confusion_matrix(y_test, preds)))
	
del churn['CustID']

#list
features = list(churn)
features.remove('Churn')

datax = churn[features]
# Transform the df to a one-hot encoding.
datax = pd.get_dummies(datax, columns=cat_features(datax))
datay = churn['Churn']

datax.head()
#decision tree using gini
le = preprocessing.LabelEncoder()
datay = le.fit_transform(datay)
xtrain, xtest, ytrain, ytest = train_test_split(datax, datay, test_size = 0.3, random_state = 4)

dtree_gini_mod = tree.DecisionTreeClassifier(criterion='gini')
dtree_gini_mod.fit(xtrain, ytrain)
preds_gini = dtree_gini_mod.predict(xtest)

print('Gini all columns')
print_binary_classif_error_report(ytest, preds_gini)
print_multiclass_classif_error_report(ytest, preds_gini)

#decision tree using entropy
dtree_entropy_mod = tree.DecisionTreeClassifier(criterion='entropy')
dtree_entropy_mod.fit(xtrain, ytrain)
preds_entropy = dtree_entropy_mod.predict(xtest)

print('Entropy all columns')
print_binary_classif_error_report(ytest, preds_entropy)
print_multiclass_classif_error_report(ytest, preds_entropy)

#removing column and doing the decision tree again.
#del datax['Gender_Male']
#del datax['Income_Upper']

#datax.head()
#redo decision tree using gini
le = preprocessing.LabelEncoder()
datay = le.fit_transform(datay)
xtrain, xtest, ytrain, ytest = train_test_split(datax, datay, test_size = 0.4, random_state = 4)

dtree_gini_mod = tree.DecisionTreeClassifier(criterion='gini')
dtree_gini_mod.fit(xtrain, ytrain)
preds_gini = dtree_gini_mod.predict(xtest)

print('more test Gini')
print_binary_classif_error_report(ytest, preds_gini)
print_multiclass_classif_error_report(ytest, preds_gini)

# re do decision tree using entropy
dtree_entropy_mod = tree.DecisionTreeClassifier(criterion='entropy')
dtree_entropy_mod.fit(xtrain, ytrain)
preds_entropy = dtree_entropy_mod.predict(xtest)

print('more test Entropy')
print_binary_classif_error_report(ytest, preds_entropy)
print_multiclass_classif_error_report(ytest, preds_entropy)

# naive bays algorithm
datax = churn[features]
datay = churn['Churn']

le = preprocessing.LabelEncoder()
datay = le.fit_transform(datay)
datax = pd.get_dummies(datax, columns=cat_features(datax))

# Split training and test sets from main set.
x_train, x_test, y_train, y_test = train_test_split(datax, datay, test_size = 0.3, random_state = 4)

# Build and evaluate the model
gnb_mod = naive_bayes.GaussianNB()
gnb_mod.fit(x_train, y_train)
preds = gnb_mod.predict(x_test)
print('Naive Bays')
print_binary_classif_error_report(y_test, preds)
print_multiclass_classif_error_report(y_test, preds)

#random forest trees
data_x = churn[features]
data_y = churn['Churn']
data_x = pd.get_dummies(data_x, columns=cat_features(data_x))

# Split training and test sets from main set. Note: random_state just enables results to be repeated.
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)

# Build a sequence of models for different n_est and depth values. **NOTE: nest=10, depth=None is equivalent to the default.
n_est = [5, 10, 50, 100]
depth = [3, 6, None]
for n in n_est:
	for dp in depth:
		# Create model and fit.
		mod = ensemble.RandomForestClassifier(n_estimators=n, max_depth=dp)
		mod.fit(x_train, y_train)

		# Make predictions - both class labels and predicted probabilities.
		preds = mod.predict(x_test)
		print('---------- EVALUATING MODEL: n_estimators = ' + str(n_est) + ', depth =' + str(dp) + ' -------------------')
		# Look at results.
		print_multiclass_classif_error_report(y_test, preds)


#conduct best model
data_x = churn[features]
data_y = churn['Churn']
data_x = pd.get_dummies(data_x, columns=cat_features(data_x))


x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)
mod = ensemble.RandomForestClassifier(n_estimators=100, max_depth=6)
mod.fit(x_train, y_train)

churn_val = pd.read_csv('./churn_validation.csv')
del churn_val['CustID']
del churn_val['Churn']

churn_val = pd.get_dummies(churn_val, columns=cat_features(churn_val))
vx_train, vx_test, vy_train, vy_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)
# Make predictions - both class labels and predicted probabilities.
preds = mod.predict(vx_test)
print('Predicting on Churn Validation')
# Look at results.
print_multiclass_classif_error_report(y_test, preds)



##trying seaborn
model = pd.get_dummies(churn, columns=cat_features(churn))
sns.set(style="whitegrid", color_codes=True)

sns.stripplot(x="Age", y="Churn_Yes", data=model, jitter=True);
plt.show()
sns.stripplot(x="FamilySize", y="Churn_Yes", data=model, jitter=True); #people who were yes, no one with one family size
plt.show()
sns.stripplot(x="Education", y="Churn_Yes", data=model, jitter=True); # yes people had more education
plt.show()
sns.stripplot(x="Calls", y="Churn_Yes", data=model, jitter=True); #more calls more likely to churn
plt.show()
sns.stripplot(x="Visits", y="Churn_Yes", data=model, jitter=True);
plt.show()
sns.stripplot(x="Gender_Female", y="Churn_Yes", data=model, jitter=True); 
plt.show()
sns.stripplot(x="Income_Lower", y="Churn_Yes", data=model, jitter=True);
plt.show()

#check significance with box and whisker
sns.factorplot(x="Churn", y="Calls", hue="Education", col="FamilySize", data=churn, kind="box", size=4, aspect=.5);
plt.show()



