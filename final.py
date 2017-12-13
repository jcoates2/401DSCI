#final project
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn import svm 
from sklearn import ensemble 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV

# Print out common error metrics for the binary classifications.
def print_multiclass_classif_error_report(y_test, preds):
	print('Accuracy: ' + str(accuracy_score(y_test, preds)))
	print('Avg. F1 (Micro): ' + str(f1_score(y_test, preds, average='micro')))
	print('Avg. F1 (Macro): ' + str(f1_score(y_test, preds, average='macro')))
	print('Avg. F1 (Weighted): ' + str(f1_score(y_test, preds, average='weighted')))
	print(classification_report(y_test, preds))
	print("Confusion Matrix:\n" + str(confusion_matrix(y_test, preds)))

#Get a list of the categorical features for a given dataframe. M
def cat_features(dataframe):
	td = pd.DataFrame({'a':[1,2,3], 'b':[1.0, 2.0, 3.0]})
	return list(filter(lambda x: not(dataframe[x].dtype in [td['a'].dtype, td['b'].dtype]), list(dataframe)))

wars = pd.read_csv('./star_wars_comma.csv')
 
del wars['Unnamed: 0']
wars = wars.drop(wars.index[0])
seens = wars[['Seen I', 'Seen II', 'Seen III', 'Seen IV', 'Seen V', 'Seen VI']] 
ranks = wars[['Rank I', 'Rank II', 'Rank III', 'Rank IV', 'Rank V', 'Rank VI']] 
others = wars[['seen Star Wars ', 'Star Wars fan?', 'Star Trek Fan', 'Gender', 'Age', 'Household Income', 'Education', 'Location (Census Region)']] 

seens = seens.fillna('No') #1187
ranks = ranks.fillna(0) #1187
others = others.dropna() #672

#seen_rank = wars[['Seen I', 'Seen II', 'Seen III', 'Seen IV', 'Seen V', 'Seen VI', 'Rank I', 'Rank II', 'Rank III', 'Rank IV', 'Rank V', 'Rank VI']]
seen_rank = pd.concat([seens, ranks], axis=1)
seen_rank = pd.get_dummies(seen_rank, columns=cat_features(seen_rank))

datay = wars['seen Star Wars ']


#first seens and predict
#Transform the df to a one-hot encoding.
seens = pd.get_dummies(seens, columns=cat_features(seens))

# Split training and test sets from main set.
x_train, x_test, y_train, y_test = train_test_split(seens, datay, test_size = 0.3, random_state = 4)
print('\n----------- DTREE WITH ENTROPY CRITERION -----------------------')
dtree_entropy_mod = tree.DecisionTreeClassifier(criterion='entropy')
dtree_entropy_mod.fit(x_train, y_train)
preds_entropy = dtree_entropy_mod.predict(x_test)
print_multiclass_classif_error_report(y_test, preds_entropy)


#second ranks then seens and rank

le = preprocessing.LabelEncoder()
data_y = le.fit_transform(datay)

# ---------- Create Linear Fit --------------------------------------------------------
# Split training and test sets from main set. Note: random_state just enables results to be repeated.
x_train, x_test, y_train, y_test = train_test_split(ranks, data_y, test_size = 0.2, random_state = 4)
linear_mod = linear_model.LinearRegression()

# Fit the model.
linear_mod.fit(x_train,y_train)

# Make predictions on test data and look at the results.
preds = linear_mod.predict(x_test)
print('MSE, MAE, R^2, EVS: ' + str([mean_squared_error(y_test, preds), median_absolute_error(y_test, preds),  r2_score(y_test, preds), explained_variance_score(y_test, preds)])) 
 

# D tree for Ranks
x_train, x_test, y_train, y_test = train_test_split(ranks, datay, test_size = 0.3, random_state = 4)
print('\n----------- DTREE WITH ENTROPY CRITERION -----------------------')
dtree_entropy_mod = tree.DecisionTreeClassifier(criterion='entropy')
dtree_entropy_mod.fit(x_train, y_train)
preds_entropy = dtree_entropy_mod.predict(x_test)
print_multiclass_classif_error_report(y_test, preds_entropy)




#For Seen Rank
#SMV GLASS
# Split training and test sets from main set. Note: random_state just enables results to be repeated.
x_train, x_test, y_train, y_test = train_test_split(seen_rank, datay, test_size = 0.4, random_state = 4)

# Build a sequence of models for different n_est and depth values. **NOTE: c=1.0 is equivalent to the default.
cs = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
for c in cs:
	# Create model and fit.
	mod = svm.SVC(C=c)
	mod.fit(x_train, y_train)

	# Make predictions - both class labels and predicted probabilities.
	preds = mod.predict(x_test)
	print('---------- EVALUATING MODEL: C = ' + str(c) + ' -------------------')
	# Look at results.
	print_multiclass_classif_error_report(y_test, preds)
	
loo = LeaveOneOut() 
loo_scores = cross_val_score(mod, seen_rank, data_y, cv=loo) # Use accuracy scoring (default) since each test case has one element.
print('CV Scores (Avg. of Leave-One-Out): ' + str(loo_scores.mean()))
print_multiclass_classif_error_report(y_test, loo_scores)

	
#others
features = list(others)
features.remove('Star Wars fan?')
data_x = others[features]
data_y = others['Star Wars fan?']

le = preprocessing.LabelEncoder()
data_y = le.fit_transform(data_y)

data_x = pd.get_dummies(data_x, columns=cat_features(data_x))

# Split training and test sets from main set.
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)

# Build and evaluate the model
gnb_mod = naive_bayes.GaussianNB()
gnb_mod.fit(x_train, y_train)
preds = gnb_mod.predict(x_test)
print_multiclass_classif_error_report(y_test, preds)

# Illustrate recoding numeric classes back into original (text-based) labels.
y_test_labs = le.inverse_transform(y_test)
pred_labs = le.inverse_transform(preds)
print('(Actual, Predicted): \n' + str(zip(y_test_labs, pred_labs)))




# Create training and test sets for later use.
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.3, random_state = 4)

# ----- PART 1: Basic Cross Validation with SVM --------
mod = svm.SVC(C=2.5)

# Illustrate the 3 major CV approaches. We will use accuracy or F1 macro as our scoring criteria in examples below.
# For a list of all premade scoring criteria see http://scikit-learn.org/stable/modules/model_evaluation.html
k_fold = KFold(n_splits=5, shuffle=True, random_state=4) # Shuffling is needed since classes are ordered.
k_fold_scores = cross_val_score(mod, data_x, data_y, scoring='f1_macro', cv=k_fold)
print('CV Scores (K-Fold): ' + str(k_fold_scores))

loo = LeaveOneOut() 
loo_scores = cross_val_score(mod, data_x, data_y, cv=loo) # Use accuracy scoring (default) since each test case has one element.
print('CV Scores (Avg. of Leave-One-Out): ' + str(loo_scores.mean()))

shuffle_split = ShuffleSplit(test_size=0.2, train_size=0.8, n_splits=5)
ss_scores = cross_val_score(mod, data_x, data_y, scoring='accuracy', cv=shuffle_split)
print('CV Scores (Shuffle-Split): ' + str(ss_scores))

# ----- PART 2: Grid Search + Cross Validation with RF --------
# Optimize a RF classifier and test with grid search.

# Below - notice that n_estimators and max_depth are both params of RF. This is how we specify params with different values to try.
param_grid = {'n_estimators':[5, 10, 50, 100], 'max_depth':[3, 6, None]} 

# Find the best RF and use that. Do a 5-fold CV and score with f1 macro.
optimized_rf = GridSearchCV(ensemble.RandomForestClassifier(), param_grid, cv=5, scoring='f1_macro') 

optimized_rf.fit(x_train, y_train) # Fit the optimized RF just like it is a single model - it essentially is!

print('Grid Search Test Score (Random Forest): ' + str(optimized_rf.score(x_test, y_test)))


# --- PART 3: Model ensemble illustrations ---------------------
# Here is a Bagging classifier that builds many SVM's.
bagging_mod = ensemble.BaggingClassifier(linear_model.LogisticRegression(), n_estimators=200)
k_fold = KFold(n_splits=5, shuffle=True, random_state=4) # Shuffling is needed since classes are ordered.
bagging_mod_scores = cross_val_score(bagging_mod, data_x, data_y, scoring='f1_macro', cv=k_fold)
print('CV Scores (Bagging NB) ' + str(bagging_mod_scores))

# Here is a basic voting classifier with CV and Grid Search.
m1 = svm.SVC()
m2 = ensemble.RandomForestClassifier()
m3 = naive_bayes.GaussianNB()
voting_mod = ensemble.VotingClassifier(estimators=[('svm', m1), ('rf', m2), ('nb', m3)], voting='hard')

# Set up params for combined Grid Search on the voting model. Notice the convention for specifying 
# parameters foreach of the different models.
param_grid = {'svm__C':[0.2, 0.5, 1.0, 2.0, 5.0, 10.0], 'rf__n_estimators':[5, 10, 50, 100], 'rf__max_depth': [3, 6, None]}
best_voting_mod = GridSearchCV(estimator=voting_mod, param_grid=param_grid, cv=5)
best_voting_mod.fit(x_train, y_train)
print('Voting Ensemble Model Test Score: ' + str(best_voting_mod.score(x_test, y_test)))

