# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%% 
from dfply import *
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.cross_validation import cross_val_score 
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from pprint import pprint 
import time 
from sklearn.externals import joblib 
import os 
from math import sqrt 
#%% Loading the data
df_survey_cost= pd.read_csv("C:/Users/Hank/Desktop/survey_cost/survey_cost_0308.csv")
#%% convert avg_bid size to numeric and check the missing value 
df_survey_cost["avg_bid"] = df_survey_cost["avg_bid"].convert_objects(convert_numeric=True)
df_survey_cost.isnull().sum()
#%% fill na with average bid (roughly $78.6)
df_survey_cost=df_survey_cost.fillna(df_survey_cost.median())

#%% Split the data to predictors and target 
df_survey_cost_X = df_survey_cost >> drop(X.cost,X.placement_id,X.impressions, X.placement, X.start_time, X.end_time)
df_survey_cost_Y = df_survey_cost >> select(X.cost)
#%% Training and test data 
X_train, X_test,y_train,y_test = train_test_split(df_survey_cost_X,df_survey_cost_Y,random_state = 0, train_size = 0.8)

#%% Random Hyperparameter Grid 

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features considered for splitting a node
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

#%% Creat the random grid parameters
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(random_grid)
#%% Use the random grid to search for best hyperparameters
# base model to tune 
rf = RandomForestRegressor()

# Random search of parameters using 5 fold cross validation

rf_random = RandomizedSearchCV(estimator=rf,param_distributions=random_grid,n_iter=50, cv=5 ,verbose=2, random_state=42)
#%%
# Fit the random search model 
start = time.time()
rf_random.fit(X_train,y_train)

end = time.time()
elapsed = (end - start)/60

#%% get the best combination 
# save to file in the current working directory
rf_random_pkl_file_name = 'rf_random.pkl'
joblib.dump(rf_random,rf_random_pkl_file_name)
#%%
# Load from file 
rf_random_pkl_file_name = 'rf_random.pkl'
rf_random_model = joblib.load(rf_random_pkl_file_name)

#%% How can I check the best combination 
best_rf = rf_random_model.best_params_

#%%   Grid Search Hyperparameters
# Q why can't we specify 1 in "min_sample_split?
grid_search = {'n_estimators': [700,800,900,1000],
               'max_features': ["sqrt"],
               'max_depth': [100,110,120,130],
               'min_samples_split': [2,4],
               'min_samples_leaf': [2,3],
               'bootstrap':[True]}

pprint(grid_search)

#%%
rf_grid = GridSearchCV(estimator= rf, param_grid = grid_search, cv=5, n_jobs = -1, verbose = 2)
#%%
start_g= time.time()
rf_grid.fit(X_train,y_train)

end_g=time.time()
#%%
elapsed_g = (end_g- start_g)/60

#%%

rf_grid.best_params_
rf_grid_pkl_file_name = 'rf_grid.pkl'
# save to file in the current working directory
joblib.dump(rf_grid, rf_grid_pkl_file_name)

#%%
# Load from file 
rf_grid_pkl_file_name = 'rf_grid.pkl'
rf_grid_model = joblib.load(rf_grid_pkl_file_name)

    
#%% Evaluate the model in terms of mse, mae, rmse 
 
def evaluate_mse(model, test_fearures, test_labels) : 
    prediction = model.predict(test_fearures)
    mse = mean_squared_error(test_labels.values.ravel(),prediction)
    mae = mean_absolute_error(test_labels.values.ravel(),prediction)
    rmse = sqrt(mse)
    
    return mse,mae, rmse

#%% need to be fixed how we can load the model and get the associated mse and 
evaluate_r = evaluate_mse(rf_random_model,X_test,y_test)
evaluate_g = evaluate_mse(rf_grid_model,X_test,y_test)

#%%mean square root error (RMSE)
from math import sqrt 

#%%
model = ["random_search", "grid_search"]
mse = [evaluate_r[0],evaluate_g[0]]
mae = [evaluate_r[1],evaluate_g[1]]
rmse = [evaluate_r[2],evaluate_g[2]]
#time = [elapsed,elapsed_g]

Final_results = pd.DataFrame({
        "Tuning method" : model,
        "MSE":mse,
        "MAE": mae,
        "RMSE":rmse})
    
Final_results = Final_results.set_index("Tuning method")
#%%
mse
mae
model
result = list(zip(model,mse,mae))
print(result)
#%% Make prediction on campaign 20486839

final_test = pd.read_csv("C:/Users/Hank/Desktop/survey_cost/20486839_test.csv")
#%%
preditors = final_test >> drop(X.cost,X.placement_id,X.impressions, X.placement, X.start_time, X.end_time)
#%%
results = rf_grid_model.predict(preditors)
results.tofile("C:/Users/Hank/Desktop/survey_cost/20486839_test_result.csv",sep=',',format='%10.5f')


