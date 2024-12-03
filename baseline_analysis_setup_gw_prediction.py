# # ============ Import Needed Models  ============ 
import pandas as pd
import numpy as np

# Model importation 
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier


# Data processing
from sklearn.impute import SimpleImputer

# Prediction scores
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# Hyper parameter tuning
from sklearn.model_selection import RandomizedSearchCV


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)



# ============ Reading Data ============ 
path_train = "~/hfactory_magic_folders/water_shortage_prediction/X_train_Hi5.csv"
df = pd.read_csv(path_train, low_memory = False)



# ============ Preprocessing Data ============ 
# Convert the column 'piezo_measurement_date' into datetime
df['piezo_measurement_date'] = pd.to_datetime(df['piezo_measurement_date'])

# Add memory in the important columns (for day i, add information about the day i-1)
important_columns = [
    'piezo_station_altitude', 'piezo_station_investigation_depth', 'meteo_evapotranspiration_grid',
    'meteo_cloudiness_height', 'meteo_wind_speed_avg_2m', 'meteo_temperature_avg',
    'meteo_humidity_avg', 'meteo_rain_height'
]

for column in important_columns:
    for i in range(1, 2): #we can add memory about more days if we want 
        new_name = f'prev_{i}_{column}'  
        df[new_name] = df[column].shift(i) 

# Dropping columns that had to much noise
seuil = 0.8
cols_to_drop = df.columns[(df.isna().sum() / len(df)) > seuil]
df.drop(columns=cols_to_drop, inplace=True)



# ============ Define the train and test set ============ 
data_before_2022 = df[((df['year'] == 2020) | (df['year'] == 2021))]

X_train = data_before_2022[data_before_2022['piezo_measurement_date'] < '2021-06-01'] # chose data before summer 
X_train = X_train.select_dtypes(exclude = ['object']) # remove categorical data which doesn't add much to the prediction and have a lot of nans
X_train.drop(columns = ["piezo_measurement_date"],inplace = True)

X_test = data_before_2022[(data_before_2022['piezo_measurement_date'] >= '2021-06-01')  & (data_before_2022['piezo_measurement_date'] < '2021-10-01')] # chose the summer part 
X_test = X_test.select_dtypes(exclude=['object']) # remove categorical data which doesn't add much to the prediction and have a lot of nans
X_test.drop(columns = ["piezo_measurement_date"],inplace = True)

# The y vector is the level of groundwater
y_train = data_before_2022[data_before_2022['piezo_measurement_date'] < '2021-06-01']["piezo_groundwater_level_category"]
y_test =  data_before_2022[(data_before_2022['piezo_measurement_date'] >= '2021-06-01') & (data_before_2022['piezo_measurement_date'] < '2021-10-01')]["piezo_groundwater_level_category"]

# Map the water level into categories  
custom_mapping = {
    'Very Low': 0,
    'Low': 1,
    'Average': 2,
    'High': 3,
    'Very High': 4
}
reverse_mapping = {
    0: 'Very Low',
    1: 'Low',
    2: 'Average',
    3: 'High',
    4: 'Very High'
}

y_train = y_train.map(custom_mapping)  
y_test = y_test.map(custom_mapping)


# ============ XGBClassifier ============
print("Training XGBoost Model")
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_labels = pd.Series(y_pred).map(reverse_mapping)

y_test_labels = y_test.map(reverse_mapping)

# Compute the F1 Score
f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')
print(f"F1-Score (weighted): {f1:.4f}")


# ============ Perform imputation for Random Forest and Extra Trees ============ 
columns_full_nan = [col for col in X_train.columns if X_train[col].isna().all()]
X_train = X_train.drop(columns = columns_full_nan)
X_test = X_test.drop(columns = columns_full_nan)

imputer = SimpleImputer(strategy='median')
imputed_data = imputer.fit_transform(X_train)

X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)


# ============ Extra Trees Model ============ 
print("Training Extra Trees Model")
model = ExtraTreesClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_labels = pd.Series(y_pred).map(reverse_mapping)

y_test_labels = y_test.map(reverse_mapping)

f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')
print(f"F1-Score (weighted): {f1:.4f}")


# ============ Random Forest Model ============ 
print("Training Random Forest Model")
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_labels = pd.Series(y_pred).map(reverse_mapping)

y_test_labels = y_test.map(reverse_mapping)

f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')
print(f"F1-Score (weighted): {f1:.4f}")



#============ Hyperparameter tuning for Random Forest ============ 
print("Tuning parameters for Random Forest")

#Hyperparameter tuning for Random Forest

param_dist = {
    'n_estimators': [50, 100, 200, 300, 400, 500], 
    'max_depth': [None, 10, 20, 30, 40, 50], 
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],  
    'max_features': ['auto', 'sqrt', 'log2'],  
    'bootstrap': [True, False] 
}

model = RandomForestClassifier()

random_search = RandomizedSearchCV(
    estimator = model,
    param_distributions = param_dist,
    n_iter = 100,
    scoring = 'f1_weighted', 
    cv = 5,  
    verbose = 2,
    random_state = 42,
    n_jobs = -1  
)

random_search.fit(X_train, y_train)
best_params = random_search.best_params_
best_model = random_search.best_estimator_

print(f"Best Parameters: {best_params}")

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
y_pred_labels = pd.Series(y_pred).map(reverse_mapping)
y_test_labels = y_test.map(reverse_mapping)

f1 = f1_score(y_test_labels, y_pred_labels, average = 'weighted')
print(f"F1-Score (weighted) with Random Forest's best model: {f1:.4f}")

#============ Hyperparameter tuning for GBoost ============ 
print("Tuning parameters for GBoost")

param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],  
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],  
    'max_depth': [3, 5, 7, 9, 11],  
    'min_child_weight': [1, 3, 5, 7],  
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],  
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],  
    'gamma': [0, 0.1, 0.2, 0.3, 0.4],  
    'reg_alpha': [0, 0.01, 0.1, 1, 10], 
    'reg_lambda': [1, 2, 5, 10], 
}

model = xgb.XGBClassifier(use_label_encoder = False, eval_metric='mlogloss')

random_search = RandomizedSearchCV(
    estimator = model,
    param_distributions = param_dist,
    n_iter =  100,  
    scoring = 'f1_weighted',  
    cv = 5,  
    verbose = 2,
    random_state = 42,
    n_jobs = -1 
)

random_search.fit(X_train, y_train)

best_params = random_search.best_params_
best_model = random_search.best_estimator_

print(f"Best Parameters: {best_params}")

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
y_pred_labels = pd.Series(y_pred).map(reverse_mapping)
y_test_labels = y_test.map(reverse_mapping)

f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')
print(f"F1-Score (weighted) with best model: {f1:.4f}")


