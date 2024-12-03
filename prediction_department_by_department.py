# ============ Import Needed Models  ============
import pandas as pd

# Model importation 
import xgboost as xgb

# Data processing
from sklearn.preprocessing import LabelEncoder

# Prediction scores
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# ============ Reading Data ============ 
path_train = "~/hfactory_magic_folders/water_shortage_prediction/X_train_Hi5.csv"
path_test = "~/hfactory_magic_folders/water_shortage_prediction/X_test_Hi5.csv"

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

# Management of the department code 
df.loc[df['piezo_station_department_code'] == '2A', 'piezo_station_department_code'] = '20'
df.loc[df['piezo_station_department_code'] == '2B', 'piezo_station_department_code'] = '92'
df.loc[df['piezo_station_department_code'] == '95', 'piezo_station_department_code'] = '94'
df['piezo_station_department_code'] = df['piezo_station_department_code'].astype(int)


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



# ============ Prediction Loop: Predicting GroundWater Levels for each department ============
y_pred_list = pd.DataFrame([], columns = ['piezo_groundwater_level_category', 'row_index'])

for code in df['piezo_station_department_code'].unique():
    
    print('Current Department Code: ', code)

    # Define the train and test set
    data_before_2022 = df[df['piezo_station_department_code'] == code]
    data_before_2022 = data_before_2022[((data_before_2022['year'] == 2020) | (data_before_2022['year'] == 2021))]

    X_train = data_before_2022[data_before_2022['piezo_measurement_date'] < '2021-06-01']
    X_train = X_train.select_dtypes(exclude=['object'])
    X_train.drop(columns = ["piezo_measurement_date"],inplace = True)

    X_test = data_before_2022[(data_before_2022['piezo_measurement_date'] >= '2021-06-01')  & (data_before_2022['piezo_measurement_date'] < '2021-10-01')]
    X_test = X_test.select_dtypes(exclude=['object'])
    X_test.drop(columns = ["piezo_measurement_date"],inplace = True)

    y_train = data_before_2022[data_before_2022['piezo_measurement_date'] < '2021-06-01']["piezo_groundwater_level_category"]
    y_test =  data_before_2022[(data_before_2022['piezo_measurement_date'] >= '2021-06-01') & (data_before_2022['piezo_measurement_date'] < '2021-10-01')]["piezo_groundwater_level_category"]
    
    y_train = y_train.map(custom_mapping)
    y_test = y_test.map(custom_mapping)
    
    # Defining the model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric = 'mlogloss')
    model.fit(X_train, y_train)
    
    # Making predicitions
    y_pred = model.predict(X_test)
    y_pred_labels = pd.Series(y_pred).map(reverse_mapping)
    y_test_labels = y_test.map(reverse_mapping)
    
    y_pred_labels = pd.DataFrame(y_pred_labels, columns = ['piezo_groundwater_level_category'])
    y_pred_labels = y_pred_labels.reset_index(drop = True)
    X_test = X_test.reset_index(drop = True)
    y_pred_labels['row_index'] = X_test['row_index']
    y_pred_list = pd.concat([y_pred_list, y_pred_labels], axis = 0).reset_index(drop = True)
    
    f1 = f1_score(y_test_labels, y_pred_labels['piezo_groundwater_level_category'], average = 'weighted')
    print(f"F1-Score (weighted): {f1:.4f}")


#  ============ Compute global F1-score using row_index to link predictions to tests ============
merged_y = pd.merge(
    y_pred_list,
    y_test,
    on = 'row_index',
    how = 'outer',  # include all rows from both DataFrames
    suffixes = ('_y_pred', '_y_test')  # suffixes for distinguishing the columns
)

merged_y['piezo_groundwater_level_category_y_test'] = merged_y['piezo_groundwater_level_category_y_test'].astype(str)
merged_y['piezo_groundwater_level_category_y_pred'] = merged_y['piezo_groundwater_level_category_y_pred'].astype(str)



#  ============ Computing the F1-score results  ============ 
f1 = f1_score(merged_y['piezo_groundwater_level_category_y_test'], merged_y['piezo_groundwater_level_category_y_pred'], average = 'weighted')
print(f"F1-Score (weighted): {f1:.4f}")



# ============ ============ ============ ============ 
# ================ Working for 2022  ================
# ============ ============ ============ ============


# ============ Preprocessing Data ============ 
df_test = pd.read_csv(path_test, low_memory = False)
df_test['piezo_measurement_date'] = pd.to_datetime(df_test['piezo_measurement_date'])

# Manage the departments
df_test.loc[df_test['piezo_station_department_code'] == '2A', 'piezo_station_department_code'] = '20'
df_test.loc[df_test['piezo_station_department_code'] == '2B', 'piezo_station_department_code'] = '92'
df_test.loc[df_test['piezo_station_department_code'] == '95', 'piezo_station_department_code'] = '94'
df_test['piezo_station_department_code'] = df_test['piezo_station_department_code'].astype(int)

#The test set contains summer 2022 and summer 2022 so for 2022 we select dates < '2023-01-01'
df_test = df_test[df_test['piezo_measurement_date'] < '2023-01-01']
df_test = df_test.select_dtypes(exclude=['object'])
df_test.drop(columns = ["piezo_measurement_date"],inplace = True)




#============ Prediction Loop: Predicting GroundWater Levels for each department ============
y_pred_list = pd.DataFrame([], columns = ['piezo_groundwater_level_category', 'row_index'])

for code in df['piezo_station_department_code'].unique():
    print('Current Department Code: ', code)
    # Define the train and test set
    data_before_2023 = df[df['piezo_station_department_code'] == code]
    
    X_train = data_before_2023[data_before_2023['piezo_measurement_date'] < '2022-06-01']
    X_train = X_train.select_dtypes(exclude=['object'])
    X_train.drop(columns = ["piezo_measurement_date"], inplace = True)

    X_test = df_test[df_test['piezo_station_department_code'] == code]

    y_train = data_before_2023[data_before_2023['piezo_measurement_date'] < '2022-06-01']["piezo_groundwater_level_category"]
    y_train = y_train.map(custom_mapping)
    
    # Define the model 
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric = 'mlogloss')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred_labels = pd.Series(y_pred).map(reverse_mapping) 
    y_pred_labels = pd.DataFrame(y_pred_labels, columns = ['piezo_groundwater_level_category'])
    y_pred_labels = y_pred_labels.reset_index(drop=True)

    X_test = X_test.reset_index(drop=True)

    y_pred_labels['row_index'] = X_test['row_index']
    y_pred_list = pd.concat([y_pred_list, y_pred_labels], axis = 0).reset_index(drop = True)

y_pred_list.to_csv('submissions_2022.csv', index = False)


# ============ ============ ============ ============ 
# ================ Working on 2023  ================
# ============ ============ ============ ============


# ============ Preprocessing Data ============ 
df_test = pd.read_csv(path_test, low_memory = False)
df_test['piezo_measurement_date'] = pd.to_datetime(df_test['piezo_measurement_date'])

# Manage the departments
df_test.loc[df_test['piezo_station_department_code'] == '2A', 'piezo_station_department_code'] = '20'
df_test.loc[df_test['piezo_station_department_code'] == '2B', 'piezo_station_department_code'] = '92'
df_test.loc[df_test['piezo_station_department_code'] == '95', 'piezo_station_department_code'] = '94'
df_test['piezo_station_department_code'] = df_test['piezo_station_department_code'].astype(int)

#The test set contains summer 2022 and summer 2023 so for 2023 we select dates > '2023-01-01'
df_test = df_test[df_test['piezo_measurement_date'] > '2023-01-01']
df_test = df_test.select_dtypes(exclude=['object'])
df_test.drop(columns = ["piezo_measurement_date"],inplace = True)




#============ Prediction Loop: Predicting GroundWater Levels for each department ============
y_pred_list = pd.DataFrame([], columns = ['piezo_groundwater_level_category', 'row_index'])

for code in df['piezo_station_department_code'].unique():
    print('current department code: ', code)
    # Define the train and test set
    data_before_2023 = df[df['piezo_station_department_code'] == code]
    
    X_train = data_before_2023[data_before_2023['piezo_measurement_date'] < '2023-06-01']
    X_train = X_train.select_dtypes(exclude = ['object'])
    X_train.drop(columns = ["piezo_measurement_date"],inplace = True)

    X_test = df_test[df_test['piezo_station_department_code'] == code]

    y_train = data_before_2023[data_before_2023['piezo_measurement_date'] < '2023-06-01']["piezo_groundwater_level_category"]
    y_train = y_train.map(custom_mapping)
    
    # Define the model 
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric = 'mlogloss')
    model.fit(X_train, y_train)
    
    # Make Predictions
    y_pred = model.predict(X_test)
    y_pred_labels = pd.Series(y_pred).map(reverse_mapping) 
    y_pred_labels = pd.DataFrame(y_pred_labels, columns = ['piezo_groundwater_level_category'])
    y_pred_labels = y_pred_labels.reset_index(drop = True)

    X_test = X_test.reset_index(drop=True)

    y_pred_labels['row_index'] = X_test['row_index']
    y_pred_list = pd.concat([y_pred_list, y_pred_labels], axis = 0).reset_index(drop = True)

y_pred_list.to_csv('submissions_2023.csv', index=False)



# ============ ============ ============ ============ 
# =========== Making the final submission ===========
# ============ ============ ============ ============
old_submission = pd.read_csv('submissions_2022.csv', low_memory = False)
new_submission = y_pred_list
final_submission = pd.concat([old_submission, new_submission], axis = 0)

final_submission = final_submission[['row_index', 'piezo_groundwater_level_category']]
final_submission.to_csv('final_submission.csv', index = False)

    