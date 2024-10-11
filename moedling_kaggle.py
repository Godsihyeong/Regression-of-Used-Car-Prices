import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold, cross_validate, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

import xgboost as xgb
import catboost as cb
import lightgbm as lgb

warnings.filterwarnings("ignore")

train = pd.read_csv("./playground-series-s4e9/train.csv", index_col='id')
test = pd.read_csv("./playground-series-s4e9/test.csv", index_col='id')
original = pd.read_csv("./playground-series-s4e9/used_cars.csv")

original['milage'] = original['milage'].str.replace(r'\D', '', regex=True).astype(int)
original['price'] = original['price'].str.replace(r'\D', '', regex=True).astype(int)

train = pd.concat([train, original]).reset_index(drop=True)

import pandas as pd
import numpy as np
import re

# Feature extraction function
def feature_extractor(df):
    df = df.copy()
    
    # Fill missing 'clean_title' values
    df['clean_title'].fillna('No', inplace=True)
    
    # Handle 'accident' column missing values using value distribution
    ratios = df['accident'].value_counts(normalize=True)
    df['accident'] = df['accident'].fillna(pd.Series(np.random.choice(ratios.index, p=ratios.values, size=len(df))))
    
    # Function to extract engine details
    def decode_engine(s: str):
        s = s.lower()
        # Extract HP
        hpgroup = re.search(r'(\d+(\.\d+)?\s*)hp', s)
        engine_hp = float(hpgroup.group(1)) if hpgroup else np.nan
        # Extract CC
        ccgroup = re.search(r'(\d+(\.\d+)?\s*)l', s)
        engine_cc = float(ccgroup.group(1)) if ccgroup else np.nan
        # Extract cylinder count
        cylindergroup = re.search(r'(\d+(\.\d+)?\s*)cylinder', s)
        engine_cyl = int(cylindergroup.group(1)) if cylindergroup else np.nan
        # Turbo
        turbogroup = re.search(r'turbo', s)
        turbo = True if turbogroup else False
        # Flex fuel
        flexgroup = re.search(r'flex fuel|flex', s)
        flex_fuel = True if flexgroup else False
        # Hybrid
        hybridgroup = re.search(r'hybrid', s)
        hybrid = True if hybridgroup else False
        # Electric
        electricgroup = re.search(r'electric', s)
        electric = True if electricgroup else False

        return engine_hp, engine_cc, engine_cyl, turbo, flex_fuel, hybrid, electric

    # Apply decode_engine function and expand it into multiple columns
    df[['engine_hp', 'engine_cc', 'engine_cyl', 'engine_turbo', 'engine_flexfuel', 'engine_hybrid', 'engine_electric']] = df['engine'].apply(decode_engine).apply(pd.Series)
    
    # Drop 'engine' and 'fuel_type' columns as they're no longer needed
    df = df.drop(columns=['engine', 'fuel_type'])
    
    return df

# Apply feature_extractor to both train and test dataframes
train = feature_extractor(train)
test = feature_extractor(test)

def extract_other_features(df):
    
    luxury_brands =  ['Mercedes-Benz', 'BMW', 'Audi', 'Porsche', 'Land', 
                    'Lexus', 'Jaguar', 'Bentley', 'Maserati', 'Lamborghini', 
                    'Rolls-Royce', 'Ferrari', 'McLaren', 'Aston', 'Maybach']
    df['Is_Luxury_Brand'] = df['brand'].apply(lambda x: 1 if x in luxury_brands else 0)
    
    return df

train = extract_other_features(train)
test =  extract_other_features(test)

def clean_transmission(trans):
    trans = trans.lower()
    if any(keyword in trans for keyword in ['a/t', 'at', 'f', 'automatic', 'cvt', 'dual shift', 'overdrive']):
        return 'Automatic'
    elif any(keyword in trans for keyword in ['m/t', 'mt', 'manual', 'single-speed fixed gear']):
        return 'Manual'
    else:
        return 'Automatic'  # 특정하지 않은 값들은 'Unknown'으로 설정
    
train['transmission_clean'] = train['transmission'].apply(clean_transmission)
test['transmission_clean'] = test['transmission'].apply(clean_transmission)


train['accident'] = train['accident'].fillna('None reported')
test['accident'] = test['accident'].fillna('None reported')
train = train.drop(['clean_title'], axis=1)
test = test.drop(['clean_title'], axis=1)

target = 'price'
features = train.drop(target, axis=1).columns.to_list()
print(features)

categorical_features = train[features].select_dtypes(include=["object", 'category']).columns.to_list() + ['model_year', 'Is_Luxury_Brand']

print(categorical_features)

train[categorical_features] = train[categorical_features].astype('category')
test[categorical_features] = test[categorical_features].astype('category')

numerical_features = list(set(features)-set(categorical_features))

for col in ['engine_cc', 'milage', 'engine_hp', 'engine_cyl']:
    imputer = SimpleImputer()
    imputer.fit(pd.concat([train[[col]], test[[col]]]))
    train[col] = imputer.transform(train[[col]])
    test[col] = imputer.transform(test[[col]])


for col in categorical_features:
    encoder = LabelEncoder()
    encoder.fit(pd.concat([train[col], test[col]]))
    train[col] = encoder.transform(train[col])
    test[col] = encoder.transform(test[col])
    
X = train.copy()
y = X.pop(target)

# def model_validator(model, X, y, n_splits=10, random_state=95):
#     kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
#     fold_scores = -cross_val_score(model, X, y, cv=kfold, scoring='neg_root_mean_squared_error')
    
    
#     scores = pd.DataFrame({
#         'Fold': range(1,n_splits+1),
#         'Scores': fold_scores
#     })
    
#     print(f"Model: {model[-1].__class__.__name__}\n")
#     print(scores.to_string(index=False))
#     print(f"\n Average Fold RMSE Score: {np.mean(fold_scores):.6f} \xb1 {np.std(fold_scores):.6f}\n")
    
def model_trainer(model, X, y, n_splits=10, random_state=95):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    rmse_scores = []
    test_preds = np.zeros(test.shape[0])
    print("="*80, f"Model: {model[-1].__class__.__name__}", "="*80, sep='\n')
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
        X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        
        y_pred = model_clone.predict(X_test)
        rmse_score = mean_squared_error(y_test, y_pred)**0.5
        rmse_scores.append(rmse_score)
        print(f"Fold {fold+1}: RMSE Score: {rmse_score:.6f}")
        test_preds += model_clone.predict(test)
    test_preds /= n_splits
    
    print(f"\n Average Fold RMSE Score: {np.mean(rmse_scores):.6f} \xb1 {np.std(rmse_scores):.6f}\n")
    
    return test_preds

lgb_params = {
    'n_estimators': 1000,
    'lambda_l1': 0.04562637246444733,
    'lambda_l2': 0.00011379033429173429,
    'num_leaves': 17,
    'feature_fraction': 0.49851697470341133,
    'bagging_fraction': 0.9558100661080114,
    'bagging_freq': 6,
    'min_child_samples': 80,
    'verbosity':-1,
    'gpu_use_dp': True,
    'device':'GPU'
 }

lgb_pipe = make_pipeline(StandardScaler(), lgb.LGBMRegressor(**lgb_params))
test_lgb_pred = model_trainer(lgb_pipe, X, y)

sample_sub = pd.read_csv('/kaggle/input/playground-series-s4e9/sample_submission.csv')
sample_sub[target] = test_lgb_pred
sample_sub.to_csv('lgbm_submission.csv', index=False)