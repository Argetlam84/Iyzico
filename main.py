import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import lightgbm as lgb
import warnings
import re
from helpers import feature_engineering as fe, variable_evaluations as ve, model as ml

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
warnings.filterwarnings('ignore')

df = pd.read_csv("datasets/iyzico_data.csv")
df.drop("Unnamed: 0", axis=1,inplace=True)
df["transaction_date"] = pd.to_datetime(df["transaction_date"])


for id in df.merchant_id.unique():
    plt.figure(figsize=(15, 15))
    plt.subplot(3, 1, 1, title = str(id) + ' 2018-2019 Transaction Count') # We want 3 lines, each with 1 graph, doing the first one
    df[(df.merchant_id == id) & ( df.transaction_date >= "2018-01-01" ) & (df.transaction_date < "2019-01-01")]["Total_Transaction"].plot()
    plt.xlabel('')
    plt.subplot(3, 1, 2,title = str(id) + ' 2019-2020 Transaction Count') # We want 3 lines, each with 1 graph, doing the second one.
    df[(df.merchant_id == id) &( df.transaction_date >= "2019-01-01" )& (df.transaction_date < "2020-01-01")]["Total_Transaction"].plot()
    plt.xlabel('')
    plt.show()



########################
# Date Features
########################
df = fe.create_date_features(df, "transaction_date")

# Observing transaction numbers of member merchants on a yearly and monthly basis
df.groupby(["merchant_id","year","month","day_of_month"]).agg({"Total_Transaction": ["sum", "mean", "median"]}).head()

# Observing total payment amounts of member merchants on a yearly and monthly basis
df.groupby(["merchant_id","year","month"]).agg({"Total_Paid": ["sum", "mean", "median"]})


########################
# Lag/Shifted Features
########################

df = fe.lag_features(df, [91,92,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,
                       350,351,352,352,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,
                       538,539,540,541,542,
                       718,719,720,721,722])


########################
# Rolling Mean Features
########################
df = fe.roll_mean_features(df, [91,92,178,179,180,181,182,359,360,361,449,450,451,539,540,541,629,630,631,720])


########################
# Exponentially Weighted Mean Features
########################

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91,92,178,179,180,181,182,359,360,361,449,450,451,539,540,541,629,630,631,720]
df = fe.ewm_features(df, alphas, lags)


########################
# Black Friday - Summer Solstice
########################
df = fe.add_special_events(df)
df.head()
########################
# One-Hot Encoding
########################
df = pd.get_dummies(df, columns=['merchant_id','day_of_week', 'month'])
df['Total_Transaction'] = np.log1p(df["Total_Transaction"].values)


########################
# Custom Cost Function
########################

# MAE: mean absolute error
# MAPE: mean absolute percentage error
# SMAPE: Symmetric mean absolute percentage error (adjusted MAPE)

########################
# Time-Based Validation Sets
########################
df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

# Train set until the 10th month of 2020.
train = df.loc[(df["transaction_date"] < "2020-10-01"), :]

# Last 3 months of 2020 as the validation set.
val = df.loc[(df["transaction_date"] >= "2020-10-01"), :]

cols = [col for col in train.columns if col not in ['transaction_date', 'id', "Total_Transaction","Total_Paid", "year" ]]

Y_train = train['Total_Transaction']
X_train = train[cols]
Y_val = val['Total_Transaction']
X_val = val[cols]

Y_train.shape, X_train.shape, Y_val.shape, X_val.shape

########################
# LightGBM Model
########################

# LightGBM parameters
lgb_params = {'metric': {'mae'},
              'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,
              'early_stopping_rounds': 200,
              'nthread': -1}


trained_model = ml.train_lightgbm_model(X_train, Y_train, X_val, Y_val, cols, lgb_params)
y_pred_val = ml.predict_with_lightgbm_model(trained_model, X_val)
fe.smape(np.expm1(y_pred_val), np.expm1(Y_val))


########################
# Variable Importance Levels
########################
ve.plot_lgb_importances(trained_model, num=30, plot=True)

lgb.plot_importance(trained_model, max_num_features=20, figsize=(10, 10), importance_type="gain")
plt.show()