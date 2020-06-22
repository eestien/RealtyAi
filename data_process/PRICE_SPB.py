import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, scorer, mean_squared_error
# import Realty.config as cf
from scipy import stats
from joblib import dump, load

from scipy import stats
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import settings_local as SETTINGS

prepared_data_VTOR = SETTINGS.DATA_SPB + '/SPB_VTOR.csv'
prepared_data_NEW = SETTINGS.DATA_SPB + '/SPB_NEW_FLATS.csv'


# Path to prices models with dummies features
PATH_TO_PRICE_MODEL_GBR_D = SETTINGS.MODEL_SPB + '/PriceModel_SPB_GBR_D.joblib'
PATH_TO_PRICE_MODEL_RF_D = SETTINGS.MODEL_SPB + '/PriceModel_SPB_RF_D.joblib'
PATH_TO_PRICE_MODEL_LGBM_D = SETTINGS.MODEL_SPB + '/PriceModel_SPB_LGBM_D.joblib'


def Price_Main(data: pd.DataFrame):

    # Remove price and term outliers (out of 3 sigmas)
    data = data[((np.abs(stats.zscore(data.price)) < 2.5) & (np.abs(stats.zscore(data.term)) < 2.5) & (
                np.abs(stats.zscore(data.full_sq)) < 2.5))]


    # Fill NaN if it appears after merging
    data[['term']] = data[['term']].fillna(data[['term']].mean())

    # Fix year
    data = data[((data.yyyy_announce == 19) | (data.yyyy_announce == 20))]

    # Log Transformation
    data["longitude"] = np.log1p(data["longitude"])
    data["latitude"] = np.log1p(data["latitude"])
    data["full_sq"] = np.log1p(data["full_sq"])
    data["life_sq"] = np.log1p(data["life_sq"])
    data["kitchen_sq"] = np.log1p(data["kitchen_sq"])
    data["to_center"] = np.log1p(data["to_center"])
    data["price"] = np.log1p(data["price"])
    X = data[['life_sq', 'to_center', 'mm_announce', 'rooms', 'renovation', 'has_elevator', 'longitude', 'latitude', 'full_sq', 'kitchen_sq',
              'time_to_metro', 'floor_last', 'floor_first', 'clusters', 'is_rented', 'rent_quarter', 'rent_year']]

    y = data[['price']].values.ravel()
    print(X.shape, y.shape, flush=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # GBR model
    gbr_model = GradientBoostingRegressor(n_estimators=350, max_depth=8, verbose=1, random_state=42)

    print(10*'-', '> GBR Spb started fitting...')
    gbr_model.fit(X_train, y_train)

    gbr_preds = gbr_model.predict(X_test)
    print('Spb GBR R2_score: ', r2_score(y_test, gbr_preds), flush=True)
    print('Spb GBR RMSE : ', mean_squared_error(y_test, gbr_preds), flush=True)

    print('Train GBR on full spb dataset: ', flush=True)
    gbr_model.fit(X, y)

    dump(gbr_model, PATH_TO_PRICE_MODEL_GBR_D)
    print('GBR Spb model saved !', flush=True)

    # RANDOM FOREST REGRESSOR
    RF = RandomForestRegressor(n_estimators=300, verbose=1, n_jobs=-1)

    print(10*'-', '> Rf Spb started fitting...')
    RF.fit(X_train, y_train)

    rf_predicts = RF.predict(X_test)

    print('Spb RF R2_score: ', r2_score(y_test, rf_predicts), flush=True)
    print('Spb RF RMSE: ', mean_squared_error(y_test, rf_predicts), flush=True)

    print('Train RF on full spb dataset: ', flush=True)
    RF.fit(X, y)

    dump(RF, PATH_TO_PRICE_MODEL_RF_D)
    print('GBR Spb model saved !', flush=True)


    # LGBM model
    lgbm_model = LGBMRegressor(objective='regression',
                               learning_rate=0.05,
                               n_estimators=1250, max_depth=7, min_child_samples=1, verbose=0)

    print(10*'-', '> LGBM Spb started fitting...')
    lgbm_model.fit(X_train, y_train)
    lgbm_preds = lgbm_model.predict(X_test)
    print('Spb RF R2_score: ', r2_score(y_test, lgbm_preds), flush=True)
    print('Spb LGBM RMSE: ', mean_squared_error(y_test, lgbm_preds), flush=True)

    print('Train LGBM on full spb dataset: ', flush=True)
    lgbm_model.fit(X, y)

    dump(lgbm_model, PATH_TO_PRICE_MODEL_LGBM_D)
    print('LGBM Spb model saved !', flush=True)



def learner_D():

    # Load Data Flats New and Data Secondary flats
    df1 = pd.read_csv(prepared_data_VTOR)
    df2 = pd.read_csv(prepared_data_NEW)

    # Concatenate two types of flats
    all_data = pd.concat([df1, df2], ignore_index=True)

    # Train models
    Price_Main(all_data)


if __name__ == '__main__':
    learner_D()
