import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, scorer, mean_squared_error
# import Realty.config as cf
from joblib import dump, load
import xgboost
import os
import settings_local as SETTINGS

prepared_data_secondary = SETTINGS.DATA_MOSCOW + '/MOSCOW_VTOR.csv'
prepared_data_new = SETTINGS.DATA_MOSCOW + '/MOSCOW_NEW_FLATS.csv'


PATH_TO_PRICE_MODEL_GBR = SETTINGS.MODEL_MOSCOW + '/PriceModel_MOSCOW_Vtor_GBR.joblib'
PATH_TO_PRICE_MODEL_RF = SETTINGS.MODEL_MOSCOW + '/PriceModel_MOSCOW_Vtor_RF.joblib'
PATH_TO_PRICE_MODEL_LGBM = SETTINGS.MODEL_MOSCOW + '/PriceModel_MOSCOW_Vtor_LGBM.joblib'
PATH_TO_PRICE_MODEL_GBR_NEW = SETTINGS.MODEL_MOSCOW + '/PriceModel_MOSCOW_NEW_GBR.joblib'
PATH_TO_PRICE_MODEL_RF_NEW = SETTINGS.MODEL_MOSCOW + '/PriceModel_MOSCOW_NEW_RF.joblib'
PATH_TO_PRICE_MODEL_LGBM_NEW = SETTINGS.MODEL_MOSCOW + '/PriceModel_MOSCOW_NEW_LGBM.joblib'

def Model_Secondary(data: pd.DataFrame):

    data = data[(np.abs(stats.zscore(data.price)) < 3)]
    data = data[(np.abs(stats.zscore(data.term)) < 3)]
    data["longitude"] = np.log1p(data["longitude"])
    data["latitude"] = np.log1p(data["latitude"])
    data["full_sq"] = np.log1p(data["full_sq"])
    data["life_sq"] = np.log1p(data["life_sq"])
    data["kitchen_sq"] = np.log1p(data["kitchen_sq"])
    data["price"] = np.log1p(data["price"])

    X = data[['life_sq', 'rooms', 'renovation', 'has_elevator', 'longitude', 'latitude', 'full_sq', 'kitchen_sq',
              'time_to_metro', 'floor_last', 'floor_first', 'clusters']]

    y = data[['price']].values.ravel()
    print(X.shape, y.shape, flush=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    gbr_model = GradientBoostingRegressor(n_estimators=350, max_depth=8, verbose=1, max_features=4, random_state=42,
                                          learning_rate=0.07)
    # gbr_model.fit(X_train, y_train)
    # gbr_preds = gbr_model.predict(X_test)
    # print('The R2_score of the Gradient boost is', r2_score(y_test, gbr_preds), flush=True)
    # print('RMSE is: \n', mean_squared_error(y_test, gbr_preds), flush=True)

    print('Train on full dataset GBR secondary Moscow: ', flush=True)
    gbr_model.fit(X, y)

    print('Save model GBR secondary Moscow: ', flush=True)
    dump(gbr_model, PATH_TO_PRICE_MODEL_GBR)

    RF = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, verbose=1, n_jobs=-1)

    # RF.fit(X_train, y_train)
    # rf_predicts = RF.predict(X_test)
    #
    # print('The accuracy of the RandomForest is', r2_score(y_test, rf_predicts), flush=True)
    # print('RMSE is: \n', mean_squared_error(y_test, rf_predicts), flush=True)

    print('Train on full dataset RF secondary Moscow: ', flush=True)
    RF.fit(X, y)

    print('Save model RF secondary Moscow: ', flush=True)
    dump(RF, PATH_TO_PRICE_MODEL_RF)

    # LGBM model
    lgbm_model = LGBMRegressor(objective='regression',
                               learning_rate=0.1,
                               n_estimators=1250, max_depth=6, min_child_samples=1, verbose=1)
    # lgbm_model.fit(X_train, y_train)
    # lgbm_preds = lgbm_model.predict(X_test)
    # print('The accuracy of the lgbm Regressor is', r2_score(y_test, lgbm_preds), flush=True)
    # print('RMSE is: \n', mean_squared_error(y_test, lgbm_preds), flush=True)

    print('Train on full dataset LGBM secondary Moscow: ', flush=True)
    lgbm_model.fit(X, y)

    print('Save model LGBM secondary Moscow: ', flush=True)
    dump(lgbm_model, PATH_TO_PRICE_MODEL_LGBM)

def Model_New(data: pd.DataFrame):


    data = data[(np.abs(stats.zscore(data.price)) < 3)]
    data = data[(np.abs(stats.zscore(data.term)) < 3)]
    data["longitude"] = np.log1p(data["longitude"])
    data["latitude"] = np.log1p(data["latitude"])
    data["full_sq"] = np.log1p(data["full_sq"])
    data["life_sq"] = np.log1p(data["life_sq"])
    data["kitchen_sq"] = np.log1p(data["kitchen_sq"])
    data["price"] = np.log1p(data["price"])
    X = data[['life_sq', 'rooms', 'renovation', 'has_elevator', 'longitude', 'latitude', 'full_sq', 'kitchen_sq', 'time_to_metro', 'floor_last', 'floor_first', 'clusters', 'is_rented', 'rent_quarter', 'rent_year']]

    y = data[['price']].values.ravel()
    print(X.shape, y.shape, flush=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    gbr_model = GradientBoostingRegressor(n_estimators=350, max_depth=8, verbose=1, max_features=4, random_state=42,
                                          learning_rate=0.07)
    # gbr_model.fit(X_train, y_train)
    # gbr_preds = gbr_model.predict(X_test)
    # print('The R2_score of the Gradient boost is', r2_score(y_test, gbr_preds), flush=True)
    # print('RMSE is: \n', mean_squared_error(y_test, gbr_preds), flush=True)

    print('Train on full dataset GBR new Moscow: ', flush=True)
    gbr_model.fit(X, y)

    print('Save model GBR new Moscow: ', flush=True)
    dump(gbr_model, PATH_TO_PRICE_MODEL_GBR_NEW)

    RF = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, verbose=1, n_jobs=-1)

    # RF.fit(X_train, y_train)
    # rf_predicts = RF.predict(X_test)
    #
    # print('The accuracy of the RandomForest is', r2_score(y_test, rf_predicts), flush=True)
    # print('RMSE is: \n', mean_squared_error(y_test, rf_predicts), flush=True)

    print('Train on full dataset RF new Moscow: ', flush=True)
    RF.fit(X, y)

    print('Save model RF new Moscow: ', flush=True)
    dump(RF, PATH_TO_PRICE_MODEL_RF_NEW)

    # LGBM model
    lgbm_model = LGBMRegressor(objective='regression',
                               learning_rate=0.1,
                               n_estimators=1250, max_depth=6, min_child_samples=1, verbose=1)
    # lgbm_model.fit(X_train, y_train)
    # lgbm_preds = lgbm_model.predict(X_test)
    # print('The accuracy of the lgbm Regressor is', r2_score(y_test, lgbm_preds), flush=True)
    # print('RMSE is: \n', mean_squared_error(y_test, lgbm_preds), flush=True)

    print('Train on full dataset LGBM new Moscow: ', flush=True)
    lgbm_model.fit(X, y)

    print('Save model LGBM new Moscow: ', flush=True)
    dump(lgbm_model, PATH_TO_PRICE_MODEL_LGBM_NEW)


def model():
    data_secondary = pd.read_csv(prepared_data_secondary)
    Model_Secondary(data_secondary)
    data_new = pd.read_csv(prepared_data_new)
    Model_New(data_new)


if __name__ == '__main__':
    model()


