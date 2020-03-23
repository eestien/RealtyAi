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
import settings_local as SETTINGS

prepared_data_VTOR = SETTINGS.DATA_SPB + '/SPB_VTOR.csv'
prepared_data_NEW = SETTINGS.DATA_SPB + '/SPB_NEW_FLATS.csv'

PATH_TO_PRICE_MODEL_GBR_VTOR = SETTINGS.MODEL_SPB + '/PriceModel_SPB_Vtor_GBR.joblib'
PATH_TO_PRICE_MODEL_RF_VTOR = SETTINGS.MODEL_SPB + '/PriceModel_SPB_Vtor_RF.joblib'
PATH_TO_PRICE_MODEL_LGBM_VTOR = SETTINGS.MODEL_SPB + '/PriceModel_SPB_Vtor_LGBM.joblib'
PATH_TO_PRICE_MODEL_GBR_NEW = SETTINGS.MODEL_SPB + '/PriceModel_SPB_NEW_GBR.joblib'
PATH_TO_PRICE_MODEL_RF_NEW = SETTINGS.MODEL_SPB + '/PriceModel_SPB_NEW_RF.joblib'
PATH_TO_PRICE_MODEL_LGBM_NEW = SETTINGS.MODEL_SPB + '/PriceModel_SPB_NEW_LGBM.joblib'




def Price_Secondary(data: pd.DataFrame):

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

    print('Train on full dataset GBR secondary Spb: ', flush=True)
    gbr_model.fit(X, y)

    print('Save model GBR secondary Spb: ', flush=True)
    dump(gbr_model, PATH_TO_PRICE_MODEL_GBR_VTOR)

    RF = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, verbose=1, n_jobs=-1)

    # RF.fit(X_train, y_train)

    # rf_predicts = RF.predict(X_test)
    #
    # print('The accuracy of the RandomForest is', r2_score(y_test, rf_predicts), flush=True)
    # print('RMSE is: \n', mean_squared_error(y_test, rf_predicts), flush=True)

    print('Train on full dataset RF secondary Spb: ', flush=True)
    RF.fit(X, y)

    print('Save model RF secondary Spb: ', flush=True)
    dump(RF, PATH_TO_PRICE_MODEL_RF_VTOR)

    # LGBM model
    lgbm_model = LGBMRegressor(objective='regression',
                               learning_rate=0.1,
                               n_estimators=1250, max_depth=6, min_child_samples=1, verbose=1)

    # lgbm_model.fit(X_train, y_train)
    # lgbm_preds = lgbm_model.predict(X_test)
    # print('The accuracy of the lgbm Regressor is', r2_score(y_test, lgbm_preds), flush=True)
    # print('RMSE is: \n', mean_squared_error(y_test, lgbm_preds), flush=True)

    print('Train on full dataset LGBM secondary Spb: ', flush=True)
    lgbm_model.fit(X, y)

    print('Save model LGBM secondary Spb: ', flush=True)
    dump(lgbm_model, PATH_TO_PRICE_MODEL_LGBM_VTOR)


def Pirce_NewFlats(data: pd.DataFrame):


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

    print('Train on full dataset GBR new Spb: ', flush=True)
    gbr_model.fit(X, y)

    print('Save model GBR new Spb: ', flush=True)
    dump(gbr_model, PATH_TO_PRICE_MODEL_GBR_NEW)

    RF = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, verbose=1, n_jobs=-1).fit(X_train, y_train)

    # rf_predicts = RF.predict(X_test)
    #
    # print('The accuracy of the RandomForest is', r2_score(y_test, rf_predicts), flush=True)
    # print('RMSE is: \n', mean_squared_error(y_test, rf_predicts), flush=True)

    print('Train on full dataset RF new Spb: ', flush=True)
    RF.fit(X, y)

    print('Save model RF new Spb: ', flush=True)
    dump(RF, PATH_TO_PRICE_MODEL_RF_NEW)

    # LGBM model
    lgbm_model = LGBMRegressor(objective='regression',
                               learning_rate=0.1,
                               n_estimators=1250, max_depth=6, min_child_samples=1, verbose=1)
    # lgbm_model.fit(X_train, y_train)
    # lgbm_preds = lgbm_model.predict(X_test)
    # print('The accuracy of the lgbm Regressor is', r2_score(y_test, lgbm_preds), flush=True)
    # print('RMSE is: \n', mean_squared_error(y_test, lgbm_preds), flush=True)

    print('Train on full dataset LGBM new Spb: ', flush=True)
    lgbm_model.fit(X, y)

    print('Save model LGBM new Spb: ', flush=True)
    dump(lgbm_model, PATH_TO_PRICE_MODEL_LGBM_NEW)


def learner():
    data = pd.read_csv(prepared_data_VTOR)
    Price_Secondary(data)
    new_data = pd.read_csv(prepared_data_NEW)
    Pirce_NewFlats(new_data)


if __name__ == '__main__':
    learner()
