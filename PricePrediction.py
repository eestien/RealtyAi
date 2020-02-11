import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, scorer, mean_squared_error
from joblib import dump, load
from lightgbm import LGBMRegressor
import os
import settings_local as SETTINGS

prepared_data = SETTINGS.DATA

PATH_TO_PRICE_MODEL_GBR = SETTINGS.MODEL + '/PriceModel_GBR.joblib'
PATH_TO_PRICE_MODEL_RF = SETTINGS.MODEL + '/PriceModel_RF.joblib'
PATH_TO_PRICE_MODEL_LGBM = SETTINGS.MODEL + '/PriceModel_LGBM.joblib'



def Model(data: pd.DataFrame):
    from scipy import stats

    data = data[(np.abs(stats.zscore(data.price)) < 3)]
    data = data[(np.abs(stats.zscore(data.term)) < 3)]
    data["longitude"] = np.log1p(data["longitude"])
    data["latitude"] = np.log1p(data["latitude"])
    data['rooms'] = np.log1p(data['rooms'])
    # data['clusters'] = np.log1p(data['clusters'])
    data["full_sq"] = np.log1p(data["full_sq"])
    data["life_sq"] = np.log1p(data["life_sq"])
    data["kitchen_sq"] = np.log1p(data["kitchen_sq"])
    data["price"] = np.log1p(data["price"])

    X = data[['life_sq', 'rooms', 'renovation', 'has_elevator', 'longitude', 'latitude', 'full_sq', 'kitchen_sq',
              'time_to_metro', 'floor_last', 'floor_first', 'clusters']]

    y = data[['price']].values.ravel()


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    gbr_model = GradientBoostingRegressor(n_estimators=350, max_depth=8, verbose=5, max_features=4, random_state=42,
                                          learning_rate=0.07).fit(X_train, y_train)
    gbr_preds = gbr_model.predict(X_test)
    print('The R2_score of the Gradient boost is', r2_score(y_test, gbr_preds))
    print('RMSE is: \n', mean_squared_error(y_test, gbr_preds))

    print('Train on full dataset: ')
    gbr_model.fit(X, y)

    print('Save model: ')
    dump(gbr_model,PATH_TO_PRICE_MODEL_GBR)

    RF = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, verbose=3, n_jobs=-1).fit(X_train, y_train)

    rf_predicts = RF.predict(X_test)

    print('The accuracy of the Gradient boost is', r2_score(y_test, rf_predicts))
    print('RMSE is: \n', mean_squared_error(y_test, rf_predicts))

    print('Train on full dataset: ')
    RF.fit(X, y)
    dump(RF, PATH_TO_PRICE_MODEL_RF)

    # LGBM model
    lgbm_model = LGBMRegressor(objective='regression',
                               learning_rate=0.1,
                               n_estimators=1250, max_depth=6, min_child_samples=1, verbose=3).fit(X_train, y_train)
    lgbm_preds = lgbm_model.predict(X_test)
    print('The accuracy of the lgbm Regressor is', r2_score(y_test, lgbm_preds))
    print('RMSE is: \n', mean_squared_error(y_test, lgbm_preds))

    print('Train on full dataset: ')
    lgbm_model.fit(X, y)

    print('Save model: ')
    dump(lgbm_model, PATH_TO_PRICE_MODEL_LGBM)

def model():
    np.random.seed(42)
    data = pd.read_csv(prepared_data + '/PREPARED_MOSCOW_VTOR.csv')
    data = data.iloc[np.random.permutation(len(data))]

    Model(data)


if __name__ == '__main__':
    model()


