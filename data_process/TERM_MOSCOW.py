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
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import settings_local as SETTINGS

prepared_data_secondary = SETTINGS.DATA_MOSCOW + '/MOSCOW_VTOR.csv'
prepared_data_new = SETTINGS.DATA_MOSCOW + '/MOSCOW_NEW_FLATS.csv'

PATH_TO_TERM_MODEL_GBR_MOSCOW= SETTINGS.MODEL_MOSCOW+ '/TermModel_Moscow_GBR.joblib'




# Function to calculate term
def train_reg(path_data_new: str, path_data_secondary: str):
    msc_new = pd.read_csv(path_data_new)
    msc_secondary = pd.read_csv(path_data_secondary)
    data = pd.concat([msc_new, msc_secondary], ignore_index=True, axis=0)

    # Log Transformation
    # data['profit'] = data['profit'] + 1 - data['profit'].min()
    data = data._get_numeric_data()
    data[data < 0] = 0

    data[['schools_500m', 'schools_1000m', 'kindergartens_500m',
          'kindergartens_1000m', 'clinics_500m', 'clinics_1000m', 'shops_500m',
          'shops_1000m']] = data[['schools_500m', 'schools_1000m', 'kindergartens_500m',
                                  'kindergartens_1000m', 'clinics_500m', 'clinics_1000m', 'shops_500m',
                                  'shops_1000m']].fillna(0)

    # Remove price and term outliers (out of 3 sigmas)
    data = data[((np.abs(stats.zscore(data.price)) < 2.5) & (np.abs(stats.zscore(data.term)) < 2.5))]


    data['price_meter_sq'] = np.log1p(data['price_meter_sq'])
    data['profit'] = np.log1p(data['profit'])
    # data['term'] = np.log1p(data['term'])
    # data['mode_price_meter_sq'] = np.log1p(data['mode_price_meter_sq'])
    # data['mean_term'] = np.log1p(data['mean_term'])

    # Create X and y for Linear Model training
    X = data[
        ['price_meter_sq', 'profit', 'mm_announce', 'yyyy_announce', 'rent_year', 'windows_view', 'renovation_type',
         'full_sq',
         'is_rented', 'schools_500m', 'schools_1000m', 'kindergartens_500m',
                                       'kindergartens_1000m', 'clinics_500m', 'clinics_1000m', 'shops_500m',
                                       'shops_1000m']]
    y = data[['term']].values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Create LinearModel and fitting
    # reg = LinearRegression().fit(X_train, y_train)
    print("Msc term model training", flush=True)
    reg = GradientBoostingRegressor(n_estimators=450, max_depth=5, verbose=1, random_state=42,
                                    learning_rate=0.07, max_features='sqrt', min_samples_split=5).fit(X_train, y_train)
    preds = reg.predict(X_test)
    acc = r2_score(y_test, preds)
    print("MSC Term R2 acc: {0}".format(acc))

    # Train model using full datset
    reg.fit(X, y)
    dump(reg, PATH_TO_TERM_MODEL_GBR_MOSCOW)
    return reg

# Function to calculate term DUMMIES
# def term_gbr(data: pd.DataFrame, type: str):
#     # Remove price and term outliers (out of 3 sigmas)
#     data1 = data[(np.abs(stats.zscore(data.price)) < 3)]
#     data2 = data[(np.abs(stats.zscore(data.term)) < 3)]
#
#     data = pd.merge(data1, data2, on=list(data.columns), how='left')
#
#     # Fill NaN if it appears after merging
#     data[['term']] = data[['term']].fillna(data[['term']].mean())
#
#     # data.full_sq = np.log1p(data.full_sq)
#     data.price_meter_sq = np.log1p(data.price_meter_sq)
#     # data.mm_announce = np.log1p(data.mm_announce)
#     data.was_opened = np.log1p(data.was_opened)
#     data.term = np.log1p(data.term)
#     data.profit = np.log1p(data.profit)
#
#     data = data[['term', 'price_meter_sq', 'profit', 'price', 'full_sq', 'kitchen_sq', 'life_sq', 'is_apartment',
#       'renovation', 'has_elevator',
#       'time_to_metro', 'floor_first', 'floor_last',
#       'is_rented', 'rent_quarter',
#       'rent_year', 'to_center', 'mm_announce__1',
#       'mm_announce__2', 'mm_announce__3', 'mm_announce__4',
#       'mm_announce__5', 'mm_announce__6', 'mm_announce__7', 'mm_announce__8', 'mm_announce__9',
#       'mm_announce__10', 'mm_announce__11', 'mm_announce__12', 'rooms__0',
#       'rooms__1', 'rooms__2', 'rooms__3', 'rooms__4', 'rooms__5', 'rooms__6', 'yyyy_announce__18',
#       'yyyy_announce__19', 'yyyy_announce__20',
#       'cluster__0', 'cluster__1',
#       'cluster__2', 'cluster__3', 'cluster__4', 'cluster__5', 'cluster__6', 'cluster__7', 'cluster__8',
#       'cluster__9', 'cluster__10', 'cluster__11', 'cluster__12', 'cluster__13', 'cluster__14', 'cluster__15',
#       'cluster__16',
#       'cluster__17', 'cluster__18', 'cluster__19',
#       'cluster__20', 'cluster__21', 'cluster__22', 'cluster__23', 'cluster__24',
#       'cluster__25', 'cluster__26', 'cluster__27', 'cluster__28', 'cluster__29', 'cluster__30',
#       'cluster__31', 'cluster__32',
#       'cluster__33', 'cluster__34', 'cluster__35', 'cluster__36', 'cluster__37', 'cluster__38',
#       'cluster__39', 'cluster__40',
#       'cluster__41', 'cluster__42', 'cluster__43', 'cluster__44', 'cluster__45', 'cluster__46',
#       'cluster__47', 'cluster__48', 'cluster__49', 'cluster__50', 'cluster__51', 'cluster__52',
#       'cluster__53', 'cluster__54', 'cluster__55',
#       'cluster__56', 'cluster__57', 'cluster__58', 'cluster__59', 'cluster__60', 'cluster__61',
#                  'cluster__62', 'cluster__63', 'cluster__64', 'cluster__65', 'cluster__66', 'cluster__67',
#                  'cluster__68', 'cluster__69',
#                  'cluster__70', 'cluster__71', 'cluster__72', 'cluster__73', 'cluster__74', 'cluster__75',
#                  'cluster__76', 'cluster__77',
#                  'cluster__78', 'cluster__79', 'cluster__80', 'cluster__81', 'cluster__82', 'cluster__83',
#                  'cluster__84',
#                  'cluster__85', 'cluster__86', 'cluster__87', 'cluster__88', 'cluster__89', 'cluster__90',
#                  'cluster__91', 'cluster__92',
#                  'cluster__93', 'cluster__94', 'cluster__95', 'cluster__96', 'cluster__97', 'cluster__98',
#                  'cluster__99', 'cluster__100', 'cluster__101', 'cluster__102', 'cluster__103',
#                  'cluster__104', 'cluster__105', 'cluster__106',
#                  'cluster__107', 'cluster__108', 'cluster__109', 'cluster__110', 'cluster__111',
#                  'cluster__112', 'cluster__113', 'cluster__114',
#                  'cluster__115', 'cluster__116', 'cluster__117', 'cluster__119', 'cluster__120',
#                  'cluster__121', 'cluster__122',
#                  'cluster__123', 'cluster__124', 'cluster__125', 'cluster__126', 'cluster__127',
#                  'cluster__128', 'cluster__129']]
#
#     X = data.drop(['term'], axis=1)
#     y = data[['term']]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#     reg = GradientBoostingRegressor(n_estimators=350, max_depth=3, verbose=1, random_state=42,
#                                     learning_rate=0.07)
#     reg.fit(X_train, y_train)
#     gbr_preds = reg.predict(X_test)
#     print('The R2_score of the Gradient boost is', r2_score(y_test, gbr_preds), flush=True)
#     print('RMSE is: \n', mean_squared_error(y_test, gbr_preds), flush=True)
#
#
#     print('Train on full dataset GBR', flush=True)
#     reg.fit(X, y)
#
#     print('Save model GBR ', flush=True)
#     if type=="New_flats":
#         dump(reg, PATH_TO_TERM_MODEL_GBR_NEW_D)
#     elif type=="Secondary":
#         dump(reg, PATH_TO_TERM_MODEL_GBR_Secondary_D)
#
#     cdf = pd.DataFrame(np.transpose(reg.feature_importances_), X.columns, columns=['Coefficients']).sort_values(
#         by=['Coefficients'], ascending=False)
#     # print(cdf)
#
#     return reg



def LEARNER_D():
    train_reg(path_data_new=prepared_data_new, path_data_secondary=prepared_data_secondary)

if __name__ == '__main__':
    LEARNER_D()


