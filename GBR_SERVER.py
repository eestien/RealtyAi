import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, scorer, mean_squared_error
# import Realty.config as cf
from joblib import dump, load
import xgboost
import os
import settings_local as SETTINGS

prepared_data = SETTINGS.DATA

PATH_TO_PRICE_MODEL = SETTINGS.MODEL + '/PriceModelGBR.joblib'
PATH_TO_PRICE_MODEL_X = SETTINGS.MODEL + '/PriceModelXGBoost.joblib'
PATH_TO_PRICE_MODEL_CAT = SETTINGS.MODEL + '/PriceModelCatGradient.joblib'



def Model(data: pd.DataFrame):
    from scipy import stats

    data = data[(np.abs(stats.zscore(data.price)) < 3)]
    data = data[(np.abs(stats.zscore(data.term)) < 3)]
    data["longitude"] = np.log1p(data["longitude"])
    data["latitude"] = np.log1p(data["latitude"])
    data["full_sq"] = np.log1p(data["full_sq"])
    data["kitchen_sq"] = np.log1p(data["kitchen_sq"])
    data["X"] = np.log1p(data["X"])
    data["Y"] = np.log1p(data["Y"])
    data["price"] = np.log1p(data["price"])
    X1 = data[['renovation', 'has_elevator', 'longitude', 'latitude', 'full_sq', 'kitchen_sq',
               'is_apartment', 'time_to_metro', 'floor_last', 'floor_first', 'X', 'Y', 'clusters']]
    y1 = data[['price']].values.ravel()
    print(X1.shape, y1.shape)

    clf = GradientBoostingRegressor(n_estimators=350, max_depth=8, verbose=5, learning_rate=0.05)
    clf.fit(X1, y1)
    dump(clf, PATH_TO_PRICE_MODEL)

    # XGBoost
    '''
    X1_xgb = X1.values
    y1_xgb = data[['price']].values

    best_xgb_model = xgboost.XGBRegressor(colsample_bytree=0.4,
                                          gamma=0.5,
                                          learning_rate=0.1,
                                          max_depth=5,
                                          min_child_weight=3,
                                          n_estimators=300,
                                          reg_alpha=0,
                                          reg_lambda=0.6,
                                          subsample=0.8,
                                          seed=42)
    print("XGB start fitting: ")
    best_xgb_model.fit(X1_xgb, y1_xgb)
    dump(best_xgb_model, PATH_TO_PRICE_MODEL_X)
    '''
    # Cat Gradient
    cat = CatBoostRegressor(random_state=42, learning_rate=0.1, iterations=1000)
    train = Pool(X1, y1)
    cat.fit(train, verbose=5)
    dump(clf, PATH_TO_PRICE_MODEL_CAT)

def model():
    data = pd.read_csv(prepared_data + '/COORDINATES_Pred_Term.csv')
    Model(data)


if __name__ == '__main__':
    model()



'''
{'Алтуфьевский': 1, 'Южное Медведково': 114, 'Лосиноостровский': 49, 'Ярославский': 117,
 'Марьина Роща': 52, 'Марфино': 51, 'Бабушкинский': 3, 'Свиблово': 84, 'Останкинский': 70,
  'Северный': 87, 'Алексеевский': 0, 'Ростокино': 80, 'Вешняки': 13, 'Восточное Измайлово': 19,
   'Гольяново': 23, 'Ивановское': 31, 'Северное Измайлово': 85, 'Сокольники': 92, 'Новогиреево': 64,
    'Перово': 73, 'Преображенское': 76, 'Восточный': 20, 'Соколиная Гора': 91, 'Метрогородок': 54,
     'Богородское': 10, 'Новокосино': 65, 'Очаково-Матвеевское': 71, 'Солнцево': 94,
      'Ново-Переделкино': 63, 'Крылатское': 44, 'Внуково': 15, 'Бескудниковский': 6, 'Аэропорт': 2,
       'Хорошёвский': 103, 'Беговой': 5, 'Савёловский': 83, 'Молжаниновский': 58, 'Царицыно': 104,
        'Чертаново Центральное': 106, 'Чертаново Северное': 105, 'Нагатино-Садовники': 60, 'Нагорный': 61,
         'Орехово-Борисово Северное': 69, 'Бирюлёво Западное': 9, 'Бирюлёво Восточное': 8, 'Куркино': 46,
          'Строгино': 96, 'Митино': 56, 'Северное Тушино': 86, 'Покровское-Стрешнево': 75, 'Печатники': 74,
           'Лефортово': 47, 'Гагаринский': 21, 'Южное Бутово': 113, 'Котловка': 41, 'Замоскворечье': 27,
            'Басманный': 4, 'Молодёжный': 59, 'Власиха': 14, 'Ивантеевка': 32, 'Котельники': 40,
             'Красноармейск': 42, 'Лосино-Петровский': 48, 'Серпухов': 88, 'Электрогорск': 112,
              'Дмитровский': 26, 'Истринский': 33, 'Клинский': 35, 'Коломенский район': 37,
               'Красногорский': 43, 'Лотошинский': 50, 'Ногинский': 67, 'Одинцовский': 68,
                'Павлово-Посадский': 72, 'Серпуховский': 89, 'Солнечногорский': 93, 'Шаховской': 109,
                 'Щелковский': 110, 'Бутырский': 12, 'Бибирево': 7, 'Тропарёво-Никулино': 100,
                  'Раменки': 78, 'Войковский': 16, 'Сокол': 90, 'Щукино': 111, 'Южное Тушино': 115,
                   'Восточное Дегунино': 18, 'Головинский': 22, 'Западное Дегунино': 28, 'Коптево': 39,
                    'Тимирязевский': 98, 'Братеево': 11, 'Даниловский': 24, 'Зябликово': 30,
                     'Чертаново Южное': 107, 'Капотня': 34, 'Кузьминки': 45, 'Нижегородский': 62,
                      'Рязанский': 82, 'Зюзино': 29, 'Черёмушки': 108, 'Тёплый Стан': 101, 'Ясенево': 118,
                       'Коньково': 38, 'Таганский': 97, 'Мещанский': 55, 'Якиманка': 116, 'Пресненский': 77,
                        'Троицк': 99, 'Михайлово-Ярцевское': 57, 'Клёновское': 36, 'Вороновское': 17,
                         'Новофёдоровское': 66, 'Роговское': 79, 'Матушкино': 53, 'Сосенское': 95, 
                         'Филимонковское': 102, 'Рязановское': 81, 'Десёновское': 25}
'''