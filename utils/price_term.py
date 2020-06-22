from datetime import datetime
import pandas as pd
import numpy as np
from joblib import dump, load
from math import sin, cos, sqrt, atan2, radians
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from scipy import stats
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import settings_local as SETTINGS


# Define paths to Moscow and Spb Secondary flats models DUMMIES
PATH_PRICE_GBR_MOSCOW_D = SETTINGS.MODEL_MOSCOW + '/PriceModel_MOSCOW_GBR_D.joblib'
PATH_PRICE_RF_MOSCOW_D = SETTINGS.MODEL_MOSCOW + '/PriceModel_MOSCOW_RF_D.joblib'
PATH_PRICE_LGBM_MOSCOW_D = SETTINGS.MODEL_MOSCOW + '/PriceModel_MOSCOW_LGBM_D.joblib'
PATH_PRICE_GBR_SPB_D = SETTINGS.MODEL_SPB + '/PriceModel_SPB_GBR_D.joblib'
PATH_PRICE_RF_SPB_D = SETTINGS.MODEL_SPB + '/PriceModel_SPB_RF_D.joblib'
PATH_PRICE_LGBM_SPB_D = SETTINGS.MODEL_SPB + '/PriceModel_SPB_LGBM_D.joblib'

# Define paths to Moscow and Spb clustering models
KMEANS_CLUSTERING_MOSCOW_MAIN = SETTINGS.MODEL_MOSCOW + '/KMEANS_CLUSTERING_MOSCOW_MAIN.joblib'
KMEANS_CLUSTERING_SPB_MAIN = SETTINGS.MODEL_SPB + '/KMEANS_CLUSTERING_SPB_MAIN.joblib'

# Define paths to Moscow and Spb data
MOSCOW_DATA_NEW = SETTINGS.DATA_MOSCOW + '/MOSCOW_NEW_FLATS.csv'
MOSCOW_DATA_SECONDARY = SETTINGS.DATA_MOSCOW + '/MOSCOW_VTOR.csv'
SPB_DATA_NEW = SETTINGS.DATA_SPB + '/SPB_NEW_FLATS.csv'
SPB_DATA_SECONDARY = SETTINGS.DATA_SPB + '/SPB_VTOR.csv'


# Predict price and term
def map_estimation(longitude, rooms, latitude, full_sq, kitchen_sq, life_sq, renovation, secondary, has_elevator,
                   floor_first, floor_last, time_to_metro, is_rented, rent_year, rent_quarter, city_id):
    # Get current time
    now = datetime.now()

    # City_id: 0 = Moscow, 1 = Spb

    def define_city(city_id: int, secondary: int):

        city_center_lon = 0
        city_center_lat = 0

        data = pd.DataFrame()
        kmeans, gbr, rf, lgbm = 0, 0, 0, 0
        if city_id == 0:
            # Load data Moscow flats
            data1 = pd.read_csv(MOSCOW_DATA_NEW)
            data2 = pd.read_csv(MOSCOW_DATA_SECONDARY)

            data = pd.concat([data1, data2], ignore_index=True)

            del data1
            del data2

            # Load KMean Clustering model
            kmeans = load(KMEANS_CLUSTERING_MOSCOW_MAIN)

            # Load Price Models Moscow Secondary
            gbr = load(PATH_PRICE_GBR_MOSCOW_D)
            rf = load(PATH_PRICE_RF_MOSCOW_D)
            lgbm = load(PATH_PRICE_LGBM_MOSCOW_D)
            print("Pretrained models loaded! Moscow")

            city_center_lon = 37.619291
            city_center_lat = 55.751474


        # # Москва вторичка
        # elif city_id == 0 and secondary == 1:
        #     # Load data Moscow secondary
        #     data = pd.read_csv(MOSCOW_DATA_SECONDARY)
        #
        #     # Load KMean Clustering model
        #     kmeans = load(KMEANS_CLUSTERING_MOSCOW_MAIN)
        #
        #     # Load Price Models Moscow Secondary
        #     gbr = load(PATH_PRICE_GBR_MOSCOW_VTOR)
        #     rf = load(PATH_PRICE_GBR_MOSCOW_VTOR)
        #     lgbm = load(PATH_PRICE_GBR_MOSCOW_VTOR)

        # Санкт-Петербург новостройки
        elif city_id == 1:
            # Load data SPb
            data1 = pd.read_csv(SPB_DATA_NEW)
            data2 = pd.read_csv(SPB_DATA_SECONDARY)
            data = pd.concat([data1, data2], ignore_index=True)

            del data1
            del data2

            # Load KMean Clustering model
            kmeans = load(KMEANS_CLUSTERING_SPB_MAIN)

            # Load Price Models Spb Secondary
            gbr = load(PATH_PRICE_GBR_SPB_D)
            rf = load(PATH_PRICE_RF_SPB_D)
            lgbm = load(PATH_PRICE_LGBM_SPB_D)
            print("Pretrained models loaded! Spb")

            city_center_lon = 30.315239
            city_center_lat = 59.940735

        # # Санкт-Петербург вторичка
        # elif city_id == 1 and secondary == 1:
        #     data = pd.read_csv(SPB_DATA_SECONDARY)
        #     # Load KMean Clustering model
        #     kmeans = load(KMEANS_CLUSTERING_SPB_MAIN)
        #
        #     # Load Price Models Spb Secondary
        #     gbr = load(PATH_PRICE_GBR_SPB_VTOR)
        #     rf = load(PATH_PRICE_RF_SPB_VTOR)
        #     lgbm = load(PATH_PRICE_LGBM_SPB_VTOR)

        print("Initial shape: ", data.shape, flush=True)
        return data, kmeans, gbr, rf, lgbm, city_center_lon, city_center_lat

    # Call define function
    data, kmeans, gbr, rf, lgbm, city_center_lon, city_center_lat = define_city(city_id=city_id, secondary=secondary)

    ####################
    #                  #
    # PRICE PREDICTION #
    #                  #
    ####################

    # Calculate distance to city_center
    # No 1. Distance from city center in km

    # approximate radius of earth in km
    R = 6373.0

    to_city_center_distance = R * 2 * atan2(sqrt(sin((radians(latitude) - radians(city_center_lat)) / 2)
                                                 ** 2 + cos(radians(city_center_lat)) * cos(radians(city_center_lat))
                                                 * sin((radians(longitude) - radians(city_center_lon)) / 2) ** 2),
                                            sqrt(1 - (sin((radians(latitude) - radians(city_center_lat)) / 2)
                                                      ** 2 + cos(radians(city_center_lat)) * cos(radians(latitude))
                                                      * sin((radians(longitude) - radians(city_center_lon)) / 2) ** 2)))

    # Predict Cluster for current flat
    def define_cluster(km_model: KMeans, lon: float, lat: float):
        current_cluster = km_model.predict([[lon, lat]])
        return current_cluster

    current_cluster = define_cluster(km_model=kmeans, lon=longitude, lat=latitude)

    print("Current cluster is : ", current_cluster, flush=True)

    # Define current month
    mm_announce = now.month

    # Predict Price using gbr, rf, lgmb if not secondary
    def calculate_price(gbr_model: GradientBoostingRegressor, rf_model: RandomForestRegressor,
                        lgbm_model: LGBMRegressor, secondary: int):
        gbr_predicted_price, lgbm_pedicted_price, rf_predicted_price = 0, 0, 0
        # New
        gbr_predicted_price = np.expm1(gbr_model.predict(
            [[np.log1p(life_sq), np.log1p(to_city_center_distance), mm_announce, rooms, renovation, has_elevator,
              np.log1p(longitude), np.log1p(latitude),
              np.log1p(full_sq),
              np.log1p(kitchen_sq), time_to_metro, floor_first, floor_last, current_cluster, is_rented, rent_quarter,
              rent_year]]))
        print("Gbr predicted price ", gbr_predicted_price, flush=True)

        rf_predicted_price = np.expm1(rf_model.predict(
            [[np.log1p(life_sq), np.log1p(to_city_center_distance), mm_announce, rooms, renovation, has_elevator,
              np.log1p(longitude), np.log1p(latitude),
              np.log1p(full_sq),
              np.log1p(kitchen_sq), time_to_metro, floor_first, floor_last, current_cluster, is_rented, rent_quarter,
              rent_year]]))
        print("rf predicted price ", rf_predicted_price, flush=True)

        lgbm_pedicted_price = np.expm1(lgbm_model.predict(
            [[np.log1p(life_sq), np.log1p(to_city_center_distance), mm_announce, rooms, renovation, has_elevator,
              np.log1p(longitude), np.log1p(latitude),
              np.log1p(full_sq),
              np.log1p(kitchen_sq), time_to_metro, floor_first, floor_last, current_cluster, is_rented, rent_quarter,
              rent_year]]))
        print("Lgbm predicted price ", lgbm_pedicted_price, flush=True)

        # Calculate mean price value based on three algorithms
        price_main = (gbr_predicted_price + lgbm_pedicted_price + rf_predicted_price) / 3
        price = int(price_main[0])
        print("Predicted Price: ", price, flush=True)

        price_meter_sq = price / full_sq
        return price, price_meter_sq

    # Calculate price
    price, price_meter_sq = calculate_price(gbr_model=gbr, rf_model=rf, lgbm_model=lgbm, secondary=secondary)

    ####################
    #                  #
    # TERM CALCULATING #
    #                  #
    ####################

    # Remove price and term outliers (out of 3 sigmas)
    data = data[((np.abs(stats.zscore(data.price)) < 3) & (np.abs(stats.zscore(data.term)) < 3))]
    print("Outliers removed", flush=True)

    # data = pd.merge(data1, data2, on=list(data.columns), how='left')

    # Fill NaN if it appears after merging
    data[['term']] = data[['term']].fillna(data[['term']].mean())

    # Create subsample of flats from same cluster (from same "geographical" district)
    df_for_current_label = data[data.clusters == current_cluster[0]]

    del data
    print('Shape of current cluster: {0}'.format(df_for_current_label.shape), flush=True)

    # Check if subsample size have more than 3 samples
    if df_for_current_label.shape[0] < 3:
        answ = {'Price': price, 'Duration': 0, 'PLot': [{"x": 0, 'y': 0}], 'FlatsTerm': 0, "OOPS": 1}
        return answ

    # Drop flats which sold more than 600 days
    df_for_current_label = df_for_current_label[df_for_current_label.term <= 600]

    # Check if still enough samples
    if df_for_current_label.shape[0] > 1:

        def LinearReg_Term(data: pd.DataFrame):

            # Handle with negative term values
            # way no1
            data = data._get_numeric_data()  # <- this increase accuracy
            data[data < 0] = 0

            # way no2
            # data['profit'] = data['profit'] + 1 - data['profit'].min()

            # Log Transformation
            data['price'] = np.log1p(data['price_meter_sq'])
            data['profit'] = np.log1p(data['profit'])
            data['term'] = np.log1p(data['term'])

            # Create X and y for Linear Model training
            X = data[['profit', 'price_meter_sq']]
            y = data[['term']].values.ravel()

            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

            # Create LinearModel and fitting
            reg = LinearRegression().fit(X, y)
            print("Term linear regression fitted", flush=True)
            return reg

        def larger(p=0, percent=2):
            larger_prices = []
            for _ in range(15):
                new_p = p + p * percent / 100
                larger_prices.append(new_p)
                percent += 2
            return larger_prices

        # Create list of N larger prices than predicted
        list_of_larger_prices = larger(int(price_meter_sq))

        def smaller(p=0, percent=2):
            smaller_prices = []
            for _ in range(15):
                new_p = p - p * percent / 100
                smaller_prices.append(new_p)
                percent += 2
            return smaller_prices[::-1]

        # Create list of N smaller prices than predicted
        list_of_smaller_prices = smaller(int(price_meter_sq))

        # Create list of N prices: which are larger and smaller than predicted
        list_of_prices = list_of_smaller_prices + list_of_larger_prices
        list_of_prices = [int(i) for i in list_of_prices]



        # Call LinearReg on term
        reg = LinearReg_Term(df_for_current_label)

        def CalculateProfit(l: list):
            list_of_terms = []
            for i in l:
                profit = price_meter_sq / i
                # Calculate term based on profit for each price
                term_on_profit = np.expm1(reg.predict([[np.log1p(profit), np.log1p(i)]]))
                list_of_terms.append(term_on_profit)

            return list_of_terms

        # Calculating term for each price from generated list of prices based on associated profit -> returns list of terms
        list_of_terms = CalculateProfit(list_of_prices)

        del reg

        # Add links to flats
        # term_links = df_for_current_label.to_dict('record')

        list_of_terms = [int(i.tolist()[0]) for i in list_of_terms]
        print("Terms: ", list_of_terms, flush=True)

        prices = list_of_prices
        prices = [int(i * full_sq) for i in prices]
        print("Prices: ", prices, flush=True)

        # Define function for creating list of dicts
        # x=term, y=price
        # Example: [{'x': int, 'y': int}, {'x': int, 'y': int}]
        def createListOfDicts(terms: list, prices: list):
            list_of_dicts = []
            list_of_dicts += ({'price': int(prc), 'term': int(trm)} for prc, trm in zip(prices, terms))
            return list_of_dicts

        # Create list of dicts
        list_of_dicts = createListOfDicts(list_of_terms, prices)

        # Check if list not empty
        if len(list_of_dicts) <= 2:
            answ = {'Price': price, 'Duration': 0, 'PLot': [{"term": 0, 'price': 0}], 'FlatsTerm': 0, "OOPS": 1}
            return answ

        print('list_of_dicts: ', list_of_dicts, flush=True)

        # Define current flat with predicted price and initial term = minimal value from list of term
        current_flat = {'term': min(list_of_terms), 'price': price}

        # Iterate over the list of dicts and try to find suitable term based on prices values
        def find_term(l: list, current_flat: dict):
            term = 0
            if l[-1].get('price') > current_flat.get('price') > l[0].get('price'):
                for i in enumerate(l):
                    print(i)
                    if l[i[0]].get('price') <= current_flat.get('price') < l[i[0] + 1].get('price'):
                        print('!')
                        current_flat['term'] = int((l[i[0]].get('term') + l[i[0] + 1].get('term')) / 2)
                        term = int((l[i[0]].get('term') + l[i[0] + 1].get('term')) / 2)
                        break
                print("New term: ", current_flat, flush=True)
            return current_flat, term

        # Find actual term for current flat price
        if (list_of_dicts[-1].get('price') > current_flat.get('price') > list_of_dicts[0].get('price')) and \
                len(set([i['term'] for i in list_of_dicts])) > 2 and list_of_terms[-1] > list_of_terms[0]:
            print(
                'Number of unique term valus in list_of_dicts = {0}'.format(len(set([i['term'] for i in list_of_dicts]))),
            flush=True)
            current_flat, term = find_term(l=list_of_dicts, current_flat=current_flat)
        else:
            answ = {'Price': price, 'Duration': 0, 'PLot': [{"term": 0, 'price': 0}], 'FlatsTerm': 0, "OOPS": 1}
            return answ

        # Leave only unique pairs [{'x': int1, 'y': int2}, {'x': int3, 'y': int4}]
        def select_unique_term_price_pairs(list_of_dicts: list):
            terms = []
            result = []

            for i in range(1, len(list_of_dicts)):
                if (list_of_dicts[i].get('term') != list_of_dicts[i - 1].get('term')) and list_of_dicts[i].get(
                        'term') not in terms:
                    if list_of_dicts[i - 1].get('term') not in terms:
                        result.append(list_of_dicts[i - 1])
                        terms.append(list_of_dicts[i - 1].get('term'))
                    result.append(list_of_dicts[i])
                    terms.append(list_of_dicts[i].get('term'))
            return result

        if len(set([i['term'] for i in list_of_dicts])) > 2:
            list_of_dicts = select_unique_term_price_pairs(list_of_dicts)
        else:
            answ = {'Price': price, 'Duration': 0, 'PLot': [{"term": 0, 'price': 0}], 'FlatsTerm': 0, "OOPS": 1}
            return answ

        def check(l: list, current_flat):
            for i in l:
                if ((i['term'] == current_flat['term']) | (i['price'] == current_flat['price'])):
                    l.remove(i)
            l.append(current_flat)
            return sorted(l, key=lambda k: k['term'])


        # Check if all dict's keys and values in list are unique
        list_of_dicts = check(list_of_dicts, current_flat)
        print('Unique term values: ', len(set([i['term'] for i in list_of_dicts])), flush=True)


        # Check if final list have items in it, otherwise set parameter "OOPS" to 1
        oops = 1 if len(list_of_dicts) <= 2 else 0
        term = 0 if len(list_of_dicts) <= 2 else term

        answ = {'Price': price, 'Duration': term, 'PLot': list_of_dicts, 'FlatsTerm': 0, "OOPS": oops}
        print('ANSWER: \nprice: {0}, \nterm: {1}, \nlist_of_dicts: {2}, \noops: {3}'.format(
            price, term, list_of_dicts, oops), flush=True)
    else:
        print("!!! Warning !!! \n----------> Not enough data to plot", flush=True)
        answ = {'Price': price, 'Duration': 0, 'PLot': [{"term": 0, 'price': 0}], 'FlatsTerm': 0, "OOPS": 1}

    return answ
