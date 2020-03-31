from flask import Flask, request, jsonify, render_template
from scipy import stats
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import LogisticRegression
import psycopg2
import settings_local as SETTINGS
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
#from catboost import Pool, CatBoostRegressor
from joblib import dump, load
import math as m
from math import sqrt
from datetime import datetime
import requests
import json
import pandas as pd
import statistics
import numpy as np
import math


PATH_PRICE_GBR_MOSCOW_VTOR = SETTINGS.MODEL_MOSCOW + '/PriceModel_MOSCOW_Vtor_GBR.joblib'
PATH_PRICE_RF_MOSCOW_VTOR = SETTINGS.MODEL_MOSCOW + '/PriceModel_MOSCOW_Vtor_RF.joblib'
PATH_PRICE_LGBM_MOSCOW_VTOR = SETTINGS.MODEL_MOSCOW + '/PriceModel_MOSCOW_Vtor_LGBM.joblib'
PATH_PRICE_GBR_SPB_VTOR = SETTINGS.MODEL_SPB + '/PriceModel_SPB_Vtor_GBR.joblib'
PATH_PRICE_RF_SPB_VTOR = SETTINGS.MODEL_SPB + '/PriceModel_SPB_Vtor_RF.joblib'
PATH_PRICE_LGBM_SPB_VTOR = SETTINGS.MODEL_SPB + '/PriceModel_SPB_Vtor_LGBM.joblib'

PATH_PRICE_GBR_MOSCOW_NEW = SETTINGS.MODEL_MOSCOW + '/PriceModel_MOSCOW_NEW_GBR.joblib'
PATH_PRICE_RF_MOSCOW_NEW = SETTINGS.MODEL_MOSCOW + '/PriceModel_MOSCOW_NEW_RF.joblib'
PATH_PRICE_LGBM_MOSCOW_NEW = SETTINGS.MODEL_MOSCOW + '/PriceModel_MOSCOW_NEW_LGBM.joblib'
PATH_PRICE_GBR_SPB_NEW = SETTINGS.MODEL_SPB + '/PriceModel_SPB_NEW_GBR.joblib'
PATH_PRICE_RF_SPB_NEW = SETTINGS.MODEL_SPB + '/PriceModel_SPB_NEW_RF.joblib'
PATH_PRICE_LGBM_SPB_NEW = SETTINGS.MODEL_SPB + '/PriceModel_SPB_NEW_LGBM.joblib'
app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/api/mean/', methods=['GET'])
def mean():
    full_sq_from = float(request.args.get('full_sq_from'))
    full_sq_to = float(request.args.get('full_sq_to'))
    latitude_from = float(request.args.get('latitude_from'))
    latitude_to = float(request.args.get('latitude_to'))
    longitude_from = float(request.args.get('longitude_from'))
    longitude_to = float(request.args.get('longitude_to'))
    rooms = float(request.args.get('rooms'))
    price_from = float(request.args.get('price_from')) if request.args.get('price_from') is not None else None
    price_to = float(request.args.get('price_to')) if request.args.get('price_to') is not None else None
    building_type_str = float(request.args.get('building_type_str')) if request.args.get(
        'building_type_str') is not None else None
    kitchen_sq = float(request.args.get('kitchen_sq')) if request.args.get('kitchen_sq') is not None else None
    life_sq = float(request.args.get('life_sq')) if request.args.get('life_sq') is not None else None
    renovation = float(request.args.get('renovation')) if request.args.get('renovation') is not None else None
    has_elevator = float(request.args.get('elevator')) if request.args.get('elevator') is not None else None
    floor_first = float(request.args.get('floor_first')) if request.args.get('floor_first') is not None else None
    floor_last = float(request.args.get('floor_last')) if request.args.get('floor_last') is not None else None
    time_to_metro = float(request.args.get('time_to_metro')) if request.args.get('time_to_metro') is not None else None
    page = int(request.args.get('page')) if request.args.get('page') is not None else 1
    sort_type = int(request.args.get('sort_type')) if request.args.get('sort_type') is not None else 0
    city_id = int(request.args.get('city_id')) if request.args.get('city_id') is not None else 0


    # Initialize DF
    data_offers = pd.DataFrame()

    # Set paths to data and price prediction models, depending on city:  0 = Moscow, 1 = Spb
    if city_id == 0:
        data_offers = pd.read_csv(SETTINGS.DATA_MOSCOW + '/MOSCOW.csv')
        data_offers = data_offers[data_offers.flat_type == 'SECONDARY']
        gbr = load(PATH_PRICE_GBR_MOSCOW_VTOR)
        rf = load(PATH_PRICE_RF_MOSCOW_VTOR)
        lgbm = load(PATH_PRICE_LGBM_MOSCOW_VTOR)
    elif city_id == 1:
        data_offers = pd.read_csv(SETTINGS.DATA_SPB + '/SPB.csv')
        data_offers = data_offers[data_offers.flat_type == 'SECONDARY']
        gbr = load(PATH_PRICE_GBR_SPB_VTOR)
        rf = load(PATH_PRICE_RF_SPB_VTOR)
        lgbm = load(PATH_PRICE_LGBM_SPB_VTOR)

    # Apply filtering flats in database on parameters: full_sq range, coordinates scope
    filter = (((data_offers.full_sq >= full_sq_from)&(data_offers.full_sq <= full_sq_to))&(data_offers.rooms == rooms) &
              ((data_offers.latitude >= latitude_from) & (data_offers.latitude <= latitude_to))
              & ((data_offers.longitude >= longitude_from) & (data_offers.longitude <= longitude_to)))
    data_offers = data_offers[filter]

    # Use only open offers
    data_offers = data_offers[data_offers['closed'] == False]

    print('Actual offers: ', data_offers.head(), flush=True)

    if time_to_metro != None:
        data_offers = data_offers[(data_offers.time_to_metro <= time_to_metro)]
    if rooms != None:
        data_offers = data_offers[data_offers.rooms == rooms]
    if building_type_str != None:
        data_offers = data_offers[data_offers.building_type_str == building_type_str]
    if kitchen_sq != None:
        data_offers = data_offers[(data_offers.kitchen_sq >= kitchen_sq - 1) & (data_offers.kitchen_sq <= kitchen_sq + 1)]
    if life_sq != None:
        data_offers = data_offers[(data_offers.life_sq >= life_sq - 5) & (data_offers.life_sq <= life_sq + 5)]
    if renovation != None:
        data_offers = data_offers[data_offers.renovation == renovation]
    if has_elevator != None:
        data_offers = data_offers[data_offers.has_elevator == has_elevator]
    if floor_first != None:
        data_offers = data_offers[data_offers.floor_first == 0]
    if floor_last != None:
        data_offers = data_offers[data_offers.floor_last == 0]
    if price_from != None:
        data_offers = data_offers[data_offers.price >= price_from]
    if price_to != None:
        data_offers = data_offers[data_offers.price <= price_to]

    # PRICE PREDICTION
    data_offers['pred_price'] = data_offers[['life_sq', 'rooms', 'renovation', 'has_elevator', 'longitude', 'latitude', 'full_sq', 'kitchen_sq',
                                          'time_to_metro', 'floor_last', 'floor_first', 'clusters']].apply(
        lambda row:
        int((np.expm1(rf.predict([[np.log1p(row.life_sq), row.rooms, row.renovation, row.has_elevator, np.log1p(row.longitude),
                                    np.log1p(row.latitude), np.log1p(row.latitude),
                                       np.log1p(row.kitchen_sq), row.time_to_metro, row.floor_first, row.floor_last, row.clusters]]))+np.expm1(lgbm.predict([[np.log1p(row.life_sq), row.rooms, row.renovation, row.has_elevator, np.log1p(row.longitude),
                                    np.log1p(row.latitude), np.log1p(row.latitude),
                                       np.log1p(row.kitchen_sq), row.time_to_metro, row.floor_first, row.floor_last, row.clusters]])))[0]/2), axis=1)

    # Calculate the profitability for each flat knowing current and the price that our model predicted
    data_offers['profit'] = data_offers[['pred_price', 'price']].apply(lambda row: ((row.pred_price*100/row.price)-100), axis=1)

    # Set threshold for showing profitable offers
    data_offers = data_offers[(data_offers.profit >= 5)]
    data_offers = data_offers.sort_values(by=['profit'], ascending=False)
    print("Profitable offers: ", data_offers[['pred_price', "price", 'profit']].head(), flush=True)

    flats = data_offers.to_dict('record')

    flats_count = len(flats)
    flats_page_count = 10
    # max_page = math.ceil(len(flats) / flats_page_count)
    max_page = 1
    page = page if page <= max_page else 1
    '''
    if sort_type == 0:
        flats = sorted(flats, key=lambda x: x['price'])[(page - 1) * flats_page_count:page * flats_page_count]
    else:
        flats = sorted(flats, key=lambda x: x['price'])[(page - 1) * flats_page_count:page * flats_page_count]
    '''
    conn = psycopg2.connect(host=SETTINGS.host, dbname=SETTINGS.name, user=SETTINGS.user, password=SETTINGS.password)
    cur = conn.cursor()
    for flat in flats:
        # print(flat.keys(), flush=True)
        cur.execute("select metro_id, time_to_metro from time_metro_buildings where building_id=%s",
                    (flat['building_id'],))
        metros_info = cur.fetchall()
        flat['metros'] = []
        for metro in metros_info:
            cur.execute("select name from metros where id=%s", (metro[0],))
            flat['metros'].append({'station': cur.fetchone()[0], 'time_to_metro': metro[1]})

        if flat['resource_id'] == 0:
            flat['link'] = 'https://realty.yandex.ru/offer/' + str(flat['offer_id'])
        else:
            flat['link'] = 'https://www.cian.ru/sale/flat/' + str(flat['offer_id'])

        cur.execute("select address from buildings where id=%s",
                    (flat['building_id'],))
        flat['address'] = cur.fetchone()[0]

        if type(flat['image']) != str:
            flat['image'] = None
        del flat['offer_id']
        del flat['building_id']
        del flat['time_to_metro']

    conn.close()

    print('flats', len(flats), flush=True)

    # if math.isnan(mean_price):
    #     mean_price = None
    return jsonify({'flats': flats, 'page': page, 'max_page': max_page, 'count': flats_count})


@app.route('/map')
def map():
    longitude = float(request.args.get('lng'))
    rooms = int(request.args.get('rooms'))
    latitude = float(request.args.get('lat'))
    full_sq = float(request.args.get('full_sq'))
    kitchen_sq = float(request.args.get('kitchen_sq'))
    life_sq = float(request.args.get('life_sq'))
    renovation = int(request.args.get('renovation'))
    secondary = int(request.args.get('secondary'))
    has_elevator = int(request.args.get('elevator'))
    floor_first = int(request.args.get('floor_first'))
    floor_last = int(request.args.get('floor_last'))
    time_to_metro = int(request.args.get('time_to_metro'))
    is_rented = int(request.args.get('is_rented')) if request.args.get('is_rented') is not None else 0
    rent_year = int(request.args.get('rent_year')) if request.args.get('rent_year') is not None else 0
    rent_quarter = int(request.args.get('rent_quarter')) if request.args.get('rent_quarter') is not None else 0
    city_id = int(request.args.get('city_id')) if request.args.get('city_id') is not None else 0

    # initialize dataframe
    data = pd.DataFrame()
    kmeans = 0
    gbr = 0
    lgbm = 0
    rf = 0
    print("Params: City id: {0}, is secondary: {1}".format(city_id, secondary), flush=True)

    # 0 = Moscow, 1 = Spb
    # Москва новостройки
    if city_id == 0 and secondary ==0:
        # Load data Moscow New flats
        data = pd.read_csv(SETTINGS.DATA_MOSCOW + '/MOSCOW_NEW_FLATS.csv')

        # Load KMean Clustering model
        kmeans = load(SETTINGS.MODEL_MOSCOW + '/KMEAN_CLUSTERING_MOSCOW_NEW_FLAT.joblib')

        # Load Price Models Moscow Secondary
        gbr = load(PATH_PRICE_GBR_MOSCOW_NEW)
        rf = load(PATH_PRICE_RF_MOSCOW_NEW)
        lgbm = load(PATH_PRICE_LGBM_MOSCOW_NEW)

    # Москва вторичка
    elif city_id == 0 and secondary == 1:
        # Load data Moscow secondary
        data = pd.read_csv(SETTINGS.DATA_MOSCOW + '/MOSCOW_VTOR.csv')

        # Load KMean Clustering model
        kmeans = load(SETTINGS.MODEL_MOSCOW + '/KMEAN_CLUSTERING_MOSCOW_VTOR.joblib')

        # Load Price Models Moscow Secondary
        gbr = load(PATH_PRICE_GBR_MOSCOW_VTOR)
        rf = load(PATH_PRICE_GBR_MOSCOW_VTOR)
        lgbm = load(PATH_PRICE_GBR_MOSCOW_VTOR)

    # Санкт-Петербург новостройки
    elif city_id == 1 and secondary == 0:
        # Load data SPb New Flats
        data = pd.read_csv(SETTINGS.DATA_SPB + '/SPB_NEW_FLATS.csv')

        # Load KMean Clustering model
        kmeans = load(SETTINGS.MODEL_SPB + 'KMEAN_CLUSTERING_NEW_FLAT_SPB.joblib')

        # Load Price Models Spb Secondary
        gbr = load(PATH_PRICE_GBR_SPB_NEW)
        rf = load(PATH_PRICE_RF_SPB_NEW)
        lgbm = load(PATH_PRICE_LGBM_SPB_NEW)

    # Санкт-Петербург вторичка
    elif city_id == 1 and secondary == 1:
        data = pd.read_csv(SETTINGS.DATA_SPB + '/SPB_VTOR.csv')
        # Load KMean Clustering model
        kmeans = load(SETTINGS.MODEL_SPB + '/KMEAN_CLUSTERING_SPB_VTOR.joblib')

        # Load Price Models Spb Secondary
        gbr = load(PATH_PRICE_GBR_SPB_VTOR)
        rf = load(PATH_PRICE_RF_SPB_VTOR)
        lgbm = load(PATH_PRICE_LGBM_SPB_VTOR)

    print("Initial shape: ", data.shape, flush=True)

    ####################
    #                  #
    # PRICE PREDICTION #
    #                  #
    ####################

    # Predict Cluster for current flat
    current_cluster = kmeans.predict([[longitude, latitude]])
    print("Current cluster is : ", current_cluster, flush=True)

    # Predict Price using gbr, rf, lgmb if not secondary 
    if secondary == 0:
        gbr_predicted_price = np.expm1(gbr.predict([[np.log1p(life_sq), rooms, renovation, has_elevator, np.log1p(longitude), np.log1p(latitude),
                               np.log1p(full_sq),
                               np.log1p(kitchen_sq), time_to_metro, floor_first, floor_last, current_cluster, is_rented, rent_quarter, rent_year]]))
        print("Gbr predicted price NOT secondary: ", gbr_predicted_price, flush=True)

        rf_predicted_price = np.expm1(rf.predict([[np.log1p(life_sq), rooms, renovation, has_elevator, np.log1p(longitude), np.log1p(latitude),
                               np.log1p(full_sq),
                               np.log1p(kitchen_sq), time_to_metro, floor_first, floor_last, current_cluster, is_rented, rent_quarter, rent_year]]))
        print("rf predicted price NOT secondary: ", rf_predicted_price, flush=True)

        lgbm_pedicted_price = np.expm1(lgbm.predict([[np.log1p(life_sq), rooms, renovation, has_elevator, np.log1p(longitude), np.log1p(latitude),
                               np.log1p(full_sq),
                               np.log1p(kitchen_sq), time_to_metro, floor_first, floor_last, current_cluster, is_rented, rent_quarter, rent_year]]))
        print("Lgbm predicted price NOT secondary: ", lgbm_pedicted_price, flush=True)

    # Predict Price using gbr, rf, lgmb if secondary 
    elif secondary == 1:
        gbr_predicted_price = np.expm1(gbr.predict([[np.log1p(life_sq), rooms, renovation, has_elevator, np.log1p(longitude), np.log1p(latitude),
                               np.log1p(full_sq), np.log1p(kitchen_sq), time_to_metro, floor_first, floor_last, current_cluster]]))
        print("Gbr predicted price secondary: ", gbr_predicted_price, flush=True)

        rf_predicted_price = np.expm1(rf.predict([[np.log1p(life_sq), rooms, renovation, has_elevator, np.log1p(longitude), np.log1p(latitude), np.log1p(full_sq),
                                       np.log1p(kitchen_sq), time_to_metro, floor_first, floor_last, current_cluster]]))
        print("rf predicted price secondary: ", rf_predicted_price, flush=True)

        lgbm_pedicted_price = np.expm1(lgbm.predict([[np.log1p(life_sq), rooms, renovation, has_elevator, np.log1p(longitude), np.log1p(latitude), np.log1p(full_sq),
                                       np.log1p(kitchen_sq), time_to_metro, floor_first, floor_last, current_cluster]]))
        print("Lgbm predicted price secondary: ", lgbm_pedicted_price, flush=True)


    # Calculate mean price value based on three algorithms
    price_main = (gbr_predicted_price+lgbm_pedicted_price+rf_predicted_price)/ 3
    price = int(price_main[0])
    print("Predicted Price: ", price, flush=True)

    price_meter_sq = price / full_sq

    ####################
    #                  #
    # TERM CALCULATING #
    #                  #
    ####################
    
    # Remove price and term outliers (out of 3 sigmas)
    data1 = data[(np.abs(stats.zscore(data.price)) < 3)]
    data2 = data[(np.abs(stats.zscore(data.term)) < 3)]


    data = pd.merge(data1, data2, on=list(data.columns), how='left')

    # Fill NaN if it appears after merging 
    data[['term']] = data[['term']].fillna(data[['term']].mean())


    # Create subsample of flats from same cluster (from same "geographical" district)
    df_for_current_label = data[data.clusters == current_cluster[0]]

    # Check if subsample size have more than 3 samples
    if df_for_current_label.shape[0] < 3:
        answ = jsonify({'Price': price, 'Duration': 0, 'PLot': [{"x": 0, 'y': 0}], 'FlatsTerm': 0, "OOPS": 1})
        return answ

    # Create SUB Classes KMeans clustering based on size of subsample
    n = int(sqrt(df_for_current_label.shape[0]))
    kmeans_sub = KMeans(n_clusters=n, random_state=42).fit(df_for_current_label[['full_sq', 'life_sq', 'kitchen_sq', 'time_to_metro', 'longitude', 'latitude', 'renovation']])#, 'nums_of_changing']])

    # Set new column equals to new SUBclusters values
    labels = kmeans_sub.labels_
    df_for_current_label['SUB_cluster'] = labels

    SUB_cluster = kmeans_sub.predict([[full_sq, life_sq, kitchen_sq, time_to_metro, longitude, latitude, renovation]])
    # print(df_for_current_label.SUB_cluster.unique(), flush=True)


    df_for_current_label = df_for_current_label[df_for_current_label.SUB_cluster == SUB_cluster[0]]

    if len(df_for_current_label) < 2: 
        df_for_current_label = data[data.clusters == current_cluster[0]]

    # Create new feature: number of flats in each SUBcluster
    # df_for_current_label['num_of_flats_in_SUB_cluster'] = df_for_current_label.groupby(['SUB_cluster'])["SUB_cluster"].transform("count")

    # Drop Outliers using Z-Score / 15-85 quartiles
    # price outliers removing
    df_for_current_label = df_for_current_label[df_for_current_label.price.between(df_for_current_label.price.quantile(.15), df_for_current_label.price.quantile(.85))]
    #  term outliers removing
    df_for_current_label = df_for_current_label[df_for_current_label.term.between(df_for_current_label.term.quantile(.15), df_for_current_label.term.quantile(.85))]
    # squares outliers removing
    df_for_current_label = df_for_current_label[df_for_current_label.full_sq.between(df_for_current_label.full_sq.quantile(.15), df_for_current_label.full_sq.quantile(.85))]
    df_for_current_label = df_for_current_label[df_for_current_label.life_sq.between(df_for_current_label.life_sq.quantile(.15), df_for_current_label.life_sq.quantile(.85))]
    df_for_current_label = df_for_current_label[df_for_current_label.kitchen_sq.between(df_for_current_label.kitchen_sq.quantile(.15), df_for_current_label.kitchen_sq.quantile(.85))]
    

    # Calculate price for each flat in SubSample based on price prediction models we have trained
    if secondary == 0:
        df_for_current_label['pred_price'] = df_for_current_label[
            ['life_sq', 'rooms', 'renovation', 'has_elevator', 'longitude', 'latitude', 'full_sq', 'kitchen_sq',
             'time_to_metro', 'floor_last', 'floor_first', 'clusters', 'is_rented', 'rent_quarter', 'rent_year']].apply(
            lambda row:
            int(((np.expm1(rf.predict([[np.log1p(row.life_sq), row.rooms, row.renovation, row.has_elevator,
                                       np.log1p(row.longitude), np.log1p(row.latitude), np.log1p(row.full_sq),
                                       np.log1p(row.kitchen_sq), row.time_to_metro, row.floor_last, row.floor_first,
                                       row.clusters, row.is_rented, row.rent_quarter, row.rent_year]]))) + 
                 (np.expm1(lgbm.predict([[np.log1p(row.life_sq), row.rooms, row.renovation, row.has_elevator,
                                       np.log1p(row.longitude), np.log1p(row.latitude), np.log1p(row.full_sq),
                                       np.log1p(row.kitchen_sq), row.time_to_metro, row.floor_last, row.floor_first,
                                       row.clusters, row.is_rented, row.rent_quarter, row.rent_year]]))))[0] / 2), axis=1)
        pass

    if secondary == 1:
        df_for_current_label['pred_price'] = df_for_current_label[
            ['life_sq', 'rooms', 'renovation', 'has_elevator', 'longitude', 'latitude', 'full_sq', 'kitchen_sq',
             'time_to_metro', 'floor_last', 'floor_first', 'clusters']].apply(
            lambda row:
            int(((np.expm1(rf.predict([[np.log1p(row.life_sq), row.rooms, row.renovation, row.has_elevator,
                                       np.log1p(row.longitude), np.log1p(row.latitude), np.log1p(row.full_sq),
                                       np.log1p(row.kitchen_sq), row.time_to_metro, row.floor_last, row.floor_first,
                                       row.clusters]]))) + 
                 (np.expm1(lgbm.predict([[np.log1p(row.life_sq), row.rooms, row.renovation, row.has_elevator,
                                       np.log1p(row.longitude), np.log1p(row.latitude), np.log1p(row.full_sq),
                                       np.log1p(row.kitchen_sq), row.time_to_metro, row.floor_last, row.floor_first,
                                       row.clusters]]))))[0] / 2), axis=1)
        pass
    # Calculate the profitability for each flat knowing the price for which the flat was sold and the price that
    # our model predicted
    df_for_current_label['profit'] = df_for_current_label[['pred_price', 'price']].apply(
        lambda row: ((row.pred_price / row.price)), axis=1)
    print(df_for_current_label[['profit', 'price', 'pred_price']].head(), flush=True)

    # Drop flats which sold more than 600 days
    df_for_current_label = df_for_current_label[df_for_current_label.term <= 600]

    # Check if still enough samples
    if df_for_current_label.shape[0] > 1:

        term = 0
        # Log Transformation
        # df_for_current_label['profit'] = np.log1p(df_for_current_label['profit'])
        df_for_current_label['price'] = np.log1p(df_for_current_label['price'])

        # Create X and y for Linear Model training
        X = df_for_current_label[['profit', 'price']]
        y = df_for_current_label[['term']].values.ravel()
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        # Create LinearModel and fitting
        reg = LinearRegression().fit(X, y)

        def larger(p=0, percent=2):
            larger_prices = []
            for _ in range(15):
                new_p = p + p * percent / 100
                larger_prices.append(new_p)
                percent += 2
            return larger_prices

        # Create list of N larger prices than predicted
        list_of_larger_prices = larger(price)

        def smaller(p=0, percent=2):
            smaller_prices = []
            for _ in range(15):
                new_p = p - p * percent / 100
                smaller_prices.append(new_p)
                percent += 2
            return smaller_prices[::-1]

        # Create list of N smaller prices than predicted
        list_of_smaller_prices = smaller(price)

        # Create list of N prices: which are larger and smaller than predicted
        list_of_prices = list_of_smaller_prices+list_of_larger_prices

        def CalculateProfit(l: list):
            list_of_terms = []
            for i in l:
                profit = i / price
                # Calculate term based on profit for each price
                term_on_profit = reg.predict([[profit, np.log1p(i)]])
                print("Predicted term is {0} based on {1} profit: ".format(term_on_profit, profit), flush=True)
                list_of_terms.append(term_on_profit)
            return list_of_terms

        # Calculating term for each price from generated list of prices based on associated profit -> returns list of terms
        list_of_terms = CalculateProfit(list_of_prices)


        # Add links to flats
        term_links = df_for_current_label.to_dict('record')

        list_of_terms = [i.tolist()[0] for i in list_of_terms]
        print("Terms: ", list_of_terms, flush=True)

        prices = list_of_prices
        print("Prices: ", prices, flush=True)

        # Create list of dictionaries
        a = []
        a += ({'x': int(trm), 'y': prc} for trm, prc in zip(list_of_terms, prices))

        # Sort list by term
        a = [i for i in a if 0 < i.get('x') <600]
        a = sorted(a, key=lambda z: z['x'], reverse=False)
        print("First sort by term: ", a, flush=True)

        def drop_duplicates_term(l: list):
            seen = set()
            new_l = []
            for item in l:
                if item.get('x') not in seen:
                    seen.add(item.get('x'))
                    new_l.append(item)
            return new_l

        # Drop duplicated terms, because FrontEnd waits only unique values
        new_list_of_dicts = drop_duplicates_term(a)
        print("After drop term duplicates: ", new_list_of_dicts, flush=True)

        b = {'x': int(term), 'y': int(price)}
        print("Predicted raw term, and exact price: ", b, flush=True)

        def drop_duplicates_price(l: list):
            seen_prices = set()
            new_list_of_prices = []
            for item in l:
                if item.get('y') not in seen_prices:
                    seen_prices.add(item.get('y'))
                    new_list_of_prices.append(item)
            return new_list_of_prices

        # Set term based on price
        if len(new_list_of_dicts) > 1:
            # Check that our predicted price lies in the price range for which we calculated the term
            if new_list_of_dicts[-1].get('y') > price > new_list_of_dicts[0].get('y'):
                for i in enumerate(new_list_of_dicts):
                    if new_list_of_dicts[i[0]].get('y') < b.get('y') < new_list_of_dicts[i[0] + 1].get('y'):
                        b['x'] = int((new_list_of_dicts[i[0]].get('x')+new_list_of_dicts[i[0] + 1].get('x'))/2)
                        term = int((new_list_of_dicts[i[0]].get('x')+new_list_of_dicts[i[0] + 1].get('x'))/2)
                        break
                print("New term: ", b, flush=True)
            else:
                answ = jsonify({'Price': price, 'Duration': 0, 'PLot': [{"x": 0, 'y': 0}], 'FlatsTerm': 0, "OOPS": 1})
                return answ

        # Drop duplicates from price, because FrontEnd waits only unique values
        new_a = drop_duplicates_price(new_list_of_dicts)
        print('Drop price duplicates:', new_a, flush=True)

        new_a.insert(0, b)

        new_a = sorted(new_a, key=lambda z: z['x'], reverse=False)
        print("Sorted finally : ", new_a, flush=True)

        # Check if final list have items in it, otherwise set parameter "OOPS" to 1
        oops = 1 if len(new_a)<=1 else 0
        term = 0 if len(new_a)<=1 else term

        answ = jsonify({'Price': price, 'Duration': term, 'PLot': new_a, 'FlatsTerm': term_links, "OOPS": oops})
    else:
        print("Not enough data to plot", flush=True)
        answ = jsonify({'Price': price, 'Duration': 0, 'PLot': [{"x": 0, 'y': 0}], 'FlatsTerm': 0, "OOPS":1})
    return answ


if __name__ == '__main__':
    app.run()
