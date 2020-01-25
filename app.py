from flask import Flask, request, jsonify, render_template
from scipy import stats
from catboost import CatBoostRegressor, Pool
import xgboost
import psycopg2
import settings_local as SETTINGS
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import GradientBoostingRegressor
#from catboost import Pool, CatBoostRegressor
from joblib import dump, load
import math as m
import math
from datetime import datetime
import requests
import json
import pandas as pd
import statistics
import numpy as np
import math

PATH_TO_PRICE_MODEL = SETTINGS.MODEL + '/PriceModelGBR.joblib'

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
    has_elevator = float(request.args.get('has_elevator')) if request.args.get('has_elevator') is not None else None
    floor_first = float(request.args.get('floor_first')) if request.args.get('floor_first') is not None else None
    floor_last = float(request.args.get('floor_last')) if request.args.get('floor_last') is not None else None
    time_to_metro = float(request.args.get('time_to_metro')) if request.args.get('time_to_metro') is not None else None
    page = int(request.args.get('page')) if request.args.get('page') is not None else 1
    sort_type = int(request.args.get('sort_type')) if request.args.get('sort_type') is not None else 0

    print(latitude_from, latitude_to, longitude_from, longitude_to, flush=True)


    data_offers = pd.read_csv(SETTINGS.DATA + '/COORDINATES_OUTLIERS.csv')

    filter = (((data_offers.full_sq >= full_sq_from)&(data_offers.full_sq <= full_sq_to))&(data_offers.rooms == rooms) &
              ((data_offers.latitude >= latitude_from) & (data_offers.latitude <= latitude_to))
              & ((data_offers.longitude >= longitude_from) & (data_offers.longitude <= longitude_to)))
    data_offers = data_offers[filter]

    print('ds', data_offers.shape, flush=True)

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



    # PRICE

    '''

    X1 = data_offers[['renovation', 'has_elevator', 'longitude', 'latitude', 'full_sq', 'kitchen_sq',
                 'is_apartment', 'time_to_metro', 'floor_last', 'floor_first', 'X', 'Y']]
    data_offers["price"] = np.log1p(data_offers["price"])
    y1 = data_offers[['price']].values.ravel()
    print(X1.shape, y1.shape)

    clf = GradientBoostingRegressor(n_estimators=350, max_depth=4, verbose=10)
    clf.fit(X1, y1)
    '''
    gbr = load(PATH_TO_PRICE_MODEL)
    cat = load(SETTINGS.MODEL + '/PriceModelCatGradient.joblib')


    # Print GradientBoosting Regression features importance
    # feat_imp = pd.Series(gbr.feature_importances_, X1.columns).sort_values(ascending=False)
    # print(feat_imp)


    data_offers['pred_price'] = data_offers[['renovation', 'has_elevator', 'longitude', 'latitude', 'full_sq', 'kitchen_sq',
                                   'is_apartment', 'time_to_metro', 'floor_last', 'floor_first', 'X', 'Y', 'clusters']].apply(
        lambda row:
        int(((np.expm1(gbr.predict([[row.renovation, row.has_elevator, np.log1p(row.longitude), np.log1p(row.latitude), np.log1p(row.full_sq),
                                   np.log1p(row.kitchen_sq), row.is_apartment, row.time_to_metro, row.floor_last,
                                   row.floor_first, np.log1p(row.X), np.log1p(row.Y), row.clusters]]))+np.expm1(cat.predict([[row.renovation, row.has_elevator, np.log1p(row.longitude), np.log1p(row.latitude), np.log1p(row.full_sq),
                                   np.log1p(row.kitchen_sq), row.is_apartment, row.time_to_metro, row.floor_last,
                                   row.floor_first, np.log1p(row.X), np.log1p(row.Y), row.clusters]])))[0]/2)), axis=1)


    # Get Profit Offers using Outliers algorithm detection
    # outliers_alg = IsolationForest(contamination=0.2)


    # outliers_alg.fit(data_offers[['price', 'full_sq', 'clusters']])
    # outliers_it = data_offers[outliers_alg.predict(data_offers[['price', 'full_sq', 'clusters']]) == -1]
    # print('Outliers: ', outliers_it.shape[0], flush=True)
    # outliers_it['flat_id'] = outliers_it.index


    # data_offers = data_offers[data_offers.price < data_offers.pred_price]
    # data_offers['flat_id'] = data_offers.index
    print('Profitable offers using price prediction model: ', data_offers.shape[0])

    # data_offers = data_offers[data_offers.flat_id.isin(outliers_it.flat_id)]
    # print('After concat: ', data_offers.shape[0])
    data_offers['profit'] = data_offers[['pred_price', 'price']].apply(lambda row: ((row.pred_price*100/row.price)-100), axis=1)
    data_offers = data_offers[(data_offers.profit >= 5)]
    data_offers = data_offers.sort_values(by=['profit'], ascending=False)
    print(data_offers[['pred_price', "price"]].head())


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
                    (flat['id_building'],))
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
                    (flat['id_building'],))
        flat['address'] = cur.fetchone()[0]

        # print(flat['image'], flush=True)

        if type(flat['image']) != str:
            flat['image'] = None
        del flat['offer_id']
        del flat['id_building']
        del flat['time_to_metro']
        # print(flat, flush=True)

    conn.close()

    print('flats', len(flats), flush=True)

    # if math.isnan(mean_price):
    #     mean_price = None
    return jsonify({'flats': flats, 'page': page, 'max_page': max_page, 'count': flats_count})


@app.route('/map')
def map():
    # building_type_str = request.args.get('building_type_str')
    longitude = float(request.args.get('lng'))
    latitude = float(request.args.get('lat'))
    full_sq = float(request.args.get('full_sq'))
    kitchen_sq = float(request.args.get('kitchen_sq'))
    # life_sq = request.args.get('life_sq')
    is_apartment = int(request.args.get('is_apartment'))
    renovation = int(request.args.get('renovation'))
    has_elevator = int(request.args.get('has_elevator'))
    floor_first = int(request.args.get('floor_first'))
    floor_last = int(request.args.get('floor_last'))
    time_to_metro = int(request.args.get('time_to_metro'))
    X = (m.cos(latitude) * m.cos(longitude))
    Y = (m.cos(latitude) * m.sin(longitude))


    # Data
    data = pd.read_csv(SETTINGS.DATA + '/COORDINATES_Pred_Term.csv')
    print("Initial shape: ", data.shape, flush=True)

    # Load KMean Clustering model
    kmeans = load(SETTINGS.MODEL + '/KMEAN_CLUSTERING.joblib')

    # Predict Cluster for current flat
    current_label = kmeans.predict([[longitude, latitude]])
    print("Current label: ", current_label, flush=True)

    list_of_requested_params_price = [renovation, has_elevator, longitude, latitude, full_sq, kitchen_sq,
                                      is_apartment, time_to_metro, floor_last, floor_first, X, Y]

    # Create subsample of flats with same cluster label value (from same "geographical" district)
    df_for_current_label = data[data.clusters == current_label[0]]
    # Drop Price and Term Outliers using Z-Score
    df = df_for_current_label[(np.abs(stats.zscore(df_for_current_label.price)) < 3)]
    ds = df_for_current_label[(np.abs(stats.zscore(df_for_current_label.term)) < 3)]

    df_for_current_label = pd.merge(df, ds, on=list(ds.columns))

    # Create subsample according to the same(+-) size of the full_sq
    df_for_current_label = df_for_current_label[((df_for_current_label.full_sq >= full_sq - full_sq * 0.018) & (
    df_for_current_label.full_sq <= full_sq + full_sq * 0.018))]

    if df_for_current_label.shape[0] > 1:


        print("Current label dataframe shape: ", df_for_current_label.shape, flush=True)

        # Reducing skew in data using LogTransformation
        df_for_current_label["longitude"] = np.log1p(df_for_current_label["longitude"])
        df_for_current_label["latitude"] = np.log1p(df_for_current_label["latitude"])
        df_for_current_label["full_sq"] = np.log1p(df_for_current_label["full_sq"])
        df_for_current_label["kitchen_sq"] = np.log1p(df_for_current_label["kitchen_sq"])
        df_for_current_label["X"] = np.log1p(df_for_current_label["X"])
        df_for_current_label["Y"] = np.log1p(df_for_current_label["Y"])

        # Flats Features for GBR PRICE fitting
        X1 = df_for_current_label[['renovation', 'has_elevator', 'longitude', 'latitude', 'full_sq', 'kitchen_sq',
                                   'is_apartment', 'time_to_metro', 'floor_last', 'floor_first', 'X', 'Y']]

        # Log Transformation for target label (price) to reduce skew of value
        df_for_current_label["price"] = np.log1p(df_for_current_label["price"])
        y1 = df_for_current_label[['price']].values.ravel()

        # PRICE PREDICTION

        # GBR
        GBR_PRCIE = GradientBoostingRegressor(n_estimators=250, max_depth=8, verbose=5, max_features=3, random_state=42, learning_rate=0.07)
        print(X1.shape, y1.shape, flush=True)
        GBR_PRCIE.fit(X1, y1)
        price_gbr_pred = np.expm1(GBR_PRCIE.predict([list_of_requested_params_price]))

        print("Price gbr: ", price_gbr_pred, flush=True)

        CAT_PRICE = load(SETTINGS.MODEL + '/PriceModelCatGradient.joblib')
        price_cat_pred = np.expm1(CAT_PRICE.predict([[renovation, has_elevator, np.log1p(longitude), np.log1p(latitude), np.log1p(full_sq), np.log1p(kitchen_sq),
                                                      is_apartment, time_to_metro, floor_last, floor_first, np.log1p(X), np.log1p(Y), current_label]]))

        print("Price cat: ", price_cat_pred, flush=True)

        # Return real value of price (reverse Log Transformation)
        df_for_current_label["price"] = np.expm1(df_for_current_label["price"])

        # df_for_current_label["longitude"] = np.expm1(df_for_current_label["longitude"])
        # df_for_current_label["latitude"] = np.expm1(df_for_current_label["latitude"])
        # df_for_current_label["full_sq"] = np.expm1(df_for_current_label["full_sq"])
        # df_for_current_label["kitchen_sq"] = np.expm1(df_for_current_label["kitchen_sq"])
        # df_for_current_label["X"] = np.expm1(df_for_current_label["X"])
        # df_for_current_label["Y"] = np.expm1(df_for_current_label["Y"])

        # Count mean of Cat and GBR algorithms prediction
        price = (price_gbr_pred+price_cat_pred)/2
        #price = price_cat
        price = int(price[0])
        print("Predicted Price: ", price, flush=True)

        price_meter_sq = price / full_sq





        # TERM
        df_for_current_label = df_for_current_label[df_for_current_label.term <= 600]
        df_for_current_label = df_for_current_label[(np.abs(stats.zscore(df_for_current_label.price)) < 3)]
        # df_for_current_label = df_for_current_label[((df_for_current_label.price_meter_sq <= price_meter_sq+price_meter_sq*0.1)&
        #                                              (df_for_current_label.price_meter_sq >= price_meter_sq-price_meter_sq*0.1))]



        X_term = df_for_current_label[['renovation', 'has_elevator', 'longitude', 'latitude', 'price', 'full_sq', 'kitchen_sq',
                                       'is_apartment', 'time_to_metro', 'floor_last', 'floor_first', 'X', 'Y',
                                       'price_meter_sq']]

        # Reducing skew in data using LogTransformation

        df_for_current_label['price_meter_sq'] = np.log1p(df_for_current_label['price_meter_sq'])
        df_for_current_label['term'] = np.log1p(df_for_current_label['term'])

        y_term = df_for_current_label[['term']]


        # GBR
        list_of_requested_params_term = [renovation, has_elevator, np.log1p(longitude), np.log1p(latitude),
                                         np.log1p(full_sq), np.log1p(kitchen_sq), is_apartment, time_to_metro, floor_last,
                                         floor_first, np.log1p(X), np.log1p(Y),
                                         np.log1p(price_meter_sq), current_label]


        '''
        most_important_features = list(df_for_current_label_term.corr().term.sort_values(ascending=False).index)[1:4]
        print("Most important features for term prediction: ", most_important_features)
        '''

        GBR_TERM = GradientBoostingRegressor(n_estimators=350, max_depth=3, verbose=10, random_state=42, learning_rate=0.05)
        # from sklearn.linear_model import LinearRegression
        # GBR_TERM = LinearRegression()
        print(X_term.shape, y_term.shape, flush=True)

        GBR_TERM.fit(X_term, y_term)

        term_gbr_pred = np.expm1(GBR_TERM.predict([list_of_requested_params_term]))

        print("Term gbr: ", term_gbr_pred, flush=True)

        cat_term = CatBoostRegressor(random_state=42, l2_leaf_reg=1, learning_rate=0.05)
        #cat = CatBoostRegressor(iterations=100, max_depth=8, l2_leaf_reg=1)
        train_time = Pool(X_term, y_term)
        cat_term.fit(train_time, verbose=5)
        term_cat = np.expm1(cat_term.predict([list_of_requested_params_term]))
        print("Term cat: ", term_cat, flush=True)


        term = (term_cat+term_gbr_pred)/2
        # term = term_cat

        print("Predicted term: ", term)


        # term = term_gbr_pred
        term = int(term.item(0))




        # df_for_current_label = df_for_current_label[(df_for_current_label.term <= term+200)]

        # DATA FOR BUILDING PRICE-TIME CORRELATION GRAPHICS
        # Add new parameters: PREDICTED_PRICE and PROFIT
        gbr = load(PATH_TO_PRICE_MODEL)
        cat = load(SETTINGS.MODEL + '/PriceModelCatGradient.joblib')

        df_for_current_label['pred_price'] = df_for_current_label[
            ['renovation', 'has_elevator', 'longitude', 'latitude', 'full_sq', 'kitchen_sq',
             'is_apartment', 'time_to_metro', 'floor_last', 'floor_first', 'X', 'Y', 'clusters']].apply(
            lambda row:
            int(((np.expm1(gbr.predict([[row.renovation, row.has_elevator, row.longitude, row.latitude, row.full_sq,
                                         row.kitchen_sq, row.is_apartment, row.time_to_metro, row.floor_last,
                                         row.floor_first, row.X, row.Y, row.clusters]])) + np.expm1(
                cat.predict([[row.renovation, row.has_elevator, row.longitude, row.latitude, row.full_sq,
                              row.kitchen_sq, row.is_apartment, row.time_to_metro, row.floor_last,
                              row.floor_first, row.X, row.Y, row.clusters]])))[0] / 2)), axis=1)

        df_for_current_label['profit'] = df_for_current_label[['pred_price', 'price']].apply(
            lambda row: ((row.pred_price / row.price)), axis=1)




        X_term_new = df_for_current_label[
            ['renovation', 'has_elevator', 'longitude', 'latitude', 'price', 'full_sq', 'kitchen_sq',
             'is_apartment', 'time_to_metro', 'floor_last', 'floor_first', 'X', 'Y',
             'price_meter_sq', 'profit']]
        # X_term_new = sc.fit_transform(X_term_new)
        # df_for_current_label['term'] = np.log1p(df_for_current_label['term'])
        y_term_new = df_for_current_label[['term']]

        GBR_TERM_NEW = GradientBoostingRegressor(n_estimators=350, max_depth=3, verbose=10, random_state=42, learning_rate=0.05)
        GBR_TERM_NEW.fit(X_term_new, y_term_new)

        cat_new = CatBoostRegressor(random_state=42)
        train_time = Pool(X_term_new, y_term_new)
        cat_new.fit(train_time, verbose=5)


        # term = term_gbr_pred
        # term = int(term.item(0))

        # Create list of N prices: which are larger and smaller than predicted
        def larger(p=0):
            larger_prices = []
            percent = 2
            for _ in range(15):
                new_p = p + p * percent / 100
                larger_prices.append(new_p)
                percent += 2
            return larger_prices
        list_of_larger_prices = larger(price)

        def smaller(p=0):
            smaller_prices = []
            percent = 2
            for _ in range(15):

                new_p = p - p * percent / 100
                smaller_prices.append(new_p)
                percent += 2
            return smaller_prices[::-1]
        list_of_smaller_prices = smaller(price)


        list_of_prices = list_of_smaller_prices+list_of_larger_prices
        max_price_from_list = max(list_of_prices)
        #
        # print("Min: ", min_profit_from_list)
        # list_of_prices_new = []
        # for i in list_of_prices:
        #     list_of_prices_new.append(i + min_profit_from_list)
        # list_of_prices = list_of_prices_new

        min_profit = ((price * 100 /max_price_from_list) - 100)
        def fn(l: list):
            list_of_terms = []
            for i in l:
                profit = i/price
                print(i, profit)
                pred_term_profit = np.expm1(GBR_TERM_NEW.predict([[renovation, has_elevator, np.log1p(longitude),
                                                                   np.log1p(latitude), price, np.log1p(full_sq), np.log1p(kitchen_sq),
                                                                   is_apartment, time_to_metro, floor_last, floor_first, np.log1p(X), np.log1p(Y), price_meter_sq, profit]]))
                term_cat_profit = np.expm1(cat_new.predict([[renovation, has_elevator, np.log1p(longitude),
                                                             np.log1p(latitude), price, np.log1p(full_sq), np.log1p(kitchen_sq),
                                                             is_apartment, time_to_metro, floor_last, floor_first, np.log1p(X), np.log1p(Y), price_meter_sq, profit]]))


                term_profit = (pred_term_profit + term_cat_profit) / 2
                print("GBR & Cat: ", pred_term_profit, term_cat_profit, flush=True)
                print("Predicted term: ", term_profit, flush=True)
                list_of_terms.append(term_profit)
            return list_of_terms
        list_of_terms = fn(list_of_prices)
        min_term = min(list_of_terms)
        min_index = list_of_terms.index(min_term)




        # Count profit for different prices

        # Add links to flats
        term_links = df_for_current_label.to_dict('record')


        # Create list of term values from subsample of "same" flats
        # terms = df_for_current_label.term
        # terms = terms.tolist()
        list_of_terms = [i.tolist()[0] for i in list_of_terms]
        # list_of_terms = list_of_terms[::-1]
        # list_of_terms +=[term]

        print("Terms: ", list_of_terms, flush=True)

        # Create list of price values from subsample of "same" flats
        # prices = df_for_current_label.price
        # prices = prices.tolist()
        prices = list_of_prices
        # prices += [price]
        print("Prices: ", prices, flush=True)


        # Create list of dictionaries
        a = []
        a += ({'x': int(trm), 'y': prc} for trm, prc in zip(list_of_terms, prices))


        # Sort list by term
        a = [i for i in a if 0 < i.get('x') <600]
        a = sorted(a, key=lambda z: z['x'], reverse=False)
        def drop_duplicat(l: list):
            seen = set()
            new_l = []
            for d in l:
                # t = tuple(d)
                # print("d: ", d)
                if d.get('x') not in seen:
                    seen.add(d.get('x'))

                    # print(seen)
                    new_l.append(d)
            return new_l

        new_l = drop_duplicat(a)
        print("After drop duplicates: ", new_l, flush=True)

        b = {'x': int(term), 'y': int(price)}
        print("b: ", b, flush=True)

        if price > new_l[0].get('y'):
            for i in enumerate(new_l):
                print(i[0])
                if new_l[i[0]].get('y') < b.get('y') < new_l[i[0] + 1].get('y'):
                    b['x'] = int((new_l[i[0]].get('x')+new_l[i[0] + 1].get('x'))/2)
                    term = int((new_l[i[0]].get('x')+new_l[i[0] + 1].get('x'))/2)
                    break
            print("B_new: ", b, flush=True)


            def range_plot(l: list):
                new_a = [l[0]]
                for i in list(range(1, len(l))):
                    print(l[i])
                    if l[i].get('y') > l[i - 1].get('y'):
                        if l[i].get('y') > new_a[-1].get('y'):
                            new_a.append(l[i])
                return new_a
            new_a = range_plot(new_l)
            print('Sorted 0 :', new_a)



            print("B_new: ", b , flush=True)
            new_a += [b]
            new_a = sorted(new_a, key=lambda z: z['x'], reverse=False)

            print("Sorted; ", new_a, flush=True)
        else:
            new_a = [{'x': 0, 'y': 0}]



        '''
        # Sort list by price
        a = [i for i in a if 0 < i.get('x') < 600]
        a = sorted(a, key=lambda z: z['y'], reverse=False)
    
        b = {'x': int(term), 'y': int(price)}
        print("b: ", b, flush=True)
    
        for i in range(1, len(a)):
            if a[i - 1].get('y') < b.get('y') < a[i].get('y'):
                b['x'] = int((a[i].get('x') + a[i - 1].get('x')) / 2)
                print(a[i], a[i - 1], flush=True)
                term = int((a[i].get('x') + a[i - 1].get('x')) / 2)
                break
    
    
        def range_plot(l: list):
            new_a = [l[0]]
            for i in list(range(1, len(l))):
                print(l[i], flush=True)
                if l[i].get('y') > l[i - 1].get('y'):
                    if l[i].get('y') > new_a[-1].get('y'):
                        new_a.append(l[i])
            return new_a
    
        a = sorted(a, key=lambda z: z['x'], reverse=False)
        new_a = range_plot(a)
        print('Sorted 0 :', new_a, flush=True)
    
    
    
        new_a += [b]
        print(new_a, flush=True)
        def range_plot(l: list):
            new_a = [l[0]]
            for i in list(range(1, len(l))):
                print("\n", l[i], flush=True)
                if l[i].get('x') > l[i - 1].get('x'):
                    if l[i].get('x') > new_a[-1].get('x'):
                        new_a.append(l[i])
            return new_a
        new_a = sorted(new_a, key=lambda z: z['y'], reverse=False)
        new_a = range_plot(new_a)
        '''
        oops = 1 if len(new_a)<=1 else 0
        term = 0 if len(new_a)<=1 else term


        if new_a[-1].get('y') == price:
            new_a.append({'x': term+2, 'y': price})
        print(new_a, flush=True)
        new_point = new_a[-1]
        print('last item: ', new_point, flush=True)
        new_point_x = int(new_point.get('x'))
        new_point_y = int(new_point.get('y'))
        print("x from last item: ", new_point_x, flush=True)


        print('new x: ', new_point, flush=True)
        new_a.append({"x": new_point_x+2, 'y': new_point_y+new_point_y*0.02})
        print("Finally new_a: ", new_a, flush=True)
        # Check if enough data for plotting




        answ = jsonify({'Price': price, 'Duration': term, 'PLot': new_a, 'FlatsTerm': term_links, "OOPS": oops})
    else:
        answ = jsonify({'Price': 0, 'Duration': 0, 'PLot': [{"x": 0, 'y': 0}], 'FlatsTerm': 0, "OOPS":1})
    return answ
        # , 'Term': term})
    # return 'Price {0} \n Estimated Sale Time: {1} days'.format(price, term)


if __name__ == '__main__':
    app.run()
