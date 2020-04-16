import pandas as pd
import numpy as np
from sklearn import preprocessing
import backports.datetime_fromisoformat as bck
from joblib import dump
import datetime
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time, ciso8601
from sklearn.cluster import KMeans
import math as m
from scipy import stats
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import settings_local as SETTINGS

RAW_DATA = SETTINGS.PATH_TO_SINGLE_CSV_FILES_SPB
PREPARED_DATA = SETTINGS.DATA_SPB
PATH_TO_MODELS = SETTINGS.MODEL_SPB


# TODO:
class MainPreprocessing():
    """Create class for data preprocessing"""
    def __init__(self):
        """Initialize class"""
        pass

    def load_and_merge(self, raw_data: str):
        prices = pd.read_csv(raw_data + "prices.csv", names=[
            'id', 'price', 'changed_date', 'flat_id', 'created_at', 'updated_at'
        ], usecols=["price", "flat_id", 'created_at', 'changed_date', 'updated_at'])

        # Count number of price changing for each unique flat and SORT changed_date for each subgroup (group consist of one flat)
        # prices['nums_of_changing'] = prices.sort_values(['changed_date'][-9:], ascending=True).groupby(['flat_id'])[
        #     "flat_id"].transform("count")
        # Group by falt_id and sort in ascending order for term counting
        # prices = prices.sort_values(['changed_date'][-9:],ascending=True).groupby('flat_id')

        # Rewrite UPDATED_AT column with last date
        prices['updated_at'] = prices.groupby('flat_id')['updated_at'].transform('last')

        # Keep just first date
        prices = prices.drop_duplicates(subset='flat_id', keep="first")
        prices = prices[((prices['changed_date'].str.contains('2020')) | (prices['changed_date'].str.contains('2019')) | (
            prices['changed_date'].str.contains('2018')))]


        # Calculating selling term. TIME UNIT: DAYS
        prices['term'] = prices[['updated_at', 'changed_date']].apply(
            lambda row: (bck.date_fromisoformat(row['updated_at'][:-9])
                         - bck.date_fromisoformat(row['changed_date'][:-9])).days, axis=1)

        flats = pd.read_csv(raw_data + "flats.csv",
                            names=['id', 'full_sq', 'kitchen_sq', 'life_sq', 'floor', 'is_apartment',
                                   'building_id', 'created_at',
                                   'updated_at', 'offer_id', 'closed', 'rooms', 'image', 'resource_id',
                                   'flat_type', 'is_rented', 'rent_quarter', 'rent_year', 'agency', 'renovation_type', 'windows_view'],
                            usecols=["id", "full_sq",
                                                       "kitchen_sq",
                                                       "life_sq",
                                                       "floor", "is_apartment",
                                                       "building_id", 'offer_id',
                                                       "closed", 'rooms', 'resource_id', 'flat_type', 'is_rented', 'rent_quarter',
                                                       'rent_year', 'renovation_type', 'windows_view'],
                            true_values="t", false_values="f", header=0)

        flats.closed = flats.closed.fillna(False)


        flats = flats.rename(columns={"id": "flat_id"})

        # Encoding categorical parameters:
        # renovation type
        flats.renovation_type = flats.renovation_type.fillna(0)
        flats.renovation_type = flats.renovation_type.map(
            {'Без ремонта': 0, 0: 1, 'Косметический': 1, 'Евроремонт': 2, 'Дизайнерский': 3}).astype(int)

        # windows view
        flats.windows_view = flats.windows_view.fillna(0)
        flats.windows_view = flats.windows_view.map(
            {'Во двор': 0, 0: 1, 'На улицу и двор': 1, 'На улицу': 2}).astype(int)

        buildings = pd.read_csv(raw_data + "buildings.csv",
                                names=["id", "max_floor", 'building_type_str', "built_year", "flats_count",
                                       "address", "renovation",
                                       "has_elevator",
                                       'longitude', 'latitude',
                                       "district_id",
                                       'created_at',
                                       'updated_at', 'schools_500m', 'schools_1000m', 'kindergartens_500m',
                                       'kindergartens_1000m', 'clinics_500m', 'clinics_1000m', 'shops_500m',
                                       'shops_1000m'],
                                usecols=["id", "max_floor", 'building_type_str', "built_year", "flats_count",
                                         "renovation",
                                         "has_elevator",
                                         "district_id", 'longitude', 'latitude', 'schools_500m', 'schools_1000m', 'kindergartens_500m',
                                       'kindergartens_1000m', 'clinics_500m', 'clinics_1000m', 'shops_500m',
                                       'shops_1000m'  # nominative scale
                                         ],
                                true_values="t", false_values="f", header=0)


        districts = pd.read_csv(raw_data + "districts.csv", names=['id', 'name', 'population', 'city_id',
                                                                   'created_at', 'updated_at', 'prefix'],
                                usecols=["name", 'id'],
                                true_values="t", false_values="f", header=0)


        districts = districts.rename(columns={"id": "district_id"})
        buildings = buildings.rename(columns={"id": "building_id"})

        time_to_metro = pd.read_csv(raw_data + "time_metro_buildings.csv",
                                    names=['id', 'building_id', 'metro_id', 'time_to_metro',
                                           'transport_type', 'created_at', 'updated_at'],
                                    usecols=["building_id", "time_to_metro", "transport_type"], header=0)

        # Sort time_to_metro values
        time_to_metro = time_to_metro[time_to_metro['transport_type'] == "ON_FOOT"].sort_values('time_to_metro',
                                                                                                ascending=True)

        # Keep just shortest time to metro
        time_to_metro = time_to_metro.drop_duplicates(subset='building_id', keep="first")

        # Merage prices and flats on flat_id
        prices_and_flats = pd.merge(prices, flats, on='flat_id', how="left")

        # Merge districts and buildings on district_id
        districts_and_buildings = pd.merge(districts, buildings, on='district_id', how='right')

        # Merge to one main DF on building_id
        df = pd.merge(prices_and_flats, districts_and_buildings, on='building_id', how='left')

        # Merge main DF and time_to_metro on building_id, fill the zero value with the mean value
        df = pd.merge(df, time_to_metro, on="building_id", how='left')

        # df[['time_to_metro']] = df[['time_to_metro']].apply(lambda x: x.fillna(x.mean()), axis=0)

        df.time_to_metro = df.time_to_metro.fillna(df.time_to_metro.mean())


        # Drop all offers without important data
        df = df.dropna(subset=['full_sq'])


        # Replace missed "IS_RENTED" with 1 and convert bool -> int
        df.is_rented = df.is_rented.fillna(True)
        df.is_rented = df.is_rented.astype(int)


        # Replace missed value 'RENT_YEAR' with posted year
        # now = datetime.datetime.now()
        # df.rent_year = df.rent_year.fillna(df.changed_date.apply(lambda x: x[:4]))
        df.rent_year = df.rent_year.fillna(0)
        df.is_rented = df.is_rented.astype(int)

        # Replace missed value "RENT_QUARTER" with current quarter, when value was posted
        df.rent_quarter = df.rent_quarter.fillna(0)
        df.is_rented = df.is_rented.astype(int)
        # df.rent_quarter = df.rent_quarter.fillna(df.changed_date.apply(lambda x: x[5:7]))
        # df.rent_quarter = df.rent_quarter.astype(int)
        # df.rent_quarter = np.where(df.changed_date.apply(lambda x: int(x[5:7])) <= 12, 4, 4)
        # df.rent_quarter = np.where(df.changed_date.apply(lambda x: int(x[5:7])) <= 9, 3, df.rent_quarter)
        # df.rent_quarter = np.where(df.changed_date.apply(lambda x: int(x[5:7])) <= 6, 2, df.rent_quarter)
        # df.rent_quarter = np.where(df.changed_date.apply(lambda x: int(x[5:7])) <= 3, 1, df.rent_quarter)

        df = df.fillna(0)

        # Transform bool values to int
        df.has_elevator = df.has_elevator.astype(int)
        df.renovation = df.renovation.astype(int)
        df.is_apartment = df.is_apartment.astype(int)
        df.has_elevator = df.has_elevator.astype(int)
        df.renovation = df.renovation.astype(int)
        df.is_apartment = df.is_apartment.astype(int)
        df.renovation_type = df.renovation_type.astype(int)
        df.windows_view = df.windows_view.astype(int)

        df = df.drop(['built_year', 'flats_count', 'district_id', 'name', 'transport_type'], axis=1)




        # Set values for floor_last/floor_first column: if floor_last/floor_first set 1, otherwise 0
        df['floor_last'] = np.where(df['max_floor'] == df['floor'], 1, 0)
        df['floor_first'] = np.where(df['floor'] == 1, 1, 0)



        # Replace all negative values with zero
        num = df._get_numeric_data()
        num[num < 0] = 0

        # Check if data contains only SPB offers
        df = df[((df['latitude'].astype('str').str.contains('59.'))|(df['latitude'].astype('str').str.contains('60.')))]

        # Count price per meter square for each flat
        df['price_meter_sq'] = df[['price', 'full_sq']].apply(
            lambda row: (row['price'] /
                         row['full_sq']), axis=1)
        print("Loaded df: {0}".format(df.shape))
        return df

    def new_features(self, data: pd.DataFrame(), full_sq_corridor_percent: float, price_corridor_percent: float, part_data: int, K_clusters: int):
        now = datetime.datetime.now()
        df = data
        # No 1. Distance from city center
        SPB_center_lon = 30.315768
        SPB_center_lat = 59.938976
        df['to_center'] = abs(SPB_center_lon - df['longitude']) + abs(SPB_center_lat - df['latitude'])

        # No 2. Fictive(for futher offer value calculating): yyyy_announc, mm_announc - year and month when flats were announced on market
        df['yyyy_announce'] = df['changed_date'].str[2:4].astype('int64')
        df['mm_announce'] = df['changed_date'].str[5:7].astype('int64')

        # No 3. Number of offers were added calculating by months and years
        df['all_offers_added_in_month'] = df.groupby(['yyyy_announce', 'mm_announce'])["flat_id"].transform("count")

        # No 4. Convert changed_date and updated_at to unix timestamp. Convert only yyyy-mm-dd hh
        df['open_date_unix'] = df['changed_date'].apply(
            lambda row: int(time.mktime(ciso8601.parse_datetime(row[:-3]).timetuple())))
        df['close_date_unix'] = df['updated_at'].apply(
            lambda row: int(time.mktime(ciso8601.parse_datetime(row[:-3]).timetuple())))

        # Take just part of data
        if part_data:
            df = df.iloc[:len(df) // part_data]

        # Calculate number of "similar" flats which were on market when each closed offer was closed.
        df['was_opened'] = [np.sum((df['open_date_unix'] < close_time) & (df['close_date_unix'] >= close_time) &
                                   (df['rooms'] == rooms) &
                                   ((df['full_sq'] <= full_sq * (1 + full_sq_corridor_percent / 100)) & (
                                           df['full_sq'] >= full_sq * (1 - full_sq_corridor_percent / 100)))) for
                            close_time, rooms, full_sq in
                            zip(df['close_date_unix'], df['rooms'], df['full_sq'])]


        def add_fictive_rows(data: pd.DataFrame(), K_clusters: int):
            data_cols = list(data.columns)
            fict_data = {}
            for i in data_cols:
                fict_data[i] = [j for j in range(K_clusters)]
            return pd.DataFrame(fict_data)

        df1 = add_fictive_rows(data=df, K_clusters=K_clusters)

        df = pd.concat([df, df1], axis=0, ignore_index=True)

        # df['rooms'] = np.where(df['rooms'] > 6, 0, df['rooms'])
        df['mm_announce'] = np.where(((0 >= df['mm_announce']) | (df['mm_announce'] > 12)), 1,
                                     df['mm_announce'])
        df['yyyy_announce'] = np.where(((17 >= df['yyyy_announce']) | (df['yyyy_announce'] > 20)), 19,
                                      df['yyyy_announce'])


        # Transform data types
        df.rooms = df.rooms.astype(int)
        df.mm_announce = df.mm_announce.astype(int)
        df.yyyy_announce = df.yyyy_announce.astype(int)

        return df

    def clustering(self, data: pd.DataFrame(), path_kmeans_models: str, K_clusters: int):
        # fit k-Means clustering on geo for SECONDARY flats

        data.longitude = data.longitude.fillna(data.longitude.mode()[0])
        data.latitude = data.latitude.fillna(data.latitude.mode()[0])
        kmeans = KMeans(n_clusters=K_clusters, random_state=42).fit(data[['longitude', 'latitude']])
        dump(kmeans, path_kmeans_models + '/KMEANS_CLUSTERING_SPB_MAIN.joblib')
        labels = kmeans.labels_
        data['clusters'] = labels

        data.clusters = data.clusters.astype(int)

        # Create dummies from cluster
        # df_clusters = pd.get_dummies(data, prefix='cluster_', columns=['clusters'])
        # data = pd.merge(data, df_clusters, how='left')

        return data

    # Transform some features (such as mm_announce, rooms, clusters) to dummies
    def to_dummies(self, data: pd.DataFrame):
        df_mm_announce = pd.get_dummies(data, prefix='mm_announce_', columns=['mm_announce'])
        # df_rooms = pd.get_dummies(data, prefix='rooms_', columns=['rooms'])
        # df = pd.merge(df_mm_announce, df_rooms, how='left')

        df_year_announce = pd.get_dummies(data=data, prefix='yyyy_announce_', columns=['yyyy_announce'])
        df = pd.merge(df_mm_announce, df_year_announce, how='left')

        df.drop(df.tail(K_CLUSTERS).index,inplace=True)

        df = df.dropna(subset=['full_sq'])
        return df

    def train_price_model(self, data: pd.DataFrame):

        df = data
        df = df[((np.abs(stats.zscore(df.price)) < 3) & (np.abs(stats.zscore(df.term)) < 3) & (
                    np.abs(stats.zscore(df.full_sq)) < 3))]

        df = df[['price', 'full_sq', 'kitchen_sq', 'life_sq', 'rooms', 'is_apartment',
                 'renovation', 'has_elevator',
                 'time_to_metro', 'floor_first', 'floor_last',
                 'is_rented', 'rent_quarter',
                 'rent_year', 'to_center', 'was_opened', 'mm_announce__1',
                 'mm_announce__2', 'mm_announce__3', 'mm_announce__4',
                 'mm_announce__5', 'mm_announce__6', 'mm_announce__7', 'mm_announce__8', 'mm_announce__9',
                 'mm_announce__10', 'mm_announce__11', 'mm_announce__12', 'yyyy_announce__18',
                 'yyyy_announce__19', 'yyyy_announce__20',
                 'clusters']]


        # df = df[['price', 'full_sq', 'kitchen_sq', 'life_sq', 'is_apartment',
        #          'renovation', 'has_elevator',
        #          'time_to_metro', 'floor_first', 'floor_last',
        #          'is_rented', 'rent_quarter',
        #          'rent_year', 'to_center', 'was_opened', 'mm_announce__1',
        #          'mm_announce__2', 'mm_announce__3', 'mm_announce__4',
        #          'mm_announce__5', 'mm_announce__6', 'mm_announce__7', 'mm_announce__8', 'mm_announce__9',
        #          'mm_announce__10', 'mm_announce__11', 'mm_announce__12', 'rooms', 'yyyy_announce__18',
        #          'yyyy_announce__19', 'yyyy_announce__20',
        #          'cluster__0', 'cluster__1',
        #          'cluster__2', 'cluster__3', 'cluster__4', 'cluster__5', 'cluster__6', 'cluster__7', 'cluster__8',
        #          'cluster__9', 'cluster__10',
        #          'cluster__11', 'cluster__12', 'cluster__13', 'cluster__14', 'cluster__15', 'cluster__16',
        #          'cluster__17', 'cluster__18', 'cluster__19',
        #          'cluster__20', 'cluster__21', 'cluster__22', 'cluster__23', 'cluster__24',
        #          'cluster__25', 'cluster__26', 'cluster__27', 'cluster__28', 'cluster__29', 'cluster__30',
        #          'cluster__31', 'cluster__32',
        #          'cluster__33', 'cluster__34', 'cluster__35', 'cluster__36', 'cluster__37', 'cluster__38',
        #          'cluster__39', 'cluster__40',
        #          'cluster__41', 'cluster__42', 'cluster__43', 'cluster__44', 'cluster__45', 'cluster__46',
        #          'cluster__47', 'cluster__48', 'cluster__49', 'cluster__50', 'cluster__51', 'cluster__52',
        #          'cluster__53', 'cluster__54', 'cluster__55',
        #          'cluster__56', 'cluster__57', 'cluster__58', 'cluster__59']]

        # Save leaved columns to variable
        columns = list(df.columns)


        # Log transformation
        df["full_sq"] = np.log1p(df["full_sq"])
        df["life_sq"] = np.log1p(df["life_sq"])
        df["kitchen_sq"] = np.log1p(df["kitchen_sq"])
        df["price"] = np.log1p(df["price"])

        # Create features - predictors
        X = df.drop(['price'], axis=1)

        # Target feature
        y = df[['price']].values.ravel()

        # Split for train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        # Define Gradient Boosting Machine model
        lgbm_model = LGBMRegressor(objective='regression',
                                  learning_rate=0.07,
                                  n_estimators=1250, max_depth=10, min_child_samples=1, verbose=0)

        # Train GBR on train dataset
        lgbm_model.fit(X_train, y_train)
        lgbm_preds = lgbm_model.predict(X_test)
        print('The R2_score of the Gradient boost is', r2_score(y_test, lgbm_preds), flush=True)
        print('RMSE is: \n', mean_squared_error(y_test, lgbm_preds), flush=True)

        # Train GBR on full dataset
        lgbm_model.fit(X, y)
        return lgbm_model, columns

    def calculate_profit(self, data: pd.DataFrame, price_model: GradientBoostingRegressor, list_of_columns: list):

        data.closed = data.closed.fillna(False)
        data_closed = data[data.closed == True]
        opened_data = data[data.closed==False]

        data_closed['pred_price'] = data_closed[list_of_columns].apply(
            lambda row:
            int(np.expm1(price_model.predict(
                [[np.log1p(row.full_sq), np.log1p(row.kitchen_sq), np.log1p(row.life_sq), row.rooms, row.is_apartment,
                  row.renovation, row.has_elevator, row.time_to_metro, row.floor_first, row.floor_last,
                  row.is_rented, row.rent_quarter, row.rent_year, row.to_center, row.was_opened, row.mm_announce__1,
                  row.mm_announce__2, row.mm_announce__3, row.mm_announce__4, row.mm_announce__5, row.mm_announce__6,
                  row.mm_announce__7, row.mm_announce__8, row.mm_announce__9,
                  row.mm_announce__10, row.mm_announce__11, row.mm_announce__12,
                  row.yyyy_announce__18, row.yyyy_announce__19, row.yyyy_announce__20,
                  row.clusters]]))[0]), axis=1)

        # data_closed['pred_price'] = data_closed[list_of_columns].apply(
        #     lambda row:
        #     int(np.expm1(price_model.predict(
        #         [[np.log1p(row.full_sq), np.log1p(row.kitchen_sq), np.log1p(row.life_sq), row.is_apartment,
        #           row.renovation, row.has_elevator, row.time_to_metro, row.floor_first, row.floor_last,
        #           row.is_rented, row.rent_quarter, row.rent_year, row.to_center, row.was_opened, row.mm_announce__1,
        #           row.mm_announce__2, row.mm_announce__3, row.mm_announce__4, row.mm_announce__5, row.mm_announce__6,
        #           row.mm_announce__7, row.mm_announce__8, row.mm_announce__9,
        #           row.mm_announce__10, row.mm_announce__11, row.mm_announce__12, row.rooms,
        #           row.yyyy_announce__18, row.yyyy_announce__19, row.yyyy_announce__20,
        #           row.cluster__0, row.cluster__1, row.cluster__2, row.cluster__3, row.cluster__4, row.cluster__5,
        #           row.cluster__6, row.cluster__7,
        #           row.cluster__8, row.cluster__9, row.cluster__10,
        #           row.cluster__11, row.cluster__12, row.cluster__13, row.cluster__14, row.cluster__15, row.cluster__16,
        #           row.cluster__17, row.cluster__18, row.cluster__19,
        #           row.cluster__20, row.cluster__21, row.cluster__22, row.cluster__23, row.cluster__24,
        #           row.cluster__25, row.cluster__26, row.cluster__27, row.cluster__28, row.cluster__29, row.cluster__30,
        #           row.cluster__31, row.cluster__32,
        #           row.cluster__33, row.cluster__34, row.cluster__35, row.cluster__36, row.cluster__37, row.cluster__38,
        #           row.cluster__39, row.cluster__40,
        #           row.cluster__41, row.cluster__42, row.cluster__43, row.cluster__44, row.cluster__45, row.cluster__46,
        #           row.cluster__47, row.cluster__48, row.cluster__49, row.cluster__50, row.cluster__51, row.cluster__52,
        #           row.cluster__53, row.cluster__54, row.cluster__55,
        #           row.cluster__56, row.cluster__57, row.cluster__58, row.cluster__59]]))[0]), axis=1)

        data_closed['profit'] = data_closed[['pred_price', 'price']].apply(
            lambda row: ((row.pred_price * 100 / row.price) - 100), axis=1)

        # Handle negative profit values
        # data_closed['profit'] = data_closed['profit'] + 1 - data_closed['profit'].min()

        # Concat opened and closed
        data = pd.concat([data_closed, opened_data], axis=0, ignore_index=True)
        data.profit = data.profit.fillna(0)

        return data

    def secondary_flats(self, data: pd.DataFrame(), path_to_save_data: str):

        # Create df with SECONDARY flats
        df_VTOR = data[(data.flat_type == 'SECONDARY')]

        # Save .csv with SECONDARY flats
        print('Saving SECONDARY flats to csv', df_VTOR.shape[0], flush=True)
        df_VTOR.to_csv(path_to_save_data + '/SPB_VTOR.csv', index=None, header=True)

    def new_flats(self, data: pd.DataFrame(), path_to_save_data: str):

        # Create df with NEW flats
        df_new_flats = data[((data.flat_type == 'NEW_FLAT') | (data.flat_type == 'NEW_SECONDARY'))]

        # Save .csv with NEW flats
        print('Saving NEW flats to csv', df_new_flats.shape[0], flush=True)
        df_new_flats.to_csv(path_to_save_data + '/SPB_NEW_FLATS.csv', index=None, header=True)


if __name__ == '__main__':

    full_sq_corridor_percent = 1.5
    price_corridor_percent = 1.5
    K_CLUSTERS = 60

    # Create obj MainPreprocessing
    mp = MainPreprocessing()

    # Load data
    print('_'*10, "SPB", "_"*10)
    print("Load data...", flush=True)
    df = mp.load_and_merge(raw_data=RAW_DATA)
    # df = df.iloc[:1000]

    # Generate new features
    print("Generate new features...", flush=True)
    features_data = mp.new_features(data=df, full_sq_corridor_percent=full_sq_corridor_percent,
                                    price_corridor_percent=price_corridor_percent, part_data=False, K_clusters=K_CLUSTERS)

    # Define clusters
    print("Defining clusters based on lon, lat...")
    cl_data = mp.clustering(features_data, path_kmeans_models=PATH_TO_MODELS, K_clusters=K_CLUSTERS)

    # Create dummies variables
    print("Transform to dummies...", flush=True)
    cat_data = mp.to_dummies(cl_data)

    # Train price model
    print("Train price model...", flush=True)
    price_model, list_columns = mp.train_price_model(data=cat_data)

    # Calculate profit for each flat
    print("Calculating profit for each closed offer in dataset...", flush=True)
    test = mp.calculate_profit(data=cat_data, price_model=price_model, list_of_columns=list_columns)

    # Create separate files for secondary flats
    mp.secondary_flats(data=test, path_to_save_data=PREPARED_DATA)

    # Create sepatare files for new flats
    mp.new_flats(data=test, path_to_save_data=PREPARED_DATA)