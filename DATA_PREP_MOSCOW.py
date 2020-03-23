import pandas as pd
import numpy as np
from sklearn import preprocessing
import backports.datetime_fromisoformat as bck
from joblib import dump
import settings_local as SETTINGS
from scipy import stats
from sklearn.cluster import KMeans
import math as m

np.random.seed(42)

raw_data = SETTINGS.PATH_TO_SINGLE_CSV_FILES_MOSCOW
prepared_data = SETTINGS.DATA_MOSCOW
PATH_TO_TIME_MODEL = SETTINGS.MODEL_MOSCOW


def main_preprocessing():
    prices = pd.read_csv(raw_data + "prices.csv", names=[
        'id', 'price', 'changed_date', 'flat_id', 'created_at', 'updated_at'
    ], usecols=["price", "flat_id", 'created_at', 'changed_date', 'updated_at'])
    print("Unique flat id in prices: ", len(prices.flat_id.unique()))

    # Count number of price changing for each unique flat and SORT changed_date for each subgroup (group consist of one flat)
    prices['nums_of_changing'] = prices.sort_values(['changed_date'][-9:], ascending=True).groupby(['flat_id'])[
        "flat_id"].transform("count")
    # Group by falt_id and sort in ascending order for term counting
    # prices = prices.sort_values(['changed_date'][-9:],ascending=True).groupby('flat_id')

    # Keep just first date 
    prices = prices.drop_duplicates(subset='flat_id', keep="first")
    prices = prices[((prices['changed_date'].str.contains('2020')) | (prices['changed_date'].str.contains('2019')) | (
        prices['changed_date'].str.contains('2018')))]
    print("Unique flats Prices 2018/2019/2020 yearS: ", len(prices.flat_id.unique()))

    # Calculating selling term. TIME UNIT: DAYS
    prices['term'] = prices[['updated_at', 'changed_date']].apply(
        lambda row: (bck.date_fromisoformat(row['updated_at'][:-9])
                     - bck.date_fromisoformat(row['changed_date'][:-9])).days, axis=1)

    flats = pd.read_csv(raw_data + "flats.csv",
                        names=['id', 'full_sq', 'kitchen_sq', 'life_sq', 'floor', 'is_apartment',
                                                 'building_id', 'created_at',
                                                 'updated_at', 'offer_id', 'closed', 'rooms', 'image', 'resource_id',
                                                        'flat_type', 'is_rented', 'rent_quarter', 'rent_year', 'agency'],
                        usecols=["id", "full_sq",
                                                   "kitchen_sq",
                                                   "life_sq",
                                                   "floor", "is_apartment",
                                                   "building_id",
                                                   "closed", 'rooms', 'resource_id', 'flat_type', 'is_rented', 'rent_quarter',
                                                   'rent_year'
                                                   ],
                        true_values="t", false_values="f", header=0)

    # Replace all missed values in FLAT_TYPE with 'SECONDARY'
    flats.flat_type = flats['flat_type'].fillna('SECONDARY')

    # Replace all missed values in CLOSED with 'False'
    flats.closed = flats.closed.fillna(False)

    # Leave only VTORICHKA
    # flats = flats[flats.flat_type == 'SECONDARY']
    flats = flats.rename(columns={"id": "flat_id"})
    print("flats only secondary: ", flats.shape)

    buildings = pd.read_csv(raw_data + "buildings.csv",
                            names=["id", "max_floor", 'building_type_str', "built_year", "flats_count",
                                   "address", "renovation",
                                   "has_elevator",
                                   'longitude', 'latitude',
                                   "district_id",
                                   'created_at',
                                   'updated_at'],
                            usecols=["id", "max_floor", 'building_type_str', "built_year", "flats_count",
                                     "renovation",
                                     "has_elevator",
                                     "district_id", 'longitude', 'latitude',  # nominative scale
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
    print('time_to_metro: \n', time_to_metro.iloc[:5])
    # Merage prices and flats on flat_id
    prices_and_flats = pd.merge(prices, flats, on='flat_id', how="left")
    print("Prices + Flats: \n", prices_and_flats.shape)

    # Merge districts and buildings on district_id
    districts_and_buildings = pd.merge(districts, buildings, on='district_id', how='right')
    print("districts + buildings: \n", districts_and_buildings.shape)

    # Merge to one main DF on building_id
    df = pd.merge(prices_and_flats, districts_and_buildings, on='building_id', how='left')

    # Merge main DF and time_to_metro on building_id, fill the zero value with the mean value
    df = pd.merge(df, time_to_metro, on="building_id", how='left')
    # df[['time_to_metro']] = df[['time_to_metro']].apply(lambda x: x.fillna(x.mean()), axis=0)
    df.time_to_metro = df.time_to_metro.fillna(df.time_to_metro.mean())
    

    print('Data types: ', df.dtypes, flush=True)
    # Check if main DF constains null values
    # print(df.isnull().sum())

    # Drop all offers without important data
    df = df.dropna(subset=['full_sq'])
    
    print("is rented before", df[['is_rented']].head())

    # Replace missed "IS_RENTED" with 1 and convert bool -> int
    df.is_rented = df.is_rented.fillna(True)
    df.is_rented = df.is_rented.astype(int)
    print("is rented after", df[['is_rented']].head())

    df = df.fillna(0)
    

    # Replace missed value 'RENT_YEAR' with posted year
    # now = datetime.datetime.now()
    df.rent_year = df.rent_year.fillna(df.changed_date.apply(lambda x: x[:4]))

    # Replace missed value "RENT_QUARTER" with current quarter, when value was posted
    df.rent_quarter = df.rent_quarter.fillna(df.changed_date.apply(lambda x: x[5:7]))
    df.rent_quarter = df.rent_quarter.astype(int)
    df.rent_quarter = np.where(df.changed_date.apply(lambda x: int(x[5:7])) <= 12, 4, 4) 
    df.rent_quarter = np.where(df.changed_date.apply(lambda x: int(x[5:7])) <= 9, 3, df.rent_quarter)
    df.rent_quarter = np.where(df.changed_date.apply(lambda x: int(x[5:7])) <= 6, 2, df.rent_quarter)
    df.rent_quarter = np.where(df.changed_date.apply(lambda x: int(x[5:7])) <= 3, 1, df.rent_quarter)

    df = df.drop(['built_year', 'flats_count', 'district_id', 'name', 'transport_type'], axis=1)

    # Transform bool values to int
    df.has_elevator = df.has_elevator.astype(int)
    df.renovation = df.renovation.astype(int)
    df.is_apartment = df.is_apartment.astype(int)
    df.has_elevator = df.has_elevator.astype(int)
    df.renovation = df.renovation.astype(int)
    df.is_apartment = df.is_apartment.astype(int)
    df.rent_year = df.rent_year.astype(int)
    # Set values for floor_last/floor_first column: if floor_last/floor_first set 1, otherwise 0
    max_floor_list = df['max_floor'].tolist()
    df['floor_last'] = np.where(df['max_floor'] == df['floor'], 1, 0)
    df['floor_first'] = np.where(df['floor'] == 1, 1, 0)


    # Replace all negative values with zero
    num = df._get_numeric_data()
    num[num < 0] = 0

    '''
    df['X'] = df[['latitude', 'longitude']].apply(
        lambda row: (m.cos(row['latitude']) *
                     m.cos(row['longitude'])), axis=1)
    df['Y'] = df[['latitude', 'longitude']].apply(
        lambda row: (m.cos(row['latitude']) *
                     m.sin(row['longitude'])), axis=1)
    '''
    # Count price per meter square for each flat
    df['price_meter_sq'] = df[['price', 'full_sq']].apply(
        lambda row: (row['price'] /
                     row['full_sq']), axis=1)

    # Remove price and term outliers (out of 3 sigmas)
    df1 = df[(np.abs(stats.zscore(df.price)) < 3)]
    df2 = df[(np.abs(stats.zscore(df.term)) < 3)]

    clean_data = pd.merge(df1, df2, on=list(df.columns), how='left')
    '''
    print("Find optimal number of K means: ")
    Sum_of_squared_distances = []
    k_list = []
    K = range(100, 110)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(clean_data[['longitude', 'latitude']])
        Sum_of_squared_distances.append(km.inertia_)
        print(k)
        k_list.append(k)
    print(list(zip(k_list, Sum_of_squared_distances)))
    '''

    # Create df with SECONDARY flats
    clean_data_VTOR = clean_data[(clean_data.flat_type == 'SECONDARY')]

    # fit k-Means clustering on geo for SECONDARY flats
    kmeans_VTOR = KMeans(n_clusters=130, random_state=42).fit(clean_data_VTOR[['longitude', 'latitude']])
    dump(kmeans_VTOR, PATH_TO_TIME_MODEL + '/KMEAN_CLUSTERING_MOSCOW_VTOR.joblib')
    labels = kmeans_VTOR.labels_
    clean_data_VTOR['clusters'] = labels

    # Save .csv with SECONDARY flats
    print('Saving to new csv', clean_data_VTOR.shape[0], flush=True)
    clean_data_VTOR.to_csv(prepared_data + '/MOSCOW_VTOR.csv', index=None, header=True)


    # Create df with NEW flats
    clean_data_new_flats = clean_data[((clean_data.flat_type == 'NEW_FLAT')|(clean_data.flat_type == 'NEW_SECONDARY'))]

    # fit k-Means clustering on geo for NEW flats 
    kmeans_NEW_FLAT = KMeans(n_clusters=30, random_state=42).fit(clean_data_new_flats[['longitude', 'latitude']])
    dump(kmeans_NEW_FLAT, PATH_TO_TIME_MODEL + '/KMEAN_CLUSTERING_MOSCOW_NEW_FLAT.joblib')
    labels = kmeans_NEW_FLAT.labels_
    clean_data_new_flats['clusters'] = labels

    # Save .csv with NEW flats
    print('Saving to new csv', clean_data_new_flats.shape[0], flush=True)
    clean_data_new_flats.to_csv(prepared_data + '/MOSCOW_NEW_FLATS.csv', index=None, header=True)


if __name__ == '__main__':
    main_preprocessing()
