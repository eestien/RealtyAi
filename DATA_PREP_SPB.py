import pandas as pd
import numpy as np
from sklearn import preprocessing
import backports.datetime_fromisoformat as bck
from joblib import dump
from sklearn.cluster import KMeans
import math as m
from scipy import stats
import settings_local as SETTINGS

RAW_DATA = SETTINGS.PATH_TO_SINGLE_CSV_FILES_SPB
PREPARED_DATA = SETTINGS.DATA_SPB
PATH_TO_MODELS = SETTINGS.MODEL_SPB

def main_preprocessing():

    prices = pd.read_csv(RAW_DATA + "prices.csv", names=[
        'id', 'price', 'changed_date', 'flat_id', 'created_at', 'updated_at'
    ], usecols=["price", "flat_id", 'created_at', 'changed_date', 'updated_at'])
    print("Unique flat id in prices: ", len(prices.flat_id.unique()))

    # Count number of price changing for each unique flat and SORT changed_date for each subgroup (group consist of one flat)
    prices['nums_of_changing'] = prices.sort_values(['changed_date'][-9:], ascending=True).groupby(['flat_id'])[
        "flat_id"].transform("count")
    # Group by falt_id and sort in ascending order for term counting
    # prices = prices.sort_values(['changed_date'][-9:],ascending=True).groupby('flat_id')

    # Rewrite UPDATED_AT column with last date
    prices['updated_at'] = prices.groupby('flat_id')['updated_at'].transform('last')

    # Keep just first date
    prices = prices.drop_duplicates(subset='flat_id', keep="first")
    prices = prices[((prices['changed_date'].str.contains('2020')) | (prices['changed_date'].str.contains('2019')) | (
        prices['changed_date'].str.contains('2018')))]

    print("Unique flats Prices 2018/2019/2020 yearS: ", len(prices.flat_id.unique()))

    # Calculating selling term. TIME UNIT: DAYS
    prices['term'] = prices[['updated_at', 'changed_date']].apply(
        lambda row: (bck.date_fromisoformat(row['updated_at'][:-9])
                     - bck.date_fromisoformat(row['changed_date'][:-9])).days, axis=1)

    flats = pd.read_csv(RAW_DATA + "flats.csv",
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
                                                   'rent_year'],
                        true_values="t", false_values="f", header=0)

    flats.closed = flats.closed.fillna(False)

    # Leave only VTORICHKA
    # flats = flats[flats.flat_type == 'SECONDARY']
    flats = flats.rename(columns={"id": "flat_id"})
    print("flats only secondary: ", flats.shape)

    buildings = pd.read_csv(RAW_DATA + "buildings.csv",
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

    districts = pd.read_csv(RAW_DATA + "districts.csv", names=['id', 'name', 'population', 'city_id',
                                                               'created_at', 'updated_at', 'prefix'],
                            usecols=["name", 'id'],
                            true_values="t", false_values="f", header=0)

    districts = districts.rename(columns={"id": "district_id"})
    buildings = buildings.rename(columns={"id": "building_id"})

    time_to_metro = pd.read_csv(RAW_DATA + "time_metro_buildings.csv",
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
    # df = df.fillna(0)

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
    

    # Transform bool values to int
    df.has_elevator = df.has_elevator.astype(int)
    df.renovation = df.renovation.astype(int)
    df.is_apartment = df.is_apartment.astype(int)
    df.has_elevator = df.has_elevator.astype(int)
    df.renovation = df.renovation.astype(int)
    df.is_apartment = df.is_apartment.astype(int)
    df.rent_year = df.rent_year.astype(int)

    df = df.drop(['built_year', 'flats_count', 'district_id', 'name', 'transport_type'], axis=1)


    

    # Set values for floor_last/floor_first column: if floor_last/floor_first set 1, otherwise 0
    max_floor_list = df['max_floor'].tolist()
    df['floor_last'] = np.where(df['max_floor'] == df['floor'], 1, 0)
    df['floor_first'] = np.where(df['floor'] == 1, 1, 0)



    # Replace all negative values with zero
    num = df._get_numeric_data()
    num[num < 0] = 0

    # Check if data contains only Moscow offers
    df = df[((df['latitude'].astype('str').str.contains('59.'))|(df['latitude'].astype('str').str.contains('60.')))]
    # df['X'] = df[['latitude', 'longitude']].apply(
    #     lambda row: (m.cos(row['latitude']) *
    #                  m.cos(row['longitude'])), axis=1)
    # df['Y'] = df[['latitude', 'longitude']].apply(
    #     lambda row: (m.cos(row['latitude']) *
    #                  m.sin(row['longitude'])), axis=1)

    # Count price per meter square for each flat
    df['price_meter_sq'] = df[['price', 'full_sq']].apply(
        lambda row: (row['price'] /
                     row['full_sq']), axis=1)


    # Create df with SECONDARY flats
    df_vtor = df[(df.flat_type == 'SECONDARY')]

    # fit k-Means clustering on geo for SECONDARY flats
    kmeans_vtor = KMeans(n_clusters=60, random_state=42).fit(df_vtor[['longitude', 'latitude']])

    dump(kmeans_vtor, PATH_TO_MODELS + '/KMEAN_CLUSTERING_SPB_VTOR.joblib')
    labels = kmeans_vtor.labels_
    df_vtor['clusters'] = labels

    # Save .csv with SECONDARY flats
    df_vtor.to_csv(PREPARED_DATA + '/SPB_VTOR.csv', index=None, header=True)


    # Create df with NEW flats
    df_new_flats = df[((df.flat_type == 'NEW_FLAT') | (df.flat_type == 'NEW_SECONDARY'))]

    # fit k-Means clustering on geo for NEW flats
    kmeans_NEW_FLAT = KMeans(n_clusters=20, random_state=42).fit(df_new_flats[['longitude', 'latitude']])

    dump(kmeans_NEW_FLAT, PATH_TO_MODELS + '/KMEAN_CLUSTERING_NEW_FLAT_SPB.joblib')
    labels = kmeans_NEW_FLAT.labels_
    df_new_flats['clusters'] = labels

    # Save .csv with NEW flats
    df_new_flats.to_csv(PREPARED_DATA + '/SPB_NEW_FLATS.csv', index=None, header=True)

if __name__ == '__main__':
    main_preprocessing()