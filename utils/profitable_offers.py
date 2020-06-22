import pandas as pd
from joblib import dump, load
import sys
from scipy import stats
import numpy as np
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


# Define paths to Moscow and Spb data
MOSCOW_DATA_NEW = SETTINGS.DATA_MOSCOW + '/MOSCOW_NEW_FLATS.csv'
MOSCOW_DATA_SECONDARY = SETTINGS.DATA_MOSCOW + '/MOSCOW_VTOR.csv'
SPB_DATA_NEW = SETTINGS.DATA_SPB + '/SPB_NEW_FLATS.csv'
SPB_DATA_SECONDARY = SETTINGS.DATA_SPB + '/SPB_VTOR.csv'


# Find profitable offers
def mean_estimation(full_sq_from, full_sq_to, latitude_from, latitude_to, longitude_from, longitude_to, rooms,
                    price_from, price_to, building_type_str, kitchen_sq, life_sq, renovation, has_elevator, floor_first,
                    floor_last, time_to_metro, city_id):
    # Initialize DF
    data_offers = pd.DataFrame()

    # Set paths to data and price prediction models, depending on city:  0 = Moscow, 1 = Spb
    if city_id == 0:
        data_offers = pd.read_csv(MOSCOW_DATA_SECONDARY)
        # data_offers = data_offers[data_offers.flat_type == 'SECONDARY']
        # gbr = load(PATH_PRICE_GBR_MOSCOW_D)
        # rf = load(PATH_PRICE_RF_MOSCOW_D)
        # lgbm = load(PATH_PRICE_LGBM_MOSCOW_D)
    elif city_id == 1:
        data_offers = pd.read_csv(SPB_DATA_SECONDARY)
        # data_offers = data_offers[data_offers.flat_type == 'SECONDARY']
        # gbr = load(PATH_PRICE_GBR_SPB_D)
        # rf = load(PATH_PRICE_RF_SPB_D)
        # lgbm = load(PATH_PRICE_LGBM_SPB_D)

    # Apply filtering flats in database on parameters: full_sq range, coordinates scope
    filter = (((data_offers.full_sq >= full_sq_from) & (data_offers.full_sq <= full_sq_to)) & (
            data_offers.rooms == rooms) &
              ((data_offers.latitude >= latitude_from) & (data_offers.latitude <= latitude_to))
              & ((data_offers.longitude >= longitude_from) & (data_offers.longitude <= longitude_to)))
    data_offers = data_offers[filter]

    # Use only open offers
    data_offers = data_offers[data_offers['closed'] == False]

    print('columns ', data_offers.columns, flush=True)

    if time_to_metro != None:
        data_offers = data_offers[(data_offers.time_to_metro <= time_to_metro)]
    if rooms != None:
        data_offers = data_offers[data_offers.rooms == rooms]
    if building_type_str != None:
        data_offers = data_offers[data_offers.building_type_str == building_type_str]
    if kitchen_sq != None:
        data_offers = data_offers[
            (data_offers.kitchen_sq >= kitchen_sq - 1) & (data_offers.kitchen_sq <= kitchen_sq + 1)]
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
    
    # Set threshold for showing profitable offers
    print(data_offers.shape, flush=True)

    # Drop outliers
    data_offers = data_offers[data_offers.price > data_offers.price.quantile(0.1)]
    data_offers = data_offers[(data_offers.profit >= 5)]
    print(data_offers.shape, flush=True)
    data_offers = data_offers.sort_values(by=['profit'], ascending=False)
    print("Profitable offers: ", data_offers[['mean_price', "price", 'profit']].head(3), flush=True)

    flats = data_offers.to_dict('record')

    return flats
