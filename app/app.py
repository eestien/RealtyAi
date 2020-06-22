from flask import Flask, request, jsonify
import json
import time
from datetime import datetime

from utils.developers import predict_developers_term
from utils.price_term import map_estimation
from utils.profitable_offers import mean_estimation
from app.db_queries import get_other_params
import settings_local as SETTINGS

app = Flask(__name__)


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

    flats = mean_estimation(full_sq_from, full_sq_to, latitude_from, latitude_to, longitude_from, longitude_to, rooms,
                            price_from, price_to, building_type_str, kitchen_sq, life_sq, renovation, has_elevator,
                            floor_first,
                            floor_last, time_to_metro, city_id)

    # print('flats info', flats, flush=True)

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

    flats = get_other_params(flats)

    print('flats', len(flats), flush=True)

    # if math.isnan(mean_price):
    #     mean_price = None

    print('COUNTED, returning answer', flush=True)
    return jsonify({'flats': flats, 'page': page, 'max_page': max_page, 'count': flats_count})


@app.route('/map')
def map():
    longitude = float(request.args.get('lng'))
    rooms = int(request.args.get('rooms')) if request.args.get('rooms') is not None else 0
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

    print("Params: City id: {0}, is secondary: {1}".format(city_id, secondary), flush=True)

    result = map_estimation(longitude, rooms, latitude, full_sq, kitchen_sq, life_sq, renovation, secondary,
                            has_elevator, floor_first, floor_last, time_to_metro, is_rented, rent_year, rent_quarter,
                            city_id)

    print('COUNTED, returning answer', flush=True)

    return jsonify(result)


@app.route('/api/builder/', methods=['POST'])
def builder():
    result = json.loads(request.data.decode())

    print('query params', result, flush=True)

    is_rented = 0
    rent_year = 0# result['rent_year'] if result['rent_year'] is not None else None
    rent_quarter = 0 # result['rent_quarter'] if result['rent_quarter'] is not None else None
    schools_500m = 0 # result['schools_500m'] if result['schools_500m'] is not None else None
    schools_1000m = 0 #result['schools_1000m'] if result['schools_1000m'] is not None else None
    kindergartens_500m = 0 #result['kindergartens_500m'] if result['kindergartens_500m'] is not None else None
    kindergartens_1000m = 0 # result['kindergartens_1000m'] if result['kindergartens_1000m'] is not None else None
    clinics_500m = 0 # result['clinics_500m'] if result['clinics_500m'] is not None else None
    clinics_1000m = 0# result['clinics_1000m'] if result['clinics_1000m'] is not None else None
    shops_500m = 0 #result['shops_500m'] if result['shops_500m'] is not None else None
    shops_1000m = 0 #result['shops_1000m'] if result['shops_1000m'] is not None else None

    try:
        city_id = int(result['city_id'])
        longitude = result['longitude']
        latitude = result['latitude']
        housing_class = int(result['housing_class'])
        floors_count = int(result['floors_count'])
        has_elevator = int(result['elevator'])
        parking = int(result['parking'])
        time_to_metro = result['time_to_metro']
        flats = result['flats']
        start_timestamp = result['start_timestamp'] # Actually string
        end_timestamp = result['end_timestamp'] # Actually string dd.mm.yyyy


    except KeyError as err:
        return jsonify({'message': str(err)})
    print(start_timestamp, end_timestamp, flush=True)

    mm_start = int(datetime.utcfromtimestamp(start_timestamp).strftime('%m'))  # Get month from unix
    # mm_start = int(start_timestamp[3:5])
    yyyy_start = int(datetime.utcfromtimestamp(start_timestamp).strftime('%Y'))  # Get year from unix
    # yyyy_start = int(start_timestamp[-4:])
    mm_end = int(datetime.utcfromtimestamp(end_timestamp).strftime('%m'))  # Get month from unix
    # mm_end = int(end_timestamp[3:5])
    yyyy_end = int(datetime.utcfromtimestamp(end_timestamp).strftime('%Y'))  # Get year from unix
    # yyyy_end = int(end_timestamp[-4:])


    first_graphic, second_graphic, third_graphic = predict_developers_term(city_id=city_id, longitude=longitude, latitude=latitude, is_rented=is_rented,
                                     rent_year=rent_year, rent_quarter=rent_quarter, floors_count=floors_count,
                                     has_elevator=has_elevator, parking=parking, time_to_metro=time_to_metro, flats=flats,
                                     sale_start_month=mm_start, sale_end_month=mm_end,
                                     sale_start_year=yyyy_start, sale_end_year=yyyy_end, schools_500m=schools_500m,
                                     schools_1000m=schools_1000m, kindergartens_500m=kindergartens_500m,
                                     kindergartens_1000m=kindergartens_1000m, clinics_500m=clinics_500m,
                                     clinics_1000m=clinics_1000m, shops_500m=shops_500m, shops_1000m=shops_1000m, housing_class=housing_class)

    print("Result OK.", flush=True)
    # print(type(result), flush=True)


    return jsonify({'first_graphic': first_graphic, 'second_graphic': second_graphic, 'third_graphic': third_graphic})


@app.route('/test-route/')
def test():
    time.sleep(80)
    return jsonify({'status': 'ok'})
