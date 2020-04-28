from flask import Flask, request, jsonify, url_for
import json

from data_process.main_process import mean_estimation, map_estimation, predict_developers_term
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

    print('flats info', flats, flush=True)

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

    image_link = SETTINGS.HOST + SETTINGS.MEDIA_ROOT + 'test.jpg'
    print(image_link, flush=True)

    result = predict_developers_term(result)

    print(result, flush=True)
    # print(type(result), flush=True)

    return jsonify({'result': result, 'image_link': image_link})
