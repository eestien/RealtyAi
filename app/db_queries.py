import psycopg2

import settings_local as SETTINGS



def get_other_params(flats):
    conn = psycopg2.connect(host=SETTINGS.host, dbname=SETTINGS.name, user=SETTINGS.user, password=SETTINGS.password)
    cur = conn.cursor()
    for flat in flats:
        # print(flat.keys(), flush=True)
        cur.execute("select metro_id, time_to_metro from time_metro_buildings where building_id=%s and transport_type='ON_FOOT';",
                    (flat['building_id'],))
        metros_info = cur.fetchall()
        flat['metros'] = []
        for metro in metros_info:
            cur.execute("select name from metros where id=%s;", (metro[0],))
            flat['metros'].append({'station': cur.fetchone()[0], 'time_to_metro': metro[1]})

        if flat['resource_id'] == 0:
            flat['link'] = 'https://realty.yandex.ru/offer/' + str(flat['offer_id'])
        else:
            flat['link'] = 'https://www.cian.ru/sale/flat/' + str(flat['offer_id'])

        cur.execute("select address from buildings where id=%s;",
                    (flat['building_id'],))
        flat['address'] = cur.fetchone()[0]

        if type(flat['image']) != str:
            flat['image'] = None
        del flat['offer_id']
        del flat['building_id']
        del flat['time_to_metro']

    conn.close()

    return flats
