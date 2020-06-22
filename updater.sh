# !/bin/bash
sudo -u postgres psql yand -c "copy (select prices.id, prices.price, prices.changed_date, prices.flat_id, prices.created_at, prices.updated_at from prices, flats, buildings, districts where prices.flat_id = flats.id and flats.building_id = buildings.id and buildings.district_id = districts.id and districts.city_id = 1) to STDOUT With CSV DELIMITER ',';" > "$1prices.csv"

sudo -u postgres psql yand -c "copy (select flats.id, flats.full_sq, flats.kitchen_sq, flats.life_sq, flats.floor, flats.is_apartment, flats.building_id, flats.created_at, flats.updated_at, flats.offer_id, flats.closed, flats.rooms_total, flats.image, flats.resource_id, flats.flat_type, flats.is_rented, flats.rent_quarter, flats.rent_year, flats.agency, flats.renovation_type, flats.windows_view, flats.close_date from flats, buildings, districts where flats.building_id = buildings.id and buildings.district_id = districts.id and districts.city_id = 1) to STDOUT With CSV DELIMITER ',';" > "$1flats.csv"

sudo -u postgres psql yand -c "copy (select buildings.id, buildings.max_floor, buildings.building_type_str, buildings.built_year, buildings.flats_count, buildings.address, buildings.renovation, buildings.has_elevator, buildings.longitude, buildings.latitude, buildings.district_id, buildings.created_at, buildings.updated_at, buildings.schools_500m, buildings.schools_1000m, buildings.kindergartens_500m, buildings.kindergartens_1000m, buildings.clinics_500m, buildings.clinics_1000m, buildings.shops_500m, buildings.shops_1000m from buildings, districts where buildings.district_id = districts.id and districts.city_id = 1) to STDOUT With CSV DELIMITER ',';" > "$1buildings.csv"

sudo -u postgres psql yand -c "copy (select districts.id, districts.name, districts.population, districts.city_id, districts.created_at, districts.updated_at, districts.prefix from districts where districts.city_id = 1) to STDOUT With CSV DELIMITER ',';" > "$1districts.csv"

sudo -u postgres psql yand -c "copy (select time_metro_buildings.id, time_metro_buildings.building_id, time_metro_buildings.metro_id, time_metro_buildings.time_to_metro, time_metro_buildings.transport_type, time_metro_buildings.created_at, time_metro_buildings.updated_at from time_metro_buildings, buildings, districts where time_metro_buildings.building_id = buildings.id and buildings.district_id = districts.id and districts.city_id = 1) to STDOUT With CSV DELIMITER ',';" > "$1time_metro_buildings.csv"

sudo -u postgres psql yand -c "copy (select * from metros where city_id = 1) to STDOUT With CSV DELIMITER ',';" > "$1metro.csv"


sudo -u postgres psql yand -c "copy (select prices.id, prices.price, prices.changed_date, prices.flat_id, prices.created_at, prices.updated_at from prices, flats, buildings, districts where prices.flat_id = flats.id and flats.building_id = buildings.id and buildings.district_id = districts.id and districts.city_id = 2) to STDOUT With CSV DELIMITER ',';" > "$2prices.csv"

sudo -u postgres psql yand -c "copy (select flats.id, flats.full_sq, flats.kitchen_sq, flats.life_sq, flats.floor, flats.is_apartment, flats.building_id, flats.created_at, flats.updated_at, flats.offer_id, flats.closed, flats.rooms_total, flats.image, flats.resource_id, flats.flat_type, flats.is_rented, flats.rent_quarter, flats.rent_year, flats.agency, flats.renovation_type, flats.windows_view, flats.close_date from flats, buildings, districts where flats.building_id = buildings.id and buildings.district_id = districts.id and districts.city_id = 2) to STDOUT With CSV DELIMITER ',';" > "$2flats.csv"

sudo -u postgres psql yand -c "copy (select buildings.id, buildings.max_floor, buildings.building_type_str, buildings.built_year, buildings.flats_count, buildings.address, buildings.renovation, buildings.has_elevator, buildings.longitude, buildings.latitude, buildings.district_id, buildings.created_at, buildings.updated_at, buildings.schools_500m, buildings.schools_1000m, buildings.kindergartens_500m, buildings.kindergartens_1000m, buildings.clinics_500m, buildings.clinics_1000m, buildings.shops_500m, buildings.shops_1000m from buildings, districts where buildings.district_id = districts.id and districts.city_id = 2) to STDOUT With CSV DELIMITER ',';" > "$2buildings.csv"

sudo -u postgres psql yand -c "copy (select districts.id, districts.name, districts.population, districts.city_id, districts.created_at, districts.updated_at, districts.prefix from districts where districts.city_id = 2) to STDOUT With CSV DELIMITER ',';" > "$2districts.csv"

sudo -u postgres psql yand -c "copy (select time_metro_buildings.id, time_metro_buildings.building_id, time_metro_buildings.metro_id, time_metro_buildings.time_to_metro, time_metro_buildings.transport_type, time_metro_buildings.created_at, time_metro_buildings.updated_at from time_metro_buildings, buildings, districts where time_metro_buildings.building_id = buildings.id and buildings.district_id = districts.id and districts.city_id = 2) to STDOUT With CSV DELIMITER ',';" > "$2time_metro_buildings.csv"

sudo -u postgres psql yand -c "copy (select * from metros where city_id = 2) to STDOUT With CSV DELIMITER ',';" > "$2metro.csv"

source "$3venv/bin/activate"


python3 "$3data_process/DATA_PREP_MOSCOW.py"
python3 "$3data_process/DATA_PREP_SPB.py"

python3 "$3data_process/PRICE_MOSCOW.py"
python3 "$3data_process/PRICE_SPB.py"
