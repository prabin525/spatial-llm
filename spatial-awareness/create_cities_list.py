import geopy.distance
from geocoding.geocoding import GeoCoding
import pandas as pd

in_file_loc = 'places_with_lat_lng_within_continent.txt'
geocoder = GeoCoding()
places = []
with open(in_file_loc) as f:
    for each in f.readlines():
        splitted = each.split("\t")
        lat, lng, _, _, _ = geocoder.get_lat_lng(splitted[0])
        places.append({
            'name': splitted[0],
            'lat': lat,
            'lng': lng
        })
for each in places:
    places2 = places.copy()
    places2.remove(each)
    for i, c in enumerate(places2):
        if i == 0:
            dis = geopy.distance.distance(
                [each['lat'], each['lng']], [c['lat'], c['lng']]
            ).km
            near_city = c['name']
            near_dis = dis
            far_city = c['name']
            far_dis = dis
        else:
            dis = geopy.distance.distance(
                [each['lat'], each['lng']], [c['lat'], c['lng']]
            ).km
            if dis < near_dis:
                near_dis = dis
                near_city = c['name']
            if dis > far_dis:
                far_dis = dis
                far_city = c['name']
    each['near_city'] = near_city
    each['far_city'] = far_city

df = pd.DataFrame(places)
df.to_pickle('cities.pkl')
