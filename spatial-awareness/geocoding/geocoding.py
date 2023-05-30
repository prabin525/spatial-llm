import geocoder
import json
from pathlib import Path


class GeoCoding:
    '''
    Geocoding object to use geonames geocoder and cache the requests made for
    future use.
    '''
    def __init__(
        self,
        dic=f'{Path(__file__).parent.resolve()}/dic.json',
        key='pbhanda2'
    ):
        self.dic = dic
        self.key = key
        f = open(self.dic)
        self.cached_requests = json.load(f)

    def dump_cached_requests(self):
        j = json.dumps(self.cached_requests)
        f = open(self.dic, "w")
        f.write(j)
        f.close()

    def get_lat_lng(self, name):
        '''
        Args:
            name (Str)
        Returns:
            (latitude, longitude, address, geoname_id)
        '''
        if name not in self.cached_requests.keys():
            print('Request Made')
            g = geocoder.geonames(name, key=self.key)
            self.cached_requests[name] = (
                g.lat,
                g.lng,
                g.address,
                g.geonames_id,
                g.population
            )
            self.dump_cached_requests()
        return self.cached_requests[name]
