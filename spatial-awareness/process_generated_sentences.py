import os
import json
import pandas as pd
import geopy.distance
import stanza
from geocoding.geocoding import GeoCoding
geocoder = GeoCoding()
nlp = stanza.Pipeline('en')

# base_dir = 'outputs/'
# base_dir = 'outputs_with_eos/'
# base_dir = 'outputs_llama2/'
# out_dir = 'outputs_processed_llama2/'
# out_dir = 'outputs_with_eos_processed/'

base_dir = 'outputs_opt/'
out_dir = 'outputs_processed_opt/'
files = os.listdir(base_dir)
files_3 = [f for f in files if '3-shot' in f]

for f in files_3:
    df = pd.read_json(open(f'{base_dir}{f}'))
    res = []
    for i, each in df.iterrows():
        for sen in each['output']:
            each2 = each.to_dict().copy()
            del (each2['output'])
            del (each2['prompt'])
            if each.p_type == 'near':
                generated_place = sen.split(
                    "\n\n"
                )[3].lstrip(f'{each["name"]} is near')
            elif each.p_type == 'close':
                generated_place = sen.split(
                    "\n\n"
                )[3].lstrip(f'{each["name"]} is close to')
            elif each.p_type == 'far':
                generated_place = sen.split(
                    "\n\n"
                )[3].lstrip(f'{each["name"]} is far from')
            elif each.p_type == 'and':
                generated_place = sen.split(
                    "\n\n"
                )[3].lstrip(f'{each["name"]} and')
            doc = nlp(generated_place)
            entities = [
                ent.text
                for sent in doc.sentences
                for ent in sent.ents
                if ent.type == 'GPE'
            ]
            if len(entities) > 0:
                lat, lng, _, _, _ = geocoder.get_lat_lng(generated_place)
                if (lat is None) or (lng is None):
                    pass
                else:
                    dis = geopy.distance.distance(
                        [each['lat'], each['lng']],
                        [lat, lng]
                    ).km
                    each2['city_b'] = generated_place
                    each2['lat_b'] = lat
                    each2['lng_b'] = lng
                    each2['dis'] = dis
                    res.append(each2)

    json.dump(
        res,
        open(
            f'{out_dir}{f}',
            'w+'
        )
    )
