import json
import pandas as pd

cities = json.load(open('outputs/gen_dis-3-shot.json'))

for each in cities:
    o = each['output'][0].replace(each['prompt'], '').rstrip('.')
    try:
        o = float(o)
    except ValueError:
        o = o.replace(',', '')
        o = float(o)
    del each['output']
    del each['prompt']
    each['predicted_dis'] = o

df = pd.DataFrame.from_dict(cities)
df.to_pickle('cities.pkl')
