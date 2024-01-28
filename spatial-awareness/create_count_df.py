import os
import re
import pandas as pd


# base_dir = './outputs_processed/'
# base_dir = './outputs_processed_llama2/'
base_dir = './outputs_processed_opt/'

files = os.listdir(base_dir)

for i, f in enumerate(files):
    if i == 0:
        df = pd.read_json(open(f'{base_dir}{f}'))
    else:
        df2 = pd.read_json(open(f'{base_dir}{f}'))
        df = pd.concat([df, df2])
cc100 = pd.read_pickle('cc100_inverted_df.pkl')
cities = list(set(cc100.a_name.to_list()))
cities = [each.lower() for each in cities]
cities_ns = [each.split(',')[0] for each in cities]
cc100['near_count'] = 0
cc100['close_count'] = 0
cc100['and_count'] = 0
cc100['far_count'] = 0


def update_value(cc100, prep, a_name, b_name, ns=False):
    if ns:
        if prep == 'near':
            cc100.loc[
                (cc100.a_name == a_name) &
                (cc100.b_name.str.lower().str.split(',').str[0] == b_name),
                'near_count'
            ] += 1
        elif prep == 'close':
            cc100.loc[
                (cc100.a_name == a_name) &
                (cc100.b_name.str.lower().str.split(',').str[0] == b_name),
                'close_count'
            ] += 1
        elif prep == 'far':
            cc100.loc[
                (cc100.a_name == a_name) &
                (cc100.b_name.str.lower().str.split(',').str[0] == b_name),
                'far_count'
            ] += 1
        elif prep == 'and':
            cc100.loc[
                (cc100.a_name == a_name) &
                (cc100.b_name.str.lower().str.split(',').str[0] == b_name),
                'and_count'
            ] += 1
    else:
        if prep == 'near':
            cc100.loc[
                (cc100.a_name == a_name) &
                (cc100.b_name.str.lower() == b_name),
                'near_count'
            ] += 1
        elif prep == 'close':
            cc100.loc[
                (cc100.a_name == a_name) &
                (cc100.b_name.str.lower() == b_name),
                'close_count'
            ] += 1
        elif prep == 'far':
            cc100.loc[
                (cc100.a_name == a_name) &
                (cc100.b_name.str.lower() == b_name),
                'far_count'
            ] += 1
        elif prep == 'and':
            cc100.loc[
                (cc100.a_name == a_name) &
                (cc100.b_name.str.lower() == b_name),
                'and_count'
            ] += 1


for i, each in df.iterrows():
    city_b = re.sub('\W+', ' ', each['city_b'].lower())
    city_b = city_b.strip().replace(' ', ', ')
    if city_b in cities:
        update_value(
            cc100,
            each['p_type'],
            each['name'],
            city_b,
        )
    elif city_b in cities_ns:
        update_value(
            cc100,
            each['p_type'],
            each['name'],
            city_b,
            ns=True
        )

cc100.to_pickle('count_df_opt.pkl')
