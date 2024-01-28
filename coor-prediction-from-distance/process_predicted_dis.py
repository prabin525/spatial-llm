import json
import argparse
import pandas as pd


def process_llama():
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


def process_llama2():
    cities = json.load(open('outputs/gen_dis_llama2-3-shot.json'))
    df = pd.read_pickle('../spatial-awareness/count_df_llama2.pkl')

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

    df_temp = pd.DataFrame.from_dict(cities)
    df['predicted_dis'] = df_temp['predicted_dis']
    df.to_pickle('cities_llama2.pkl')


def process_opt():
    cities = json.load(open('outputs/gen_dis_opt-3-shot.json'))
    df = pd.read_pickle('../spatial-awareness/count_df_opt.pkl')

    for each in cities:
        o = each['output'][0].replace(each['prompt'], '').rstrip('.')
        try:
            o = float(o)
        except ValueError:
            try:
                o = o.replace(',', '')
                o = float(o)
            except ValueError:
                o = None
        del each['output']
        del each['prompt']
        each['predicted_dis'] = o

    df_temp = pd.DataFrame.from_dict(cities)
    df['predicted_dis'] = df_temp['predicted_dis']
    df.to_pickle('cities_opt.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process predicted distance from LLAMA'
    )
    parser.add_argument(
        '--model',
        dest='model',
        choices=[
            'llama',
            'llama2',
            'opt',
        ],
        default='opt'
    )
    args = parser.parse_args()
    print(args)

    if args.model == 'llama':
        process_llama()
    elif args.model == 'llama2':
        process_llama2()
    elif args.model == 'opt':
        process_opt()
    else:
        pass
