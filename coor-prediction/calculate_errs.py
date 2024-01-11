import re
import os
import pandas as pd
from dms2dec.dms_convert import dms2dec
import numpy as np
from geopy import distance


def cal_errs_llm(a):
    a['predicted_Latitude'] = 0
    a['predicted_Longitude'] = 0
    for i, each in a.iterrows():
        rr = re.findall(
            "[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?",
            each['output'].split('\n')[0]
        )
        if len(rr) == 2:
            a.loc[i, 'predicted_Latitude'] = rr[0]
            a.loc[i, 'predicted_Longitude'] = rr[1]
        else:
            rr = re.findall(
                "(\d+)\s?\°\s?(\d+)\s?\'\s?(\d{1,}\.?\,?\d{0,}?)\"\s?(N|W|S|E)",
                each['output']
            )
            if len(rr) == 2:
                a.loc[i, 'predicted_Latitude'] = dms2dec(
                    f'''{rr[0][0]}°{rr[0][1]}'{rr[0][2]}"{rr[0][3]}'''
                )
                a.loc[i, 'predicted_Longitude'] = dms2dec(
                    f'''{rr[1][0]}°{rr[1][1]}'{rr[1][2]}"{rr[1][3]}'''
                )
            else:
                rr = re.findall(
                    "(\d+)\s?\°\s?(\d+)\s?\'\s?(N|W|S|E)",
                    each['output']
                )
                if len(rr) == 2:
                    a.loc[i, 'predicted_Latitude'] = dms2dec(
                        f'''{rr[0][0]}°{rr[0][1]}'0"{rr[0][2]}'''
                    )
                    a.loc[i, 'predicted_Longitude'] = dms2dec(
                        f'''{rr[1][0]}°{rr[1][1]}'0"{rr[1][2]}'''
                    )
                else:
                    rr = re.findall(
                        "[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?",
                        each['output'].split('\n')[0].split(". ")[0].lstrip(": ")
                    )
                    if len(rr) == 2:
                        a.loc[i, 'predicted_Latitude'] = rr[0]
                        a.loc[i, 'predicted_Longitude'] = rr[1]
                    else:
                        rr = re.findall(
                            "(\d+)\s?\°\s?(\d+)\s?\'\s?(\d{1,}\.?\,?\d{0,}?)\"\s?(N|W|S|E)",
                            each['output'].split('\n')[0].split(". ")[0].lstrip(": ")
                        )
                        if len(rr) == 2:
                            a.loc[i, 'predicted_Latitude'] = dms2dec(
                                f'''{rr[0][0]}°{rr[0][1]}'{rr[0][2]}"{rr[0][3]}'''
                            )
                            a.loc[i, 'predicted_Longitude'] = dms2dec(
                                f'''{rr[1][0]}°{rr[1][1]}'{rr[1][2]}"{rr[1][3]}'''
                            )
                        else:
                            # print(each['output'])
                            a.loc[i, 'predicted_Latitude'] = np.nan
                            a.loc[i, 'predicted_Longitude'] = np.nan
    a['err'] = 0
    for i, each in a.iterrows():
        if each['predicted_Latitude'] is not np.nan:
            try:
                err = distance.distance(
                    (each['Latitude'], each['Longitude']),
                    (each['predicted_Latitude'], each['predicted_Longitude'])
                ).km
            except ValueError:
                err = np.NaN
        else:
            err = np.NaN
        a.loc[i, 'err'] = err
    return a


def cal_errs_instruction(a):
    a['predicted_Latitude'] = 0
    a['predicted_Longitude'] = 0
    for i, each in a.iterrows():
        rr = re.findall(
            "[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?",
            each['output']
        )
        if len(rr) == 2:
            a.loc[i, 'predicted_Latitude'] = rr[0]
            a.loc[i, 'predicted_Longitude'] = rr[1]
        else:
            rr = re.findall(
                "(\d+)\s?\°\s?(\d+)\s?\'\s?(\d{1,}\.?\,?\d{0,}?)\"\s?(N|W|S|E)",
                each['output']
            )
            if len(rr) == 2:
                a.loc[i, 'predicted_Latitude'] = dms2dec(
                    f'''{rr[0][0]}°{rr[0][1]}'{rr[0][2]}"{rr[0][3]}'''
                )
                a.loc[i, 'predicted_Longitude'] = dms2dec(
                    f'''{rr[1][0]}°{rr[1][1]}'{rr[1][2]}"{rr[1][3]}'''
                )
            else:
                rr = re.findall(
                    "(\d+)\s?\°\s?(\d+)\s?\'\s?(N|W|S|E)", each['output']
                )
                if len(rr) == 2:
                    a.loc[i, 'predicted_Latitude'] = dms2dec(
                        f'''{rr[0][0]}°{rr[0][1]}'0"{rr[0][2]}'''
                    )
                    a.loc[i, 'predicted_Longitude'] = dms2dec(
                        f'''{rr[1][0]}°{rr[1][1]}'0"{rr[1][2]}'''
                    )
                else:
                    a.loc[i, 'predicted_Latitude'] = np.nan
                    a.loc[i, 'predicted_Longitude'] = np.nan
    a['err'] = 0
    for i, each in a.iterrows():
        if each['predicted_Latitude'] is not np.nan:
            try:
                err = distance.distance(
                    (each['Latitude'], each['Longitude']),
                    (each['predicted_Latitude'], each['predicted_Longitude'])
                ).km
            except ValueError:
                err = np.NaN
        else:
            err = np.NaN
        a.loc[i, 'err'] = err
    return a


if __name__ == '__main__':
    folder_loc = './outputs'
    out_folder = './processed_outputs'
    dirs = os.listdir(folder_loc)
    for file in dirs:
        model = file.split("-")[0].split("_")[2]
        a = pd.read_json(open(f'{folder_loc}/{file}'))
        if model in ['opt', 'llama', 'llama2']:
            a = cal_errs_llm(a)
            a.to_json(f'{out_folder}/{file}')
        elif model in ['alpaca', 'llama2chat']:
            a = cal_errs_instruction(a)
            a.to_json(f'{out_folder}/{file}')
