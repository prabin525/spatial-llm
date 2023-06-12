import random
import numpy as np
from geopy import distance
from sklearn.manifold import MDS


def transform_umeyama(X, Y):
    """
    Finds the optimal rigid-body transformation between two sets of
    corresponding 2D points X and Y using Umeyama's algorithm.

    Args:
        X (np.ndarray): An N x 2 array representing the first set of points.
        Y (np.ndarray): An N x 2 array representing the second set of points.

    Returns:
        tuple: A tuple (R, t, s) representing the optimal rotation matrix,
        translation vector, and scaling factor.
    """
    # N = X.shape[0]

    # Compute means of X and Y
    mean_X = np.mean(X, axis=0)
    mean_Y = np.mean(Y, axis=0)

    # Center the points
    X_centered = X - mean_X
    Y_centered = Y - mean_Y

    # Compute covariance matrix and its SVD
    C = np.dot(X_centered.T, Y_centered)
    U, s, Vt = np.linalg.svd(C)

    # Construct the optimal rotation matrix
    R = np.dot(Vt.T, U.T)

    # Compute the optimal scaling factor
    trace_C = np.trace(C)
    trace_X = np.trace(np.dot(X_centered.T, X_centered))
    s = trace_C / trace_X

    # Compute the optimal translation vector
    t = mean_Y.T - s * np.dot(R, mean_X.T)

    # Return the optimal rotation matrix, translation vector and scaling factor
    return R, t, s
# R,t,s = transform_umeyama(B,A)
# R,t,s
# P_centered = B - np.mean(B, axis=0)
# P_scaled = s * P_centered
# P_rotated = np.dot(R, P_scaled.T)
# P_translated = P_rotated.T + t
# P_translated


def peform_mds(
        cities_list,
        disparity_df,
        coods_dic,
        metric=True,
        asymmetric=False,
        similarity_measure_used=False
):
    vals = []
    for each in cities_list:
        c = cities_list.copy()
        random.shuffle(c)
        c.remove(each)
        c.append(each)
        disparity_temp = disparity_df[c].loc[c]
        test_key = disparity_temp.columns[-1]
        temp_cods = {k: v for k, v in coods_dic.items() if k in c}
        temp_cods_train = {k: v for k, v in temp_cods.items() if k != test_key}
        temp_cods_test = {k: v for k, v in temp_cods.items() if k == test_key}
        temp_cods_t = {}
        for each in c[0:-1]:
            temp_cods_t[each] = temp_cods_train[each]
        temp_cods_train = temp_cods_t
        disparity_temp = disparity_df[c].loc[c]
        disparity_temp = disparity_temp.to_numpy()
        if asymmetric:
            disparity_temp += disparity_temp.T
        if similarity_measure_used:
            with np.errstate(divide='ignore'):
                disparity_temp = np.reciprocal(disparity_temp)
            disparity_temp[disparity_temp == np.inf] = 2
            disparity_temp = disparity_temp
            np.fill_diagonal(disparity_temp, 0)

        mds_sklearn = MDS(
            metric=metric,
            n_components=2,
            dissimilarity='precomputed',
            random_state=None,
            normalized_stress='auto',
            max_iter=30000,
            eps=1e-5,
            n_init=1
        )
        a = np.array(list(temp_cods_train.values())).astype('float64')
        b = a.mean(axis=0).reshape(1, -1)
        init_value = np.concatenate((a, b), axis=0)
        x_sklearn = mds_sklearn.fit_transform(disparity_temp, init=init_value)

        A = np.array(list(temp_cods_train.values())).astype('float64')
        B = x_sklearn[0:-1]
        R, t, s = transform_umeyama(B, A)

        B = x_sklearn
        P_centered = B - np.mean(B, axis=0)
        P_scaled = s * P_centered
        P_rotated = np.dot(R, P_scaled.T)
        P_translated = P_rotated.T + t
        err = distance.distance(P_translated[-1], temp_cods_test.values()).km
        vals.append({
            'name': list(temp_cods_test.keys())[0],
            'lat': list(temp_cods_test.values())[0][0],
            'lng': list(temp_cods_test.values())[0][1],
            'pred_lat': P_translated[-1][0],
            'pred_lng': P_translated[-1][1],
            'err': err
        })
    return vals


def min_max_scaler(x, df, min_val=0, max_val=1):
    t = np.concatenate(df.to_numpy(), axis=0)
    ma = np.max(t)
    mi = np.min(t)
    std = (x - mi) / (ma - mi)
    x_scaled = std * (max_val - min_val) + min_val
    return x_scaled


# Census Bureau-designated regions and divisions
new_england_divsion = [
    'Bangor, Maine',
    'Boston, Massachusetts',
    'Eastport, Maine',
    'Providence, Rhode Island',
    'New Haven, Connecticut',
]
middle_atlantic_division = [
    'Buffalo, New York',
    'Newark, New Jersey',
    'Pittsburgh, Pennsylvania',
    'Albany, New York',
    'New York, New York',
    'Philadelphia, Pennsylvania',
    'Syracuse, New York',
]

northeast_region = []
northeast_region.extend(new_england_divsion)
northeast_region.extend(middle_atlantic_division)

east_north_central_division = [
    'Grand Rapids, Michigan',
    'Detroit, Michigan',
    'Chicago, Illinois',
    'Milwaukee, Wisconsin',
    'Cincinnati, Ohio',
    'Cleveland, Ohio',
    'Toledo, Ohio',
    'Columbus, Ohio',
    'Indianapolis, Indiana',
]
west_north_central_division = [
    'Des Moines, Iowa',
    'Duluth, Minnesota',
    'Wichita, Kansas',
    'Kansas City, Missouri',
    'Minneapolis, Minnesota',
    'Sioux Falls, South Dakota',
    'Bismarck, North Dakota',
    'St. Louis, Missouri',
    'Dubuque, Iowa',
    'Omaha, Nebraska',
    'Fargo, North Dakota',
    'Pierre, South Dakota',
]

midwest_region = []
midwest_region.extend(east_north_central_division)
midwest_region.extend(west_north_central_division)


south_atlantic_division = [
    'Savannah, Georgia',
    'Atlanta, Georgia',
    'Virginia Beach, Virginia',
    'Tampa, Florida',
    'Jacksonville, Florida',
    'Key West, Florida',
    'Raleigh, North Carolina',
    'Baltimore, Maryland',
    'Charlotte, North Carolina',
    'Miami, Florida',
    'Montpelier, Virginia',
    'Roanoke, Virginia',
    'Richmond, Virginia',
    'Wilmington, North Carolina'
]
east_south_central_division = [
    'Louisville, Kentucky',
    'Memphis, Tennessee',
    'Knoxville, Tennessee',
    'Montgomery, Alabama',
    'Nashville, Tennessee'
]
west_south_central_division = [
    'Oklahoma City, Oklahoma',
    'Dallas, Texas',
    'Fort Worth, Texas',
    'Houston, Texas',
    'San Antonio, Texas',
    'New Orleans, Louisiana',
    'Hot Springs, Arkansas',
    'Austin, Texas',
    'Tulsa, Oklahoma',
    'Amarillo, Texas',
    'Shreveport, Louisiana',
]

south_region = []
south_region.extend(south_atlantic_division)
south_region.extend(east_south_central_division)
south_region.extend(west_south_central_division)

mountain_division = [
    'Santa Fe, New Mexico',
    'Phoenix, Arizona',
    'Havre, Montana',
    'Carlsbad, New Mexico',
    'Richfield, Utah',
    'Boise, Idaho',
    'Lewiston, Idaho',
    'Las Vegas, Nevada',
    'Flagstaff, Arizona',
    'Grand Junction, Colorado',
    'Albuquerque, New Mexico',
    'Denver, Colorado',
    'Salt Lake City, Utah',
    'Helena, Montana',
    'Cheyenne, Wyoming',
    'Idaho Falls, Idaho',
    'Reno, Nevada',
]
pacific_division = [
    'Eugene, Oregon',
    'San Diego, California',
    'Los Angeles, California',
    'San Francisco, California',
    'Sacramento, California',
    'El Centro, California',
    'Klamath Falls, Oregon',
    'San Jose, California',
    'Long Beach, California',
    'Oakland, California',
    'Seattle, Washington State',
    'Fresno, California',
]

west_region = []
west_region.extend(mountain_division)
west_region.extend(pacific_division)


def peform_mds_division(
        disparity_df,
        coods_dic,
        metric=True,
        asymmetric=False,
        similarity_measure_used=False
):
    vals = []
    divisions = [
        new_england_divsion,
        middle_atlantic_division,
        east_north_central_division,
        west_north_central_division,
        south_atlantic_division,
        east_south_central_division,
        west_south_central_division,
        mountain_division,
        pacific_division
    ]
    for div in divisions:
        vals.extend(
            peform_mds(
                cities_list=div,
                disparity_df=disparity_df,
                coods_dic=coods_dic,
                metric=metric,
                asymmetric=asymmetric,
                similarity_measure_used=similarity_measure_used
            )
        )
    return vals
