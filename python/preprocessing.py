import numpy as np


def print_bold(text): print(f"\033[1m {text} \033[0m")


def normalize(l, verbose=False):
    l = (l - np.mean(l)) / np.std(l)
    l = [i.tolist() for i in l]
    if verbose:
        print("max: ", max(l))
        print("mean: ", np.mean(l))
        print("std: ", np.std(l))
    return l


def normalize_lenghts(l, verbose=False, max_lenght=0):
    """l is a list of lists"""
    if max_lenght == 0:
        max_lenght = max_listoflists_lenght(l, verbose)
    new_l = [np.interp(np.linspace(0, 1, max_lenght).astype('float'), np.linspace(0, 1, len(l_i)).astype('float'), l_i)
             for l_i in l]
    if verbose:
        min_listoflists_lenght(l)
    return new_l


def normalize_single_lenght(l, max_lenght=0):
    """l is a list"""
    if max_lenght == 0:
        max_lenght = max([len(i) for i in l])
    print("max_lenght: ", max_lenght)
    new_l = np.interp(np.linspace(0, 1, max_lenght).astype('float'), np.linspace(0, 1, len(l)).astype('float'), l)
    return new_l


def max_listoflists_lenght(l, verbose=False):
    """l is a list of lists"""
    max_lenght = 0
    for i in l:
        if max_lenght < len(i):
            max_lenght = len(i)
    if verbose:
        print("min_lenght: ", min_listoflists_lenght(l))
        print("max_lenght: ", max_lenght)
    return max_lenght


def min_listoflists_lenght(l):
    """l is a list of lists"""
    min_lenght = 999999
    for i in l:
        if min_lenght > len(i):
            min_lenght = len(i)
    return min_lenght


def normalize_data(df, by_value=True, by_lenght=True, verbose=True):
    # normalization
    print("checking sequence lenghts matching (first 20)...\n")

    if verbose:
        for ix, (d, p) in enumerate(zip(df['ts_distance'], df['ts_pupil'])):
            print(f"lenght of sequence in ts_distance: {len(d)}, lenght of sequence in ts_pupil: {len(p)}", end=" ")
            print_bold(len(d) == len(p))
            if ix == 15:
                break

    print("\nmin lenght in ts_distance = ", min_listoflists_lenght(df['ts_distance'].tolist()))
    print("min lenght in ts_pupil = ", min_listoflists_lenght(df['ts_pupil'].tolist()))
    print("max lenght in ts_distance = ", max_listoflists_lenght(df['ts_distance'].tolist()))
    print("max lenght in ts_pupil = ", max_listoflists_lenght(df['ts_pupil'].tolist()))

    # apply normalization to the DataFrame (value and lenght)
    if by_value:
        df.ts_distance = df.ts_distance.apply(normalize)
    if by_lenght:
        df.ts_pupil = df.ts_pupil.apply(normalize)
    print("\nthe data has been normalized by value, mean=0, std=1")

    df.ts_distance = normalize_lenghts(df.ts_distance.tolist())
    df.ts_pupil = normalize_lenghts(df.ts_pupil.tolist(),
                                    max_lenght=max_listoflists_lenght(df['ts_distance']))
    print("the data has been normalized by lenght (stretched to the max lenght)")

    if verbose:
        print("\nmin lenght in ts_distance = ", min_listoflists_lenght(df['ts_distance'].tolist()))
        print("min lenght in ts_pupil = ", min_listoflists_lenght(df['ts_pupil'].tolist()))
        print("max lenght in ts_distance = ", max_listoflists_lenght(df['ts_distance'].tolist()))
        print("max lenght in ts_pupil = ", max_listoflists_lenght(df['ts_pupil'].tolist()))

    max_ts_lenght = max([max_listoflists_lenght(df['ts_distance'].tolist()), max_listoflists_lenght(df['ts_pupil'].tolist())])
    return df, max_ts_lenght
