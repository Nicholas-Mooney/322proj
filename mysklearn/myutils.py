'''
Author: Nicholas Mooney
4/6/2022
PA6
'''
import numpy as np
import copy



def get_column(matrix, i):
    return [row[i] for row in matrix]

def shuffle_list(mylist, seed):
    #print(seed)
    #print(mylist)
    if seed == None:
        seed = 0
    np.random.seed(seed)
    list2 = sorted(mylist, key=lambda k: np.random.random())
    #print(list2)
    return copy.deepcopy(list2)


def get_uniques(mylist):
    list_set = set(mylist)
    unique_list = (list(list_set))
    return copy.deepcopy(unique_list)


def mpg_rating(num):
    if(0 < num <= 13):
        return 1
    elif num < 14:
        return 2
    elif num < 16:
        return 3
    elif num < 19:
        return 4
    elif num < 23:
        return 5
    elif num < 26:
        return 6
    elif num < 30:
        return 7
    elif num < 36:
        return 8
    else:
        return 9
