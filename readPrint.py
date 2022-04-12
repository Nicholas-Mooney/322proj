import csv as c
import pandas as pd
import copy as copy

def unique(list1):
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
    for x in unique_list:
        if(x[0] != '5'):
            if( not (x[0:3] == '3rd')):
                if( not (x[0:3] == '4th')):
                    print(x)
    return unique_list

def get_agregate(list1):
    new_uniques = []
    for item in list1:
        item_split = item.split(' ')
        #print(item_split)
        for item_thing in item_split:
            if(item_thing not in new_uniques):
                new_uniques.append(item_thing)
                print(item_thing)
    print(new_uniques.sort())
    return new_uniques

def main():
    # reading CSV file
    data = pd.read_csv("mp_routes.csv")

    # converting column data to list
    difficulties = data['Rating'].tolist()
    print(unique(difficulties))
    print(len(unique(difficulties)))
    print(get_agregate(unique(difficulties)))
if(__name__ == '__main__'):
    main()
