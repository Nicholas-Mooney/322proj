import csv as c
import pandas as pd
import copy as copy
from mysklearn.mypytable import MyPyTable
from mysklearn.myutils import *
from mysklearn.myclassifiers import *
from mysklearn.myevaluation import *

def unique(list1):
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
    #for x in unique_list:
    #    if(x[0] != '5'):
    #        if( not (x[0:3] == '3rd')):
    #            if( not (x[0:3] == '4th')):
     #               print(x)
    return unique_list

def get_agregate(list1):
    new_uniques = []
    for item in list1:
        item_split = item.split(' ')
        #print(item_split)
        for item_thing in item_split:
            if(item_thing not in new_uniques):
                new_uniques.append(item_thing)
                #print(item_thing)
    print(new_uniques.sort())
    return new_uniques

def accTradSport(list1):
    trad = []
    sport = []
    for item in list1:
        if 'Trad' in item:
            trad.append('isTrad')
        else:
            trad.append('notTrad')
        if 'Sport' in item:
            sport.append('isSport')
        else:
            sport.append('notSport')
    #print(trad)
    return trad, sport

def main():
    # reading CSV file
    data = pd.read_csv("mp_routes.csv")
    new_table = MyPyTable(data)
    # converting column data to list
    difficulties = data['Rating'].tolist()
    stars = data['Avg Stars'].tolist()
    type = data['Route Type'].tolist()
    #stars = list(map(str, stars))

    isTrad = []
    isSport = []
    isTrad, isSport = accTradSport(type)

    #print(unique(difficulties))
    print(len(unique(difficulties)))
    print(get_agregate(unique(difficulties)))

    print()
    print(unique(stars))
    print(len(unique(stars)))
    #print(get_agregate(unique(stars)))

    #print(unique(type))
    print()
    new_agg = []
    list_agg_type = get_agregate(unique(type))
    for item in list_agg_type:
        if(',' in item):
            print('wtf')
            item = 'awd'
        else:
            new_agg.append(item)

    print(len(unique(type)))
    print(list_agg_type)

    print()
    header = ['trad','sport']
    X = list(zip(isTrad,isSport))

    dt = MyDecisionTreeClassifier()
    dt.fit(X,stars)
    dt.print_decision_rules1(header)
    dt.print_decision_rules(header)
    print(new_agg)
    #nb = MyNaiveBayesClassifier()
    #nb.fit(X,stars)
    #print(nb.predict([['True','True'],['True','False'],['False','True'],['False','False']]))

if(__name__ == '__main__'):
    main()
