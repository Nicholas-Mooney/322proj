#3hrs
###tree classifier
#fit predict print

#3hrs 7:30+
#dataset isolation
#more attributes
### three other classifiers
### classifier-evaluation
    #2hrs 10:30+
    ### analysis
    ### visuals
    ### API
        #2hrs   12:30+
        ##POWERPOINT
        #PRACTICE + SCRIPT 5minimport csv as c
import pandas as pd
import copy as copy
from readPrint import *
from mysklearn.mypytable import MyPyTable
from mysklearn.myutils import *
from mysklearn.myclassifiers import *
from mysklearn.myevaluation import *

def round_trad(num):
    new_num = int(num)
    if(num - new_num < 0.5):
        return new_num
    else:
        return new_num+1
def round_30(num):
    if(num <= 1):
        return 1
    else:
        return 2
class forest:
    def __init__(self, N, M, F):
        self.X_train = None
        self.y_train = None
        self.trees = []
        self.trees_acc = []
        self.trees_final = []
        self.tree_to_add = None
        self.N = N #num tree to generate bootstrapped
        self.M = M #trees selected from top accuracy
        self.F = F #num attributes to randomly calc entopy on???

    def fit(self, X_train, y_train, split):
        '''
        create N bootstraps
        create N trees with bootstraps
        take top M from N accuracies
        pass F as a param
        '''
        x = []
        y = []

        print('fit0')
        x_split, x_acc, y_split, y_acc = train_test_split(X_train, y_train,
                    test_size=split, random_state=12, shuffle=True)\

        print('fit1')
        tree_num = 0
        while(tree_num < self.N):
            print('N', tree_num)
            print('fit 1.1 - sampling')
            print(len(x_split))
            print(len(y_split))
            #split data
            x_temp_rand,x_test,y_temp_rand,y_test = bootstrap_sample(X=x_split, y=y_split,
                    n_samples=None, random_state=tree_num)

            print('fit 1.2 - tree')
            print(len(x_temp_rand))
            print(len(y_temp_rand))
            #build tree
            dt = MyDecisionTreeClassifier()
            dt.fit(x_temp_rand, y_temp_rand)
            dt.print_decision_rules1()

            print('fit 1.3 - accuracy')
            print(len(y_acc))
            print(len(x_acc))
            #acc tree against acc set
            accuracy = 0
            guesses = dt.predict(x_acc)
            print(len(guesses))
            print((guesses[0]))
            print((y_acc[0]))
            accuracy = accuracy_score(y_acc, guesses, normalize=True)
            print(accuracy)

            #store both
            self.trees.append(dt.tree)
            self.trees_acc.append(copy.deepcopy(accuracy))
            tree_num += 1

        print('fit2')
        print(self.trees_acc)
        #get top M trees
        list1 = self.trees
        list2 = self.trees_acc

        zipped_lists = zip(list1, list2)
        sorted_pairs = sorted(zipped_lists)

        tuples = zip(*sorted_pairs)
        list1, list2 = [ list(tuple1) for tuple1 in tuples]
        print(list1)
        for i in range(0, self.M):
            self.trees_final.append(list1[i])
        print(list2)
        pass

    def predict(self, x_test3):
        print('predict - forest')
        #take majority vote from M trees
        guesses = []
        dtemp = MyDecisionTreeClassifier()
        return_guesses = []
        for x in x_test3:
            guesses = []
            guess_temp = 0
            print(x)
            #print(self.trees_final)
            for tree in self.trees_final:
                #print('guess')
                #print(tree)
                dtemp.tree = tree
                #print(dtemp.tree)
                #print('guess1')
                guess_temp = dtemp.predict1([x])
                #print('guess2')
                guesses.append(guess_temp[0])
                #print('guess3')
            #print('guesses created')
            consensus = dtemp.get_majority(guesses)
            return_guesses.append(consensus)
        print('ret: ', return_guesses)
        return return_guesses
    def print(self):
        #print M trees
        pass

def tree_forest_test():
    data = pd.read_csv("mp_routes.csv")
    new_table = MyPyTable(data)
    difficulties = data['Rating'].tolist()
    stars = data['Avg Stars'].tolist()
    type = data['Route Type'].tolist()
    pitches = data['Pitches'].tolist()  #clean for greater than x

#alpine
#ice
#Rating
#Length

    #star to int
    for index,star in enumerate(stars):
        stars[index] = round_trad(star)
    for index,pitch in enumerate(pitches):
        pitches[index] = round_30(pitch)

    print(stars[0])
    print(pitches[0])
    print(get_uniques(pitches))


    isTrad = []
    isSport = []
    isAlpine = []
    isIce = []
    isAid = []
    isTrad, isSport = accTradSport(type)
    isAlpine, isIce, isAid = accAlpineIceAid(type)
    physicalities,danger = parseDiff(difficulties)
    header = ['trad', 'sport', 'pitches', 'alpine', 'ice', 'aid', 'physicalities']
    X = list(zip(isTrad, isSport, pitches, isAlpine, isIce, isAid, physicalities,danger))


    print('what1')
    forest1 = forest(1, 1, 12)
    print('what2')
    forest1.fit(X, stars, 0.9)
    print('what3')

    working = "awd"
    typeTrad = 'notTrad'
    typeSport = 'isSport'
    typePitch = 1
    typeAlpine = 'notAlpine'
    typeIce = 'notIce'
    typeAid = 'notAid'
    typeDif = '5.10'
    danger = 'none'
    forest1.predict([[typeTrad, typeSport, 1, typeAlpine, typeIce, typeAid,typeDif,danger]])
    forest1.predict([[typeTrad, typeSport, 2, typeAlpine, typeIce, typeAid,typeDif,danger]])
    #forest1.predict([["isTrad", "isSport",1]])
    #forest1.predict([["notTrad", "isSport",1]])
    #forest1.predict([["isTrad", "notSport",2]])
    #forest1.predict([["isTrad", "isSport",2]])
    #forest1.predict([["notTrad", "isSport",2]])
    print('what4')
    assert True
    pass

tree_forest_test()