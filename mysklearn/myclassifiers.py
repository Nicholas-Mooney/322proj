'''
Author: Nicholas Mooney
4/6/2022
PA6
'''
import math
import copy
import random
import numpy
from mysklearn import myutils


class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).
    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data
    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.
        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        slope_numerator = 0
        slope_denominator = 0

        yhat = sum(y_train)/len(y_train)
        xhat = sum(X_train)/len(X_train)

        for i in range(len((y_train))):
            slope_numerator += (X_train[i]-xhat)*(y_train[i]-yhat)
            slope_denominator += (X_train[i]-xhat)**2
        slope = slope_numerator / slope_denominator
        intercept = (yhat-(xhat*slope))

        self.regressor = slope, intercept
        return slope, intercept

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        prediction = []
        for x in X_test:
            prediction.append(self.discretizer(self.regressor[0]*x + self.regressor[1]))
        return prediction
class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.
    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.
        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closest neighbors of each test instance.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        ret_distances = []
        ret_indices = []
        #for every new test instance
        for x in X_test:
            check_distance = []
            add_indexes = []
            add_distance = []
            #go through the train list
            for _,train in enumerate(self.X_train):
                sum1 = 0
                #calc the dist go through every value of the train
                for k,_ in enumerate(x):
                    sum1 += (x[k] - train[k]) ** 2
                    dist = sum1 ** 0.5

                #add dist
                check_distance.append(dist)
            new_check_dist = copy.deepcopy(check_distance)

            #print(new_check_dist)
            #take lowest knn
            for i in range(self.n_neighbors):
                add_distance.append(min(new_check_dist))    #dist
                add_indexes.append(check_distance.index(add_distance[i]))#index
                new_check_dist.remove(min(new_check_dist)) #remove

            ret_indices.append(add_indexes)
            ret_distances.append(add_distance)
        return ret_distances, ret_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """

        neighbors = self.kneighbors(X_test)
        prediction = []

        #for every test instance
        #go through neighbors
        for neighbor_list in enumerate(neighbors[1]):
            count = []
            result = []
            for neighbor_index in neighbor_list[1]:

                if self.y_train[neighbor_index] not in result:
                    result.append(self.y_train[neighbor_index])
                    count.append(1)
                else:
                    index = result.index(self.y_train[neighbor_index])
                    count[index] = count[index] + 1
            #find majority
            majority_index = count.index(max(count))
            #append prediction
            prediction.append(result[majority_index])
        return prediction
class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.
    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()
    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.
        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        count = []
        result = []
        for i,y in enumerate(y_train):
            #count them
            if y not in result:
                result.append(y)
                count.append(1)
            else:
                index = result.index(y)
                count[index] = count[index] + 1
        self.most_common_label = result[count.index(max(count))]
        pass

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        prediction = []
        for i in X_test:
            prediction.append(self.most_common_label)
        return prediction
class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        #calc priors
        #X_train priors dict
        prior_list = []
        attribute_list = []
        num_attributes = len(X_train[0])
        #print(num_attributes)
        #print()
        #print('FIT PRIORS')
        for x in range(num_attributes):
            #print('X', x)
            dict1 = {}
            for instance in X_train:
                #print('inst', instance, end='-')
                value = str(instance[x])
                #print('value',value)
                #print(value in dict1, dict1)
                if value in dict1:
                    dict1[value] += 1
                else:
                    dict1[value] = 1
            for key in dict1:
                dict1[key] = dict1[key]/len(X_train)
            #print('dict - ', dict1)
            prior_list.append(dict1)

        #Y_train priors dict
        #print('Y', x)
        dict1 = {}
        for instance in y_train:
            value = str(instance)
            #print('inst', instance, end='-')
            #print('value',value)
            #print(value in dict1, dict1)
            if value in dict1:
                dict1[value] += 1
            else:
                dict1[value] = 1
        for key in dict1:
            dict1[key] = dict1[key]/len(X_train)
        #print('dict - ', dict1)
        prior_list.append(dict1)
        #print(prior_list)
        #print()
        #print('FIT POSTS')
        #calc posteriors
        classifiers = prior_list[len(prior_list)-1]
        #print(classifiers)
        post_list = []
        for x in range(num_attributes):
            #print('X', x)
            dict1 = {}
            column = myutils.get_column(X_train, x)
            uniques = myutils.get_uniques(column)
            for unique_attribute in uniques:
                class_dict = {}
                for classific in classifiers:
                    #print(classific)
                    class_dict[classific] = 0
                dict1[str(unique_attribute)] =(class_dict)
            #print(dict1)

            for j, instance in enumerate(X_train):
                #print('inst', instance, end='-')
                value = str(instance[x])
                #print('value', value)
                #print(dict1)
                if value in dict1:
                    dict1[value][y_train[j]] += 1
                else:
                    dict1[value][y_train[j]] = 1
                #print(dict1)

            #normalize to 0-1 for length of set
            for key in dict1:
                for classific in classifiers:
                    num_classifiers = len(X_train)*prior_list[len(prior_list)-1][classific]
                    dict2 = dict1[key]
                    value = dict2[classific]
                    dict1[key][classific] = value/num_classifiers
            #print('dict - ', dict1)
            post_list.append(dict1)

        #store
        self.priors = prior_list
        self.posteriors = post_list
        #print(self.priors)
        #print(self.posteriors)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        prediction_list = []
        classifiers = self.priors[len(self.priors)-1]
        #print()
        #print('PREDICT')
        #print(classifiers)
        for i, x_test in enumerate(X_test):
            #print('test:', i)
            classifier_probabilities = {}
            for classifier in classifiers:
                #find probability to add
                # P(Ci | X) = P(Ci) * P(X | Ci)
                #xtest
                #classifier
                #Pc = product(all attributes) # priors
                prior_list = []
                for att_index, attribute in enumerate(x_test):
                    att_dict = self.priors[att_index]
                    #print(att_dict)
                    prior_list.append(att_dict[str(attribute)])
                prior_prob = numpy.prod(prior_list)
                #print(prior_prob)

                #P(X | Ci) = product of all attributes given yes # posts
                post_list = []
                for att_index, attribute in enumerate(x_test):
                    att_dict = self.posteriors[att_index]
                    #print(att_dict)
                    post_list.append(att_dict[str(attribute)][classifier])
                post_prob = numpy.prod(post_list)
                #print(post_prob)
                probability = post_prob * prior_prob
                classifier_probabilities[classifier] = probability
            #print(classifier_probabilities)
            #print(list(classifier_probabilities))
            #print()
            #choose prediction
            key_list = list(classifier_probabilities.keys())
            val_list = list(classifier_probabilities.values())
            prediction = max(list(classifier_probabilities.values()))
            prediction = key_list[val_list.index(prediction)]

            prediction_list.append(prediction)
        return prediction_list

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train

        attributes_above = []

        #pick attr
        num_attr = len(X_train[0])
        rand_attr_index = random.randint(0, num_attr-1)
        #print()

        rand_attr_index = self.calc_attr(self.y_train, [], [])
        attributes_above.append(rand_attr_index)

        #append attr
        #print("using attr" + str(rand_attr_index))
        self.tree = ['Attribute', "att" + str(rand_attr_index)]

        #find values
        value_column = myutils.get_column(X_train, rand_attr_index)
        attibutes_values = myutils.get_uniques(value_column)
        attibutes_values.sort()

        #append values
        print(attibutes_values)
        for value in attibutes_values:
            print(value)
            value_subtree = []
            value_subtree = self.create_subtree(copy.deepcopy(attributes_above),[value])

            print(value_subtree)
            self.tree.append(["Value", value, value_subtree])


        #self.print_decision_rules()
        '''
        append attribute and all values

        for all values
            make leaf
                if all same class
            otherwise split on attribute
                append all new values
            if no more attributes
                make leaf
                if clash
                    idk'''
        pass

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        x_guesses = []
        for x_inst in X_test:
            guess = 'idk'
            spot = self.tree
            #print()
            #print('start')
            while guess == 'idk':
                values = []

                if spot[0] == "Leaf":
                    #print(spot)
                    guess = spot[1]

                elif spot[0] == "Attribute":
                    #print(spot)
                    att_string = spot[1]
                    att_int = int(att_string[3])
                    value_inst = x_inst[att_int]
                    #print(value_inst)

                    for i in range(len(spot)-2):
                        values.append(spot[i+2])

                    new_spot = 'notfoundyet'
                    #for value in values:
                    #    print(value)

                    for value_check in values:
                        #print(new_spot, value_inst, value_check[1])
                        if value_inst == value_check[1]:
                            new_spot = value_check
                            #print('FOUND',new_spot, value_inst, value_check[1])

                    spot = new_spot[2]

            x_guesses.append(guess)
        return x_guesses

    def print_decision_rules1(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        print(self.tree)
        print()
        if self.tree[0] == "Attribute":
            print(self.tree[0], end=", ")
            print(self.tree[1])
            for value in range(len(self.tree)-2):
                self.print_val(self.tree[value+2],0)
        else:
            print(self.tree[0])

    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass

    def print_attr(self,attr, num_tabs):
        if attr[0] == "Attribute":
            for i in range(num_tabs):
                print('\t', end="")
            print('[',attr[0], ',', attr[1], ',')
            for value in range(len(attr)-2):
                self.print_val(attr[value+2], num_tabs)

        elif attr[0] == "Leaf":
            for i in range(num_tabs):
                print('\t', end="")
            print(attr)

    def print_val(self, value, num_tabs):
        for i in range(num_tabs):
            print('\t', end="")
        print('[', value[0], ',', value[1], ',')
        self.print_attr(value[2], num_tabs+1)
        for i in range(num_tabs):
            print('\t', end="")
        print(']')
        pass

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        #print(self.tree)
        print()
        if self.tree[0] == "Attribute":
            for value in range(len(self.tree)-2):
                str1 = (" IF ")
                str1 += self.attr_name(attribute_names, str(self.tree[1])) + " == "
                self.print_val_final(str1, self.tree[value+2],attribute_names)
        else:
            print(self.tree[0])

    def print_attr_final(self, str1, attr,attribute_names):
        if attr[0] == "Attribute":
            str1 += ' AND ' + self.attr_name( attribute_names,str(attr[1])) + ' == '
            for value in range(len(attr)-2):
                self.print_val_final(str1, attr[value+2],attribute_names)

        elif attr[0] == "Leaf":
            print(str1, 'THEN class =', attr[1])

    def print_val_final(self, str1, value,attribute_names):
        str1 += str(value[1])
        self.print_attr_final(str1, value[2],attribute_names)

    def create_subtree(self, attributes_used,values_used,):
        subtree = []
        print()
        print('subtree')
        print(attributes_used, values_used)
        subtrain_x = self.get_sub_train(attributes_used, values_used)
        subtrain_y = self.get_sub_train_y(attributes_used, values_used)

        values_copy = copy.deepcopy(values_used)
        values_copy.pop(len(values_copy)-1)
        attributes_used_copy = copy.deepcopy(attributes_used)
        attributes_used_copy.pop(len(attributes_used_copy)-1)
        subtrain_y_above = self.get_sub_train_y(attributes_used_copy, values_copy)

        #if all match
        #print('sub y uniques',myutils.get_uniques(subtrain_y))
        if len(myutils.get_uniques(subtrain_y)) == 1:
            subtree = ['Leaf', subtrain_y[0], len(subtrain_y), len(subtrain_y_above)]
        #return leaf of class

        #if no elements or no instances
        elif len(self.X_train[0]) == len(attributes_used):
            if len(subtrain_y) == 0:
                #no instances
                subtree = ['Leaf', 'out of attributes']
            else:
                #no attributes return majority
                subtree = ['Leaf', self.get_majority(subtrain_y), len(subtrain_y), len(subtrain_y_above)]

        #pick attr to split on
        else:
            num_attr = len(self.X_train[0])
            rand_attr_index = random.randint(0, num_attr-1)
            rand_attr_index = self.calc_attr(self.y_train, attributes_used, values_used)
            while rand_attr_index in attributes_used:
                rand_attr_index = random.randint(0, num_attr-1)

            attributes_used.append(rand_attr_index)
            subtree = ['Attribute', "att" + str(rand_attr_index)]
            #create value lists
            subtree

            #find values
            value_column = myutils.get_column(self.X_train, rand_attr_index)
            attibutes_values = myutils.get_uniques(value_column)
            attibutes_values.sort()

            #append values
            no_attr = False
            for value in attibutes_values:
                value_subtree = []
                value_subtree = self.create_subtree(copy.deepcopy(attributes_used), values_used + [value])
                subtree.append(["Value", value, value_subtree])
                if(value_subtree == ['Leaf', 'out of attributes']):
                    no_attr = True
            if no_attr:
                if len(subtrain_y) == 0:
                    #no instances
                    subtree = ['Leaf', 'out of attributes']
                else:
                    #no attributes return majority
                    subtree = ['Leaf', self.get_majority(subtrain_y), len(subtrain_y), len(subtrain_y_above)]
            #if any of the values is out of attr
            #just return majority

        #if leaf but clash
        #this is sorted out in get majority

        return subtree

    def calc_attr(self, current_ytrain, attributes_used, values_used):
        attributes_left = []
        #print(attributes_used)
        #print(values_used)
        for num in range(len(self.X_train[0])):
            if num not in attributes_used:
                attributes_left.append(num)
        #print(attributes_left)
        entropy_list = []
        #print(entropy_list)
        for attribute in attributes_left:
            entropy_values = []
            value_column = myutils.get_column(self.X_train, attribute)
            attibutes_values = myutils.get_uniques(value_column)
            attibutes_values.sort()
            for attr_value in attibutes_values:
                xs = self.get_sub_train(attributes_used+[attribute], values_used+[attr_value])
                ys = self.get_sub_train_y(attributes_used+[attribute], values_used+[attr_value])
                unique_ys = myutils.get_uniques(ys)
                #calc entropies
                #create sub_ytrains
                entropy = 0
                for y in unique_ys:
                    #print('entropy: ',ys.count(y), len(ys), end=" | ")
                    entropy += -(ys.count(y)/len(ys))*math.log2(ys.count(y)/len(ys))
                    #print(entropy)
                entropy_values.append(entropy*(len(ys)/len(current_ytrain)))
            #calc entropy of attribute
            entropy_list.append(sum(entropy_values))
            #print(entropy_list)
        #print(entropy_list)
        #pick from attribute
        #from entropies
        index_max = entropy_list.index(min(entropy_list))
        return attributes_left[index_max]

    def get_sub_train(self, attributes_used, values_used):
        if attributes_used == []:
            return self.X_train
        ret_train = []
        #print(attributes_used)
        #for each instance
        for i, instance in enumerate(self.X_train):
            is_match = True
            #match each attribute
            for j, value in enumerate(values_used):
                #print(j,value, attributes_used, values_used)
                attribute_index = attributes_used[j]
                if instance[attribute_index] != value:
                    is_match = False
            if is_match:
                ret_train.append(instance)
        return ret_train

    def get_sub_train_y(self, attributes_used, values_used):
        if attributes_used == []:
            return self.y_train
        ret_train = []
        #print(attributes_used)
        #for each instance
        for i, instance in enumerate(self.X_train):
            is_match = True
            #match each attribute
            for j, value in enumerate(values_used):
                attribute_index = attributes_used[j]
                if instance[attribute_index] != value:
                    is_match = False
            if is_match:
                ret_train.append(self.y_train[i])
        return ret_train

    def get_majority(self, list1):
        print('boop')
        #print(list1)
        if(isinstance(list1[0],float)):
            print('float')
            return sum(list1)/len(list1)
        uniques = myutils.get_uniques(list1)
        counts = []
        #print('y', list1)
        for unique in uniques:
            counts.append(list1.count(unique))
        index_max = counts.index(max(counts))
        if counts.count(max(counts)) > 1:
            maxes = []
            for i, maxq in enumerate(counts):
                if maxq == max(counts):
                    maxes.append(list1[i])
            maxes.sort()
            return maxes[0]
        else:
            return uniques[index_max]

    def attr_name(self, attribute_names, attribute_str):
        if attribute_names is None:
            return attribute_str

        att_int = int(attribute_str[3])
        return attribute_names[att_int]
