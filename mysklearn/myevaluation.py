'''
Author: Nicholas Mooney
4/6/2022
PA6
'''
import numpy as np
import copy
from mysklearn import myutils

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    xtrain = []
    ytrain = []
    xtest = []
    ytest=[]
    if shuffle:
        seed = random_state
        new_y = myutils.shuffle_list(y, seed)
        new_x = myutils.shuffle_list(X, seed)
    else:
        new_y = y
        new_x = X
    if test_size < 1:
        for index, x_inst in enumerate(new_x):
            if (index+1)/len(new_x) < 1-test_size:
                #test split
                xtrain.append(x_inst)
                ytrain.append(new_y[index])
            else:
                #train split
                xtest.append(x_inst)
                ytest.append(new_y[index])
    else:
        for index, x_inst in enumerate(new_x):
            if index+1 <= len(new_x)-test_size:
                #test split
                xtrain.append(x_inst)
                ytrain.append(new_y[index])
            else:
                #train split
                xtest.append(x_inst)
                ytest.append(new_y[index])
    return xtrain, xtest, ytrain, ytest
def kfold_cross_validation(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    train_folds = []
    test_folds = []
    new_xinst = []
    for i in range(0,len(X)):
        new_xinst.append(i)

    if shuffle:
        new_x = myutils.shuffle_list(new_xinst,random_state)
    else:
        new_x = new_xinst
    #print('-')
    #print(shuffle, ' ', random_state, ' ', newX)

    for fold_index in range(0,n_splits):
        new_train = []
        new_test = []
        for inst_index, index_actual in enumerate(new_x):
            #print(fold_index/n_splits, ' ',inst_index/len(newX), ' ', (fold_index+1)/n_splits)
            if fold_index/n_splits <= inst_index/len(X) < (fold_index+1)/n_splits:
                new_test.append(index_actual)
            else:
                new_train.append(index_actual)
        train_folds.append(new_train)
        test_folds.append(new_test)
    return train_folds, test_folds
def stratified_kfold_cross_validation(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    #print('-')
    #print(random_state, ' ', shuffle)

    counter_list = []
    for y_inst in y:
        if y_inst not in counter_list:
            counter_list.append(y_inst)

    newy = copy.deepcopy(y)
    index_tracker_test = []
    index_tracker_train = []
    for index in range(0,len(newy)):
        index_tracker_test.append(index)
        index_tracker_train.append(index)
    if shuffle:
        index_tracker_test = myutils.shuffle_list(index_tracker_test, random_state)
        index_tracker_train = copy.deepcopy(index_tracker_test)
    x_train = []
    x_test = []

    #make n lists for folds
    for _ in range(0, n_splits):
        x_train.append([])
        x_test.append([])

    #deal for each individual label
    num_dealt = 0
    current_fold_index = 0
    while num_dealt < len(y):
        if counter_list[0] in newy:
            #print('dealing: ', counter_list[0], counter_list, ' ', newy)
            index_temp = newy.index(counter_list[0])
            #add
            x_test[current_fold_index].append(index_tracker_test[index_temp])
            num_dealt += 1
            current_fold_index += 1
            if current_fold_index == n_splits:
                current_fold_index = 0
            #remove
            newy.pop(index_temp)
            index_tracker_test.pop(index_temp)
        else:
            #no more of label
            counter_list.pop(0)
    i = 0
    for test_fold in x_test:
        for index_rest in index_tracker_train:
            if index_rest not in test_fold and index_rest not in x_train[i]:
                x_train[i].append(index_rest)
        i +=1
    #print(x_train, ' ', x_test)
    return x_train, x_test
def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)

        n_samples(int): Number of samples to generate.
            If left to None (default) this
            is automatically
            set to the first dimension of X.

        random_state(int): integer used for seeding a random number generator for reproducible results
    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
    """
    print('boot1')
    print(len(X))
    print(len(y))
    print((n_samples))
    print((random_state))

    #set rand to seed0      and num samples to len
    if n_samples is None:
        n_samples = len(X)
    if random_state is None:
        random_state = 0
    np.random.seed(random_state)

    index_list = []
    for i in range(0, len(X)):
        index_list.append(i)
    chosen_indexes = []
    for _ in range(0, n_samples):
        chosen_indexes.append(index_list[int(np.random.randint(0, len(index_list)))])

    print('boot2')
    print(len(index_list))
    print(len(chosen_indexes))
    #print(random_state, ' ', n_samples, y)
    #print(index_list)
    #print(chosen_indexes)

    X_sample = []
    X_out_of_bag = []
    y_sample = []
    y_out_of_bag = []

    for i in chosen_indexes:
        X_sample.append(X[i])
        if not y is None:
            y_sample.append(y[i])


    print('boot3')
    print(len(X_sample))
    print(len(y_sample))
    print(len(X_out_of_bag))
    print(len(y_out_of_bag))

    for l, j in enumerate(index_list):
        #print(l)
        if j not in chosen_indexes:
            X_out_of_bag.append(X[j])
        if not y is None:
            if j not in chosen_indexes:
                y_out_of_bag.append(y[j])
    if y is None:
        y_sample = None
        y_out_of_bag = None

    print('boot5')
    #print(chosen_indexes, ' ', )
    #print('x ', X_sample)
    #print()
    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = []
    #print(labels)
    for label_i in labels:
        #print('actual ', label_i)
        row = []
        for label_j in labels:
            row.append(0)
        #print(matrix)
        for label_j in labels:
            #print('pred ', label_j)
            for i, y in enumerate(y_pred):
                if(y == label_j) and y_true[i] == label_i:
                    index = labels.index(y)
                    row[index] += 1
            #print(row)
        matrix.append(row)
    #print(matrix)
    return matrix
def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If false, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == true, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    print('')
    print('acc calc')
    print(len(y_pred))
    print(len(y_true))
    print(y_pred[0])
    print(y_true[0])

    accuracy = 0
    total = 0
    for index, value in enumerate(y_true):
        total += 1
        if(y_true[index] == y_pred[index]):
            accuracy += 1
    if normalize:
        accuracy /= total
    return accuracy
def old_accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If false, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == true, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    print('')
    print('acc calc')
    print(len(y_pred))
    print(len(y_true))
    print(y_pred[0])
    print(y_true[0])
    labels = myutils.get_uniques(y_true)
    matrix = confusion_matrix(y_true, y_pred, labels)
    score = 0
    total = 0
    for i, row in enumerate(matrix):
        for j, column in enumerate(row):
            if i == j:
                score += matrix[i][j]
            total += matrix[i][j]
    accuracy = score
    if normalize:
        accuracy /= total
    return accuracy
def accuracy_score_confidence_interval(accuracy, n_samples, confidence_level=0.95):
    """Compute the classification prediction accuracy score confidence interval.

    Args:
        accuracy(float): Classification accuracy to compute a confidence interval for
        n_samples(int): Number of samples in the test set used to compute the accuracy
        confidence_level(float): Level of confidence to use for computing a confidence interval
            0.9, 0.95, and 0.99 are supported. Default is 0.95

    Returns:
        lower_bound(float): Lower bound of the accuracy confidence interval
        upper_bound(float): Upper bound of the accuracy confidence interval

    Notes:
        Raise ValueError on invalid confidence_level
        Assumes accuracy and n_samples are correct based on training/testing
            set generation method used (e.g. holdout, cross validation, bootstrap, etc.)
            See Bramer Chapter 7 for more details
    """
    if confidence_level not in [0.95, 0.99, 0.9]:
        raise ValueError('invalid confidence interval')
    elif confidence_level == 0.9:
        confidence_mult = 1.64
    elif confidence_level == 0.95:
        confidence_mult = 1.96
    elif confidence_level == 0.99:
        confidence_mult = 2.58

    stnd_dev = ((accuracy*(1-accuracy))/n_samples)**0.5
    lower = accuracy-confidence_mult*stnd_dev
    higher = accuracy+confidence_mult*stnd_dev
    #print(stnd_dev, accuracy, n_samples, confidence_level, confidence_mult)
    #print(lower,higher)
    return lower, higher

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    negatives = 0
    positives = 0
    true_negatives = 0
    true_positives = 0
    false_negatives = 0
    false_poitives = 0
    positive_label = pos_label
    if labels[1] == positive_label:
        negative_label = labels[0]
    else:
        negative_label = labels[1]
    #print(y_true)
    #print('prec')
    for index in range(len(y_true)):
        #print(index, y_true[index], ' : ', y_pred[index],end='')
        if y_true[index] == y_pred[index]:
            #print('match ')
            if y_pred[index] == positive_label:
                true_positives += 1
                positives += 1
            else:
                true_negatives += 1
                negatives += 1
            #print(true_positives,end='')
            #print(false_poitives,end='')
            #print(true_negatives,end='')
            #(false_negatives)
        else:
            #print('no match ')
            if y_pred[index] == negative_label:
                false_negatives += 1
                negatives += 1
            else:
                false_poitives += 1
                positives += 1
            #print(true_positives,end='')
            #print(false_poitives,end='')
            #print(true_negatives,end='')
            #(false_negatives)
    #(true_positives)
    #print(false_poitives)
    #print('TN',true_negatives)
    #print(false_negatives)
    if false_poitives + true_positives == 0:
        if true_positives > 0:
            return 1
        else:
            return 0
    else:
        ret_val = true_positives / (false_poitives + true_positives)
        return ret_val

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    negatives = 0
    positives = 0
    true_negatives = 0
    true_positives = 0
    false_negatives = 0
    false_poitives = 0
    positive_label = pos_label
    if labels[1] == positive_label:
        negative_label = labels[0]
    else:
        negative_label = labels[1]
    #print(y_true)
    #print('recall')
    for index in range(len(y_true)):
        #print(index, y_true[index], ' : ', y_pred[index],end='')
        if y_true[index] == y_pred[index]:
            #print('match ')
            if y_pred[index] == positive_label:
                true_positives += 1
                positives += 1
            else:
                true_negatives += 1
                negatives += 1
            #print(true_positives,end='')
            #print(false_poitives,end='')
            #print(true_negatives,end='')
            #print(false_negatives)
        else:
            #print('no match ')
            if y_pred[index] == negative_label:
                false_negatives += 1
                negatives += 1
            else:
                false_poitives += 1
                positives += 1
            #print(true_positives,end='')
            #(false_poitives,end='')
            #print(true_negatives,end='')
            #print(false_negatives)
    #print(true_positives)
    #print(false_poitives)
    #print('TN',true_negatives)
    #print(false_negatives)
    if false_negatives + true_positives == 0:
        if true_positives > 0:
            return 1
        else:
            return 0
    else:
        ret_val = true_positives / (false_negatives + true_positives)
        return ret_val

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    #print(y_true)
    #print(y_pred)
    negatives = 0
    positives = 0
    true_negatives = 0
    true_positives = 0
    false_negatives = 0
    false_poitives = 0
    positive_label = pos_label
    if labels[1] == positive_label:
        negative_label = labels[0]
    else:
        negative_label = labels[1]
    #print(y_true)
    #print('f1')
    for index in range(len(y_true)):
        #print(index, y_true[index], ' : ', y_pred[index],end='')
        if y_true[index] == y_pred[index]:
            #print('match ')
            if y_pred[index] == positive_label:
                true_positives += 1
                positives += 1
            else:
                true_negatives += 1
                negatives += 1
            #print(true_positives,end='')
            #print(false_poitives,end='')
            #print(true_negatives,end='')
            #print(false_negatives)
        else:
            #print('no match ')
            if y_pred[index] == negative_label:
                false_negatives += 1
                negatives += 1
            else:
                false_poitives += 1
                positives += 1
            #print(true_positives,end='')
            #print(false_poitives,end='')
            #print(true_negatives,end='')
            #print(false_negatives)
    #print(true_positives)
    #print(false_poitives)
    #print('TN',true_negatives)
    #print(false_negatives)
    if false_poitives + true_positives != 0:
        precision = true_positives / (false_poitives + true_positives)
    else:
        precision = 0
        if true_positives != 0:
            precision = 1

    if false_negatives + true_positives != 0:
        recall = true_positives / (false_negatives + true_positives)
    else:
        recall = 0
        if true_positives != 0:
            recall = 1
    if precision + recall == 0:
        return 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
