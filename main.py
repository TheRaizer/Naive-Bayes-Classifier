import numpy as np

"""
This project was inspired by and utilizes much of the info described in https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
However the code differs as this aims to implement naive bayes using numpy.
"""


def load_data(path, delimiter, usecols, skiprows):
    """ Load csv data from a given path into numpy arrays.

    Parameters
    ----------
    path:string
        The path to the csv.
    delimiter:string
        The delimiter separating each data point.
    usecols:tuple
        The columns to use in the csv.
    skiprows:tuple
        The rows to skip in the csv.

    Returns
    -------
    X.T: The input data where each row represents a category of the data, and the columns represent the data sample.
    Y.T: The labels where there exists one row containing the label for each data sample.

    """
    X = np.loadtxt(open(path, 'rb'), delimiter=delimiter, usecols=usecols, skiprows=skiprows)
    # get every row of data except only the last column which contains the labels
    Y = X[:, X.shape[1] - 1]
    # increase dimension of Y from 1D to 2D
    Y = Y[..., np.newaxis]
    # remove the last column as it contains the labels, which aren't input data
    X = np.delete(X, X.shape[1] - 1, axis=1)
    return X.T, Y.T


def separate_by_class(X, Y):
    """ Seperate the input data by class.

    Parameters
    ----------
    X:ndarray
        The input data.
    Y:ndarray
        The labels.

    Returns
    -------
    separated:dict
        The keys are each label in the dataset. The values are a ndarray with shape
        (X.shape[0], # of data samples that contain the key label). Note X.shape[0]/X rows
        is the number of attributes that each X data sample contains.
    """
    separated = {}

    # iterate through each data sample in the input.
    for i in range(X.shape[1]):
        if Y[0, i] not in separated:
            # if the label is not in the dict add it as the key
            separated[Y[0, i]] = np.zeros((X.shape[0], 0))

        # get the entire column of attributes from data sample i.
        temp = X[0:, i]
        # increase dimensions from 1D to 2D
        temp = temp[..., np.newaxis]
        # take the initial array of zeros we created and append the attributes to its columns.
        separated[Y[0, i]] = np.append(separated[Y[0, i]], temp, axis=1)

    return separated


def summarize_data(X, Y):
    """ Summarize the data into 3 key values, the mean, standard deviation and the number of data samples.

    Parameters
    ----------
    X:ndarray
        The input data.
    Y:ndarray
        The labels.

    Returns
    -------
    summaries: dict
        Dictionary with key as the label, and value as a tuple of 3 items.

    """
    separated_data = separate_by_class(X, Y)
    summaries = {}

    for label, input in separated_data.items():
        # input is an ndarray where each column is a data sample and each row is an input variable

        # Ends up as a single column where each row is the mean of that input variable.
        means = np.mean(input, axis=1, keepdims=True)
        # Ends up as a single column where each row is the standard deviation of that input variable.
        stds = np.std(input, axis=1, keepdims=True)
        summaries[label] = (means, stds, input.shape[1])

    return summaries


def gaussian_prob_density_func(X, means, stds):
    """ Calculates a gaussian probability density function given the input, means, and standard deviations.
    It maps the input X onto the normal distribution and returns those mapped values.

    Parameters
    ----------
    X:ndarray
        input data.
    means:ndarray
        the means of the attributes of the input data.
    stds:ndarray
        the standard deviations of the attributes of the input data.

    Returns
    -------
    An ndarray containing each input value mapped onto the normal distribution.

    """
    exponent = np.exp(-np.square((X - means)) / (2 * np.square(stds)))
    val = (1 / (np.sqrt(2 * np.pi) * stds)) * exponent
    return val


def calculate_class_probabilities(class_summaries, X):
    """ Probabilities are calculated using bayes theorem P(class|data) = (P(data|class) * P(class)) / P(data).
    In our case we don't need the exact probability we just want a value that is still maximized
    meaning we still take the highest value to be the true class. This allows us to avoid integrating
    between two points on the normal distribution, and dividing by the P(data) which stops needless computations.

    In this naive bayes implementation the probability is calculated through the formula
    P(class=0|X1,X2) = P(X1|class=0) * P(X2|class=0) * P(class=0) where each input variable's probability
    of being in that class is being multiplied together which treats it independently hence naive.


    Parameters
    ----------
    class_summaries:tuple
        The summaries of the training data
    X:ndarray
        The input data that will be classified

    Returns
    -------
    Dictionary where each key is a label and the value is an array where for every index there is
    the total "probability" of that label being true for the input relating to that index.
    Eg. index 0 contains the total "probability" of that label being true for the first set of input variables.

    """
    class_probabilities = {}
    total_data_count = 0

    for label, class_summary in class_summaries.items():
        total_data_count += class_summary[2]

    # check for each class/label
    for label, class_summary in class_summaries.items():
        _, _, data_count = class_summary
        if label not in class_probabilities:
            # probability that the given class is the correct one (P(class))
            class_probabilities[label] = data_count / total_data_count
        means, stds, _ = class_summary
        # probabilities that each input variable belongs to the given class/label are calculated using the
        # prob density function of that class_summary.
        # Each input variable probability is multiplied together
        class_probabilities[label] *= np.prod(gaussian_prob_density_func(X, means, stds), axis=0)

    return class_probabilities


def testModel():
    # load the training data
    X_loaded, Y_loaded = load_data('csv_data/IrisTraining.csv', ',', (1, 2, 3, 4, 5), 1)

    # summarize the training data
    class_summaries = summarize_data(X_loaded, Y_loaded)

    # load the test data
    X, Y = load_data('csv_data/IrisTest.csv', ',', (1, 2, 3, 4, 5), 1)

    # calculate the probabilities using the normal distribution obtained from the training data.
    class_probabilities = calculate_class_probabilities(class_summaries, X)

    print(class_probabilities)

    # loop through all inputs
    for i in range(X.shape[1]):
        # take the highest probability
        label = max(class_probabilities[0][i], class_probabilities[1][i], class_probabilities[2][i])

        # check if true label is same as predicted label
        if label == class_probabilities[0][i]:
            label = 0
            print(label == Y[0][i])
        if label == class_probabilities[1][i]:
            label = 1
            print(label == Y[0][i])
        if label == class_probabilities[2][i]:
            label = 2
            print(label == Y[0][i])


def testModelBinary():
    # load the training data
    X_loaded, Y_loaded = load_data('csv_data/IrisTrainingBinary.csv', ',', (1, 2, 3, 4, 5), 1)

    # summarize the training data
    summaries = summarize_data(X_loaded, Y_loaded)

    # load the test data
    X, Y = load_data('csv_data/IrisTestBinary.csv', ',', (1, 2, 3, 4, 5), 1)

    # calculate the probabilities using the normal distribution obtained from the training data.
    class_probabilities = calculate_class_probabilities(summaries, X)

    print(class_probabilities)

    # loop through all inputs
    for i in range(X.shape[1]):
        # take the highest probability
        label = max(class_probabilities[0][i], class_probabilities[1][i])

        # check if true label is same as predicted label
        if label == class_probabilities[0][i]:
            label = 0
            print(label == Y[0][i])
        if label == class_probabilities[1][i]:
            label = 1
            print(label == Y[0][i])


if __name__ == '__main__':
    testModel()

