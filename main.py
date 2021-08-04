import numpy as np
import math


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
    """

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
    """

    Parameters
    ----------
    X:ndarray
        The input data.
    Y:ndarray
        The labels.

    Returns
    -------
    summaries: dict
        Dictionary of tuple for each label. First item is the means of the label's data sample's attributes.
        The second item is the standard deviations of the label's data sample's attributes.
        The third item is the number of attributes that all data samples contain.

    """
    separated_data = separate_by_class(X, Y)
    summaries = {}

    for label, input in separated_data.items():
        # input is an ndarray where each column is a data sample and each row is an attribute

        # Ends up as a single row containing the mean of each data sample's attributes.
        means = np.mean(input, axis=1, keepdims=True)
        # Ends up as a single row containing the standard deviation of each data sample's attributes.
        stds = np.std(input, axis=1, keepdims=True)
        summaries[label] = (means, stds, input.shape[1])

    return summaries


def gaussian_prob_density_func(X, means, stds):
    exponent = np.exp(-np.square((X - means)) / (2 * np.square(stds)))
    return (1 / (np.sqrt(2 * np.pi) * stds)) * exponent


def calculate_class_probabilities(summaries, X):
    class_probabilities = {}
    total_data_count = 0

    for label, summary in summaries.items():
        total_data_count += summary[2]

    for _, _ in summaries.items():  # loop through each item
        for label, summary in summaries.items():  # calculate the probability of each column in 'input' being of label 'lbl'
            _, _, num_labeled = summary
            if label not in class_probabilities:
                class_probabilities[label] = math.log(num_labeled / total_data_count, 10)  # log probabilities.
            means, stds, _ = summary
            class_probabilities[label] *= np.log10(np.prod(gaussian_prob_density_func(X, means, stds), axis=0))

    return class_probabilities


if __name__ == '__main__':
    X_summary, Y_summary = load_data('IrisTrainingBinary.csv', ',', (1, 2, 3, 4, 5), 1)
    summaries = summarize_data(X_summary, Y_summary)
    X, Y = load_data('IrisTestBinary.csv', ',', (1, 2, 3, 4, 5), 1)
    class_probabilities = calculate_class_probabilities(summaries, X)

    print(class_probabilities)

    if class_probabilities[0][0] > class_probabilities[1][0]:
        print('correct')
