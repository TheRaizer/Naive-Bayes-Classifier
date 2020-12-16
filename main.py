import numpy as np
import math
from matplotlib import pyplot as plt


def load_data(path, delimiter, usecols, skiprows):
    X = np.loadtxt(open(path, 'rb'), delimiter=delimiter, usecols=usecols, skiprows=skiprows)
    Y = X[:, X.shape[1] - 1]
    Y = Y[..., np.newaxis]
    X = np.delete(X, X.shape[1] - 1, axis=1)
    return X.T, Y.T


def separate_by_class(X, Y):
    separated = {}

    for i in range(X.shape[1]):
        if Y[0, i] not in separated:
            separated[Y[0, i]] = np.zeros((X.shape[0], 0))

        temp = X[0:, i]
        temp = temp[..., np.newaxis]
        separated[Y[0, i]] = np.append(separated[Y[0, i]], temp, axis=1)
    return separated


def summarize_data(X, Y):
    separated_data = separate_by_class(X, Y)
    summaries = {}

    for label, input in separated_data.items():
        means = np.mean(input, axis=1, keepdims=True)
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
                class_probabilities[label] = math.log(num_labeled / total_data_count, 10)  # log probabilities. Maybe?
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
