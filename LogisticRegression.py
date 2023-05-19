import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from ConfusionMatrix import ConfusionMatrix


def initial_weights(n_features):
    weights = np.zeros((1, n_features))
    bias = 0
    return weights, bias


def z(weights, bias, X):
    return np.dot(X, weights.T) + bias


def sigmoid(weights, bias, X):
    return 1 / (1 + np.exp(-z(weights, bias, X)))


def cost_function(m_samples, weights, bias, X, y):
    return (-1 / m_samples) * (np.sum(y * np.log(sigmoid(weights, bias, X)) +
                                      (1 - y) * np.log(1 - sigmoid(weights, bias, X))))


def normalisation(X):
    m_samples, n_features = X.shape
    # create array with zeros in correct shape
    normalised_data = np.zeros((m_samples, n_features))

    for i in range(n_features):
        max_value, min_value = X[:, i].max(), X[:, i].min()
        normalised_data[:, i] = (X[:, i] - min_value) / (max_value - min_value)
    return normalised_data


def model_optimisation(X, y, learning_rate, regularisation, iterations):
    m_samples, n_features = X.shape

    # initialising weights and bias values
    weights, bias = initial_weights(n_features)

    # list for costs
    costs = []

    # run logistic regression algorithm
    for i in range(iterations):
        # probability
        y_hat = sigmoid(weights, bias, X)

        # cost
        iteration_cost = cost_function(m_samples, weights, bias, X, y)

        if i % 100 == 0 and i != 0:
            if costs[-1] == iteration_cost:
                print("converged in {} iterations...".format(i + 1))
                break

        # update cost list
        if i % 10 == 0:
            costs.append(iteration_cost)

        # update rules
        dw = (1 / m_samples) * np.dot(X.T, (y_hat - y))
        db = (1 / m_samples) * np.sum((y_hat - y))

        # updating weights and bias
        weights -= learning_rate * dw.T + regularisation_fcn(regularisation, m_samples, weights)
        bias -= learning_rate * db

    return weights, bias, costs


def regularisation_fcn(regularisation_term, m, weight_vector):
    return (regularisation_term / m) * weight_vector


def predict(weights, bias, X):
    y_predicted = sigmoid(weights, bias, X)
    y_predicted_array = [1 if i > 0.50 else 0 for i in y_predicted]
    return y_predicted_array


if __name__ == "__main__":
    # learning rate
    alpha = 1

    # regularisation term
    gamma = 0.75

    # number of iterations
    no_iter = 100000

    # process confusion matrix or not
    process_confusion = True

    # importing data
    df = pd.read_csv("diabetes.csv")

    X_data = df.iloc[:, :-1].values
    y_data = df.iloc[:, -1:].values

    # normalising data
    X_data = normalisation(X_data)

    # splitting data for train and testing
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.20, random_state=0)

    # run logistic regression
    w, b, cost_list = model_optimisation(X_train, y_train, alpha, gamma, no_iter)

    # get prediction using test data
    prediction = predict(w, b, X_test)

    # if process_confusion is true, get the confusion matrix
    if process_confusion:
        Confusion_Matrix_Calculate = ConfusionMatrix(y_test, prediction)
        Confusion_Matrix = Confusion_Matrix_Calculate.Calculate_Confusion_Matrix()

        # plotting confusion matrix
        plt.figure(2, figsize=(8, 6))
        df_confusion_matrix = pd.DataFrame(Confusion_Matrix, range(2), range(2))
        sns.set(font_scale=1)
        sns.heatmap(df_confusion_matrix, annot=True, annot_kws={"size": 10}, fmt="1.0f")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

    # plotting cost over iterations
    x_vals = np.arange(X_train.min(), X_train.max(), .01)
    x_vals = x_vals.reshape(len(x_vals), 1)
    p_vals = 1 / (1 + np.exp(-(b + (np.dot(x_vals, w)))))

    x_vals_costs = [i * 10 for i in list(range(1, len(cost_list) + 1))]
    plt.figure(1, figsize=(8, 6))
    plt.plot(x_vals_costs, cost_list)
    plt.title("Costs over iterations")
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")

    plt.show()
