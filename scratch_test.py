# Alex Joslin
# Description: A neural network that predicts if an animal is a cat or a dog based on binary input
# Date: March 16th, 2020

import math
import pickle
import pandas as pd


class MatrixFunctions:
    """Contains the necessary matrix methods for the perceptron."""
    # add, sub, div, mul constant to a matrix
    @staticmethod
    def scalar(m, num, operator):
        switch = False
        if type(m) == int:
            switch = True
            temp = num
            num = m
            m = temp

        m_row = len(m)
        m_col = len(m[0])

        new_matrix = [[0 for _ in range(m_col)] for _ in range(m_row)]

        for i in range(m_row):
            for j in range(m_col):
                if operator == '+':
                    new_matrix[i][j] = m[i][j] + num
                if operator == '-':
                    if switch:
                        new_matrix[i][j] = num - m[i][j]
                    else:
                        new_matrix[i][j] = m[i][j] - num
                if operator == '*':
                    new_matrix[i][j] = m[i][j] * num
                if operator == '/':
                    if switch:
                        new_matrix[i][j] = num / m[i][j]
                    else:
                        new_matrix[i][j] = m[i][j] / num

        return new_matrix

    # resulting matrix of e to the power of a matrix
    @staticmethod
    def exp(m):
        m_row = len(m)
        m_col = len(m[0])

        new_matrix = [[0 for _ in range(m_col)] for _ in range(m_row)]

        for i in range(m_row):
            for j in range(m_col):
                new_matrix[i][j] = math.e ** m[i][j]

        return new_matrix

    # subtract a matrix from another matrix
    @staticmethod
    def subtract(a, b):
        a_row = len(a)
        a_col = len(a[0])
        b_row = len(b)
        b_col = len(b[0])

        assert a_row == b_row
        assert a_col == b_col

        new_matrix = [[0 for _ in range(a_col)] for _ in range(a_row)]
        for i in range(a_row):
            for j in range(a_col):
                new_matrix[i][j] = a[i][j] - b[i][j]

        return new_matrix

    # multiply two matrices
    @staticmethod
    def multiply(a, b):
        a_row = len(a)
        a_col = len(a[0])
        b_row = len(b)
        b_col = len(b[0])

        assert a_row == b_row
        assert a_col == b_col

        new_matrix = [[0 for _ in range(a_col)] for _ in range(a_row)]
        for i in range(a_row):
            for j in range(a_col):
                new_matrix[i][j] = a[i][j] * b[i][j]

        return new_matrix

    # add two matrices
    @staticmethod
    def add(a, b):
        a_row = len(a)
        a_col = len(a[0])
        b_row = len(b)
        b_col = len(b[0])

        assert a_row == b_row
        assert a_col == b_col

        new_matrix = [[0 for _ in range(a_col)] for _ in range(a_row)]
        for i in range(a_row):
            for j in range(a_col):
                new_matrix[i][j] = a[i][j] + b[i][j]

        return new_matrix

    # transpose a matrix
    @staticmethod
    def transpose(m):
        m_row = len(m)
        m_col = len(m[0])

        new_matrix = [[0 for _ in range(m_row)] for _ in range(m_col)]

        for i in range(m_row):
            for j in range(m_col):
                new_matrix[j][i] = m[i][j]

        return new_matrix

    # dot product of two matrices
    @staticmethod
    def dot_product(a, b):
        a_row = len(a)
        a_col = len(a[0])
        b_row = len(b)
        b_col = len(b[0])

        assert a_col == b_row

        new_matrix = [[0 for _ in range(b_col)] for _ in range(a_row)]
        for i in range(a_row):
            for j in range(b_col):
                for k in range(b_row):
                    new_matrix[i][j] += a[i][k] * b[k][j]

        return new_matrix

    # round the elements of a matrix
    @staticmethod
    def round_matrix(m):
        m_row = len(m)
        m_col = len(m[0])

        new_matrix = [[0 for _ in range(m_col)] for _ in range(m_row)]

        for i in range(m_row):
            for j in range(m_col):
                new_matrix[i][j] = round(m[i][j])

        return new_matrix

    # sigmoital normalization function
    def sigmoid_normalization(self, m):
        neg = self.scalar(-1, m, '*')
        ep = self.exp(neg)
        denominator = self.scalar(1, ep, '+')
        total = self.scalar(1, denominator, '/')
        return total

    # derivative of the sigmoital normalization function
    def sigmoid_normalization_derivative(self, m):
        result = self.scalar(1, m, '-')
        total = self.multiply(m, result)
        return total


class NeuralNetwork(MatrixFunctions):
    """Contains the necessary methods to process the neural network."""
    def __init__(self):
        self.weights = None

    # method to train the neural network
    def train(self, inputs, outputs, epoch=1, verbose=1):
        num_of_weights = len(inputs[0])
        self.weights = [[0.5] for i in range(num_of_weights)]                       # STEP 2:  SYNAPSE -WEIGHTS

        if verbose == 2:
            print("\033[1mInitial Weights:\033[0m\n", self.weights)

        prediction = 0
        for i in range(epoch):
            sum_ = self.dot_product(inputs, self.weights)                           # STEP 3:  NEURON -SUMMATION
            prediction = self.sigmoid_normalization(sum_)                           # STEP 4:  OUTPUT LAYER
            error = self.subtract(outputs, prediction)                              # STEP 5:  ERROR LOSS
            snd = self.sigmoid_normalization_derivative(prediction)                 # STEP 6:  ADJUST THE WEIGHTS
            adjustment = self.dot_product(self.transpose(inputs), self.multiply(error, snd))
            self.weights = self.add(self.weights, adjustment)

            if verbose == 2:
                print("\n\033[31mEPOCH %d\033[0m" % (i+1))
                print("\033[1mPrediction:\033[0m\n", prediction)
                print("\033[1mError:\033[0m\n", error)
                print("\033[1mAdjustment:\033[0m\n", adjustment)
                print("\033[1mAdjusted Weights:\033[0m\n", self.weights, end='\n\n')

        if verbose == 1 or verbose == 2:
            print("\033[34m-------------Result-------------\033[0m")
            print("\033[10m\033[1mFinal Prediction:\033[0m\n", prediction, end='\n\n')      # RESULT

    # method is called to predict the new data after the train method is called
    def predict(self, new_input):
        sum_ = self.dot_product(new_input, self.weights)
        prediction = self.sigmoid_normalization(sum_)
        return prediction

    # save the neural network.
    def save(self):
        with open("model.pkl", "wb") as file:
            pickle.dump(self, file)


def main():
    # STEP 1:  INPUT LAYER
    training_inputs = pd.read_csv('DogsVsCatsTraining.csv', usecols=[1,2,3,4,5,6,7]).values
    training_outputs = pd.read_csv('DogsVsCatsTraining.csv', usecols=[8]).values

    model = NeuralNetwork()
    model.train(training_inputs, training_outputs, epoch=1000, verbose=1)

    '''
    model.save()
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    '''

    new_data = pd.read_csv('NewData.csv', usecols=[1,2,3,4,5,6,7]).values
    result = model.predict(new_data)
    result = int(round(result[0][0]))
    print("Input:", new_data[0])
    print("Output:", result)
    print()

    if result:
        print("It is a Dog")
    else:
        print("It is a Cat")


if __name__ == '__main__':
    main()
