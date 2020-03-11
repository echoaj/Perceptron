import numpy as np
import pickle
import pandas as pd


class NeuralNetwork:
    """Contains the necessary methods to process the neural network."""
    def __init__(self):
        np.random.seed(1)
        self.weights = None

    # sigmoital normalization function
    @staticmethod
    def sigmoid_normalization(m):
        return 1 / (1 + np.exp(-m))

    # derivative of the sigmoital normalization function
    @staticmethod
    def sigmoid_normalization_derivative(m):
        return m * (1 - m)

    # method to train the neural network
    def train(self, inputs, outputs, epoch=1, verbose=0):
        num_of_weights = len(inputs[0])
        self.weights = 2 * np.random.random((num_of_weights, 1)) - 1    # STEP 2:  SYNAPSE -WEIGHTS

        if verbose == 2:
            print("\033[1mInitial Weights:\033[0m\n", self.weights)

        prediction = 0
        for i in range(epoch):
            sum_ = np.dot(inputs, self.weights)                         # STEP 3:  NEURON -SUMMATION
            prediction = self.sigmoid_normalization(sum_)               # STEP 4:  OUTPUT LAYER
            error = outputs - prediction                                # STEP 5:  ERROR LOSS
            snd = self.sigmoid_normalization_derivative(prediction)     # STEP 6:  ADJUST THE WEIGHTS
            adjustment = np.dot(inputs.T, error * snd)
            self.weights += adjustment

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
        sum_ = np.dot(new_input, self.weights)
        prediction = self.sigmoid_normalization(sum_)
        return prediction

    # save the neural network.
    def save(self):
        with open("model.pkl", "wb") as file:
            pickle.dump(self, file)


def main():
    # STEP 1:  INPUT LAYER
    training_inputs = pd.read_csv('DogsVsCatsTraining.csv', usecols=[1, 2, 3, 4, 5, 6, 7]).values
    training_outputs = pd.read_csv('DogsVsCatsTraining.csv', usecols=[8]).values

    model = NeuralNetwork()
    model.train(training_inputs, training_outputs, epoch=20000, verbose=1)

    '''
    model.save()
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    '''

    new_data = np.array([[1, 0,	0,	0,	1,	1,	1]])
    result = model.predict(new_data)
    result = result[0][0]
    print("Input:", new_data[0])
    print("Output:", result)
    print()

    if result:
        print("It is a Dog")
    else:
        print("It is a Cat")


if __name__ == '__main__':
    main()
