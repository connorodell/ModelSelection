# Forward stepwise selection for best predictor subset selection

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class ForwardSelection:
    '''Class for selecting the best predictor subset to train a linear model
    such and LinearRegression or LogisticRegression'''

    def __init__(self, pipe):
        self.pipe = pipe

    forward_results = None

    def plot_results(self, size=(10, 6)):
        fig, ax = plt.subplots(figsize=size)
        ax.plot([len(x) for x in self.forward_results[:, 0]], self.forward_results[:, 1])
        plt.title('R^2 versus number of best predictors')
        plt.xlabel('Number of predictors in model')
        plt.ylabel('R^2')

    def bestfit(self, X, y, include=[], random_state=None):
        '''Class method that finds the best predictors using forward selection.
        User may specify predictors to include in all models via the optional include argument.'''
        # Split input into test and train sets
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

        # List to store best result for each number of predictors
        best_per_level = []

        # Algorithm starts with no predictors
        my_predictors = [] + include

        # List of predictors that have not been chosen and remove any the user specifies for inclusion
        remaining_predictors = X.columns.tolist()
        for item in my_predictors:
            remaining_predictors.remove(item)

        def give_score(score_data):
            return score_data[1]

        # Find the total number of iterations in the forward selection algorithm (nth triangular number)
        total_iterations = len(remaining_predictors)*(len(remaining_predictors)+1)/2
        print('The total number of iterations is {}.'.format(total_iterations))

        # Forward stepwise algorithm
        current_iteration = 0
        while len(remaining_predictors) > 0:
            testing_data = []  # results for current level
            for predictor in remaining_predictors:
                current_predictors = my_predictors + [predictor]
                self.pipe.fit(x_train[current_predictors], y_train)
                score = self.pipe.score(x_test[current_predictors], y_test)
                testing_data.append([current_predictors, score, predictor])

                # Progress bar
                current_iteration += 1
                progress = 100 * current_iteration / total_iterations
                print('Current progress: {:.2f}%'.format(progress), end='\r', flush=True)

            # Find the best predictors at current level and store result to list
            testing_data.sort(key=give_score)
            my_predictors.append(testing_data[-1][2])
            best_per_level.append((testing_data[-1][0], testing_data[-1][1]))

            # Remove chosen predictor from list of remaining predictors
            remaining_predictors.remove(testing_data[-1][2])

        print('Current progress: 100.00%')

        # Save results to class parameter
        self.forward_results = np.array(best_per_level)

        # Find the best overall model and print result
        best_per_level.sort(key=give_score)
        print(best_per_level[-1])
        print('The best linear model found uses {} predictors.'.format(len(best_per_level[-1][0])))