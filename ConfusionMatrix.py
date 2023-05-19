import numpy as np


def scoring_formulas(fp, fn, tp, tn):
    accuracy = (tp + tn) / (fp + fn + tp + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1_score


class ConfusionMatrix:

    def __init__(self, actual_results, predicted_results):
        self.false_positive = 0
        self.false_negative = 0

        self.true_positive = 0
        self.true_negative = 0

        self.actual_results = actual_results
        self.predicted_results = predicted_results

    def Calculate_Confusion_Matrix(self):
        for self.actual_results, self.predicted_results in zip(self.actual_results, self.predicted_results):
            if self.predicted_results == self.actual_results:
                if self.predicted_results == 1:
                    self.true_positive += 1
                else:
                    self.true_negative += 1
            else:
                if self.predicted_results == 1:
                    self.false_positive += 1
                else:
                    self.false_negative += 1

        Matrix = np.array([[self.true_negative, self.false_positive],
                           [self.false_negative, self.true_positive]])

        accuracy, precision, recall, f1_score = \
            scoring_formulas(self.false_positive, self.false_negative, self.true_positive, self.true_negative)

        print("Accuracy: {}%".format(round(accuracy * 100, 2)))
        print("Precision: {}%".format(round(precision * 100, 2)))
        print("Recall: {}%".format(round(recall * 100, 2)))
        print("F1-Score: {}%".format(round(f1_score * 100, 2)))

        return Matrix
