# Group names:
# Daniel Cazarez
# Lea Albano
# Jennah Kanan
# Aditya Shah

import csv
import sys
from time import strptime

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Loads data from a .csv file called <filename> and extracts evidence
    and labels from the file to be returned as a tuple.
    """

    def get_date_int(date_str: str) -> int:
        """Converts date shorthand to int"""
        return strptime(date_str[:3].strip(), '%b').tm_mon

    bool_dict = {
        'TRUE': 1,
        'FALSE': 0,
    }

    visitor_type_dict = {
        'Returning_Visitor': 1,
        'New_Visitor': 0,
        'Other': 0,
    }

    evidence = []
    labels = []

    with open(filename, 'r') as csv_file:
        # opt to use dictionary csv_reader as opposed to numerical indexed reader
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            # append current label

            labels.append(bool_dict[row['Revenue']])

            # opt to use list literal as opposed to appending
            curr_evidence = [
                int(row['Administrative']),
                float(row['Administrative_Duration']),
                int(row['Informational']),
                float(row['Informational_Duration']),
                int(row['ProductRelated']),
                float(row['ProductRelated_Duration']),
                float(row['BounceRates']),
                float(row['ExitRates']),
                float(row['PageValues']),
                float(row['SpecialDay']),
                get_date_int(row['Month']),
                int(row['OperatingSystems']),
                int(row['Browser']),
                int(row['Region']),
                int(row['TrafficType']),
                visitor_type_dict[row['VisitorType']],
                bool_dict[row['Weekend']]
            ]

            # convert date type to int
            evidence.append(curr_evidence)

    if len(evidence) != len(labels):
        sys.exit('There was an issue reading the csv, mismatched evidence and label list lengths.\n')

    return evidence, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, this returns a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(evidence, labels)

    return neigh


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    this returns a tuple (sensitivity, specificity), where sensitivity
    represents the true positive rate and specificity represents the true
    negative rate
    """
    true_positives = 0
    true_negatives = 0

    for label, prediction in zip(labels, predictions):
        if prediction == 1 and label == 1:
            true_positives += 1
        elif prediction == 0 and label == 0:
            true_negatives += 1

    sensitivity = true_positives / len(labels)
    specificity = true_negatives / len(labels)

    return sensitivity, specificity


if __name__ == "__main__":
    main()
