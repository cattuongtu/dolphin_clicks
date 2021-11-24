# Might make your life easier for appending to lists
from collections import defaultdict
from lib.buildmodels import build_model

# Third party libraries
import numpy as np
# Only needed if you plot your confusion matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense

# our libraries
from lib.partition import split_by_day
import lib.file_utilities as util

# Any other modules you create


def dolphin_classifier(data_directory):
    """
    Neural net classification of dolphin echolocation clicks to species
    :param data_directory:  root directory of data
    :return:  None
    """

    plt.ion()   # enable interactive plotting

    use_onlyN = np.Inf  # debug, only read this many files for each species

    # Get, parse, and split files for Risso(Gg)
    risso_file_list = util.get_files("./features/features/Gg", stop_after=25)
    risso_parsed_files = util.parse_files(risso_file_list)
    risso_files_by_day = split_by_day(risso_parsed_files)

    # Get, parse, and split files for Pacific(Lo)
    pacific_file_list = util.get_files("./features/features/Lo", stop_after=25)
    pacific_parsed_files = util.parse_files(pacific_file_list)
    pacific_files_by_day = split_by_day(pacific_parsed_files)

    # Training and test data for Risso(Gg) and Pacific(Lo)
    all_risso_records = [(k, records) for k, record in risso_files_by_day.items() for records in record]
    risso_train, risso_test = train_test_split(list(all_risso_records))
    all_pacific_records = [(k, records) for k, record in pacific_files_by_day.items() for records in record]
    pacific_train, pacific_test = train_test_split(list(all_pacific_records))


    # Preparing data and training the model with the data

    # risso_prepared_data = np.vstack(risso_train)
    # print(len(risso_prepared_data))
    # pacific_prepared_data = np.vstack(pacific_train)
    # print(len(pacific_prepared_data))

    all_train_data = np.concatenate((np.vstack(risso_train), np.vstack(pacific_train)), axis=0)
    all_test_data = np.concatenate((np.vstack(risso_test), np.vstack(pacific_test)), axis=0)

    model = build_model([(Dense, [10], {'activation':'relu', 'input_dim': 20}),
     (Dense, [10], {'activation':'relu', 'input_dim':10}),
     (Dense, [2], {'activation':'softmax', 'input_dim':10})
    ])
    model.compile(optimizer = "Adam",loss = "categorical_crossentropy",metrics = ["accuracy"])
    model.summary()

    examples = get_features(all_train_data)
    labels = to_categorical(get_labels(all_train_data),num_classes=2)

    # for i in range(len(examples)):
    #     print(f"Examples {i}: {len(examples[i])}")
    # print(labels)

    test_examples = get_features(all_test_data)
    test_labels = to_categorical(get_labels(all_test_data),num_classes=2)

    # print(examples)
    # print(onehotlabels)

    model.fit(examples, labels, batch_size=100, epochs=10)

    # results = model.evaluate(test_examples, test_labels)
    # print(results[1])

    # print(risso_prepared_data[0][1].features)
    # print(np.shape(pacific_prepared_data))

def get_features(data):
    features = []

    for days in data:
        features.append(days[1].features)

    return features

def get_labels(data):
    labels = []

    for days in data:
        if (days[1].label == 'Gg'):
            labels.append(0)

        else:
            labels.append(1)

    return labels

if __name__ == "__main__":
    data_directory = "path\to\data"  # root directory of data
    dolphin_classifier(data_directory)