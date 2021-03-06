# Might make your life easier for appending to lists
from collections import defaultdict

# Third party libraries
import numpy as np
# Only needed if you plot your confusion matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential


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
    model = Sequential()
    risso_prepared_data = np.vstack(risso_train)
    pacific_prepared_data = np.vstack(pacific_train)
    print(risso_prepared_data)
    print(pacific_prepared_data)



if __name__ == "__main__":
    data_directory = "path\to\data"  # root directory of data
    dolphin_classifier(data_directory)
