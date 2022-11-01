import os

import pandas as pd
import sys
import traceback


class MissingColumnError(AttributeError):
    """Error indicating that an imported DataFrame lacks necessary columns"""
    pass


def load_arguments_from_tsv(filepath, default_usage='test'):
    """
        Reads arguments from tsv file

        Parameters
        ----------
        filepath : str
            The path to the tsv file
        default_usage : str, optional
            The default value if the column "Usage" is missing

        Returns
        -------
        pd.DataFrame
            the DataFrame with all arguments

        Raises
        ------
        MissingColumnError
            if the required columns "Argument ID" or "Premise" are missing in the read data
        IOError
            if the file can't be read
        """
    try:
        dataframe = pd.read_csv(filepath, encoding='utf-8', sep='\t', header=0)
        if not {'Argument ID', 'Premise'}.issubset(set(dataframe.columns.values)):
            raise MissingColumnError('The argument "%s" file does not contain the minimum required columns [Argument ID, Premise].' % filepath)
        if 'Usage' not in dataframe.columns.values:
            dataframe['Usage'] = [default_usage] * len(dataframe)
        return dataframe
    except IOError:
        traceback.print_exc()
        raise


def load_arguments(argument_filepath, default_usage='train'):
    # load arguments
    df_arguments = load_arguments_from_tsv(argument_filepath, default_usage=default_usage)
    if len(df_arguments) < 1:
        print('There are no arguments in file "%s"' % argument_filepath)
        sys.exit(2)
    return df_arguments


def load_labels_from_tsv(filepath, label_order):
    """
        Reads label annotations from tsv file

        Parameters
        ----------
        filepath : str
            The path to the tsv file
        label_order : list[str]
            The listing and order of the labels to use from the read data

        Returns
        -------
        pd.DataFrame
            the DataFrame with the annotations

        Raises
        ------
        MissingColumnError
            if the required columns "Argument ID" or names from `label_order` are missing in the read data
        IOError
            if the file can't be read
        """
    try:
        dataframe = pd.read_csv(filepath, encoding='utf-8', sep='\t', header=0)
        # dataframe = dataframe[['Argument ID'] + label_order]
        return dataframe
    except IOError:
        traceback.print_exc()
        raise
    except KeyError:
        raise MissingColumnError('The file "%s" does not contain the required columns for its level.' % filepath)


def combine_columns(df_arguments, df_labels):
    """Combines the two `DataFrames` on column `Argument ID`"""
    return pd.merge(df_arguments, df_labels, on='Argument ID')


def split_arguments(df_arguments):
    """Splits `DataFrame` by column `Usage` into `train`-, `validation`-, and `test`-arguments"""
    train_arguments = df_arguments.loc[df_arguments['Usage'] == 'train'].drop(['Usage'], axis=1).reset_index(drop=True)
    valid_arguments = df_arguments.loc[df_arguments['Usage'] == 'validation'].drop(['Usage'], axis=1).reset_index(drop=True)
    test_arguments = df_arguments.loc[df_arguments['Usage'] == 'test'].drop(['Usage'], axis=1).reset_index(drop=True)

    return train_arguments, valid_arguments, test_arguments


def format_dataset(data_dir, argument_filepath, validate=False):
    df_arguments = load_arguments(argument_filepath)
    df_train_all = []
    df_valid_all = []

    for label_filepath in [os.path.join(data_dir, 'labels-training.tsv'),
                                 os.path.join(data_dir, 'level1-labels-training.tsv')
                                 ]:
        if not os.path.isfile(label_filepath):
            print('The required file ``{}`` is not present in the data directory'.format(label_filepath))
            sys.exit(2)
        # read labels from .tsv file
        df_labels = load_labels_from_tsv(label_filepath, 'arguments-training')
        # join arguments and labels
        df_full_level = combine_columns(df_arguments, df_labels)
        # split dataframe by usage
        train_arguments, valid_arguments, _ = split_arguments(df_full_level)
        df_train_all.append(train_arguments)
        df_valid_all.append(valid_arguments)

    if len(df_train_all[0]) < 1:
        print('There are no arguments listed for training.')
        sys.exit()

    if validate and len(df_valid_all[0]) < 1:
        print('There are no arguments listed for validation. Proceeding without validation.')

    return df_train_all, df_valid_all

if __name__ == '__main__':
    LEVELS = ["1", "2"]
    NUM_LEVELS = len(LEVELS)

    v = format_dataset('./value_data/','./value_data/arguments-training.tsv')
    print(v)
