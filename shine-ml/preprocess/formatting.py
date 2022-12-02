import json
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
            raise MissingColumnError(
                'The argument "%s" file does not contain the minimum required columns [Argument ID, Premise].' % filepath)
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
        dataframe = dataframe[['Argument ID'] + label_order]
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
    valid_arguments = df_arguments.loc[df_arguments['Usage'] == 'validation'].drop(['Usage'], axis=1).reset_index(
        drop=True)
    test_arguments = df_arguments.loc[df_arguments['Usage'] == 'test'].drop(['Usage'], axis=1).reset_index(drop=True)

    return train_arguments, valid_arguments, test_arguments


def format_dataset(data_dir, argument_filepath, validate=False):
    df_arguments = load_arguments(argument_filepath)

    label_filepath = os.path.join(data_dir, 'labels-training.tsv')
    if not os.path.isfile(label_filepath):
        print('The required file "labels-level{}.tsv" is not present in the data directory'.format(LEVELS[i]))
        sys.exit(2)
    if not os.path.isfile(label_filepath):
        print('The required file ``{}`` is not present in the data directory'.format(label_filepath))
        sys.exit(2)
    # read labels from .tsv file
    df_labels = load_labels_from_tsv(label_filepath, VALUES["2"])
    # join arguments and labels
    df_full_level = combine_columns(df_arguments, df_labels)
    # split dataframe by usage
    train_arguments, valid_arguments, _ = split_arguments(df_full_level)

    return train_arguments


def load_json_file(filepath):
    """Load content of json-file from `filepath`"""
    with open(filepath, 'r') as json_file:
        return json.load(json_file)


def load_values_from_json(filepath):
    """Load values per level from json-file from `filepath`"""
    json_values = load_json_file(filepath)
    values = {"1": set(), "2": set()}
    for value in json_values["values"]:
        values["1"].add(value["name"])
        values["2"].add(value["level2"])

    values["1"] = sorted(values["1"])
    values["2"] = sorted(values["2"])

    return values


if __name__ == '__main__':
    LEVELS = ["1", "2"]
    NUM_LEVELS = len(LEVELS)
    VALUES = load_values_from_json('value_data/values.json')

    dataset = format_dataset('value_data/', './value_data/arguments-training.tsv')
    snippet_format = {'train': {}, 'test': {}}
    for id_row, row in dataset.iterrows():
        key = 'train'
        if id_row > 5000:
            key = 'test'
        # if id_row > 600:
        #     key = 'test'
        #     break
        snippet_format[key].update({
            str(id_row):
                {
                    'text': row['Premise'],
                    # 'label': row['Universalism: concern']
                    'label': [b for a, b in row.to_dict().items() if a not in ['Argument ID',
                                                                       "Conclusion",
                                                                       "Stance",
                                                                       "Premise"]]
                }

        })
    with open('snippets_split.json', 'w') as jfile:
        json.dump(snippet_format, jfile)
