"""
This script is used to output the label mapping CSV file for labeling.

It provides a function to read the edited CSV files (train, valid, test) and convert them back to a nested dictionary format for further processing.
"""

from os import path
import pandas as pd
from perseus.settings import PROJECT_ROOT


def read_labeling_csv_back_to_dict(train_test_valid: str):
    """
    Reads the edited labeling CSV file (train, valid, or test) and converts it into a nested dictionary.

    Args:
        train_test_valid (str): One of 'train', 'valid', or 'test' to specify which labeling file to read.

    Returns:
        dict: A nested dictionary mapping Symbol -> Code -> Value for the specified dataset split.
    """
    if train_test_valid == "train":
        train_edited_df = pd.read_csv(
            path.join(PROJECT_ROOT, "data", "train_labeling.csv")
        )

        edited_train_label_mapping = {}
        for _, row in train_edited_df.iterrows():
            if row["Symbol"] not in edited_train_label_mapping:
                edited_train_label_mapping[row["Symbol"]] = {}
            edited_train_label_mapping[row["Symbol"]][row["Code"]] = row["Value"]

        return edited_train_label_mapping

    elif train_test_valid == "valid":
        valid_edited_df = pd.read_csv(
            path.join(PROJECT_ROOT, "data", "valid_labeling.csv")
        )

        edited_valie_label_mapping = {}
        for _, row in valid_edited_df.iterrows():
            if row["Symbol"] not in edited_valie_label_mapping:
                edited_valie_label_mapping[row["Symbol"]] = {}
            edited_valie_label_mapping[row["Symbol"]][row["Code"]] = row["Value"]

        return edited_valie_label_mapping

    elif train_test_valid == "test":
        test_edited_df = pd.read_csv(
            path.join(PROJECT_ROOT, "data", "test_labeling.csv")
        )

        edited_test_label_mapping = {}
        for _, row in test_edited_df.iterrows():
            if row["Symbol"] not in edited_test_label_mapping:
                edited_test_label_mapping[row["Symbol"]] = {}
            edited_test_label_mapping[row["Symbol"]][row["Code"]] = row["Value"]

        return edited_test_label_mapping


if __name__ == "__main__":
    """
    Example usage: Reads the label mappings for train, test, and valid splits and stores them in variables a, b, and c.
    """
    a = read_labeling_csv_back_to_dict("train")
    b = read_labeling_csv_back_to_dict("test")
    c = read_labeling_csv_back_to_dict("valid")
