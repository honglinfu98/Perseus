"""
We serperate the data into train, test and validate sets.
"""

from os import path
import pickle
from perseus.settings import PROJECT_ROOT


def separate_by_year():
    """
    Separate the data into yearly sets from 2018 to 2023.
    """

    with open(path.join(PROJECT_ROOT, "data", "all_scored.pkl"), "rb") as file:
        df = pickle.load(file)

    df = df[~df["telegram_chat_id"].isna()]

    # Create a dictionary to hold the dataframes for each year
    yearly_dfs = {}

    # Define the years we are interested in
    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

    # Filter the dataframe for each year and store in the dictionary
    for year in years:
        yearly_dfs[year] = df[df["source_posted_at"].dt.year == year]

    # Save each year's dataframe to a separate file
    for year, year_df in yearly_dfs.items():
        with open(path.join(PROJECT_ROOT, "data", f"{year}_signals.pkl"), "wb") as file:
            pickle.dump(year_df, file)

    return


def seperate_train_test_validate(start_date="2024-01-10", end_date="2024-02-05"):
    """
    Seperate the data into train, test and validate sets.
    """

    with open(path.join(PROJECT_ROOT, "data", "all_810.pkl"), "rb") as file:
        df = pickle.load(file)

    df = df[~df["telegram_chat_id"].isna()]

    train_df = df[(df["source_posted_at"] < start_date)]

    validate_df = df[
        (df["source_posted_at"] >= start_date) & (df["source_posted_at"] <= end_date)
    ]

    test_df = df[(df["source_posted_at"] > end_date)]

    with open(path.join(PROJECT_ROOT, "data", "train_signals.pkl"), "wb") as file:
        pickle.dump(train_df, file)

    with open(path.join(PROJECT_ROOT, "data", "validate_signals.pkl"), "wb") as file:
        pickle.dump(validate_df, file)

    with open(path.join(PROJECT_ROOT, "data", "test_signals.pkl"), "wb") as file:
        pickle.dump(test_df, file)

    return train_df, validate_df, test_df


def seperate_dataset(start_date="2018-01-01", end_date="2024-02-05"):
    """
    Seperate the data into train, test and validate sets.
    """

    with open(path.join(PROJECT_ROOT, "data", "all_810.pkl"), "rb") as file:
        df = pickle.load(file)

    df = df[~df["telegram_chat_id"].isna()]

    train_df = df[(df["source_posted_at"] < start_date)]

    validate_df = df[
        (df["source_posted_at"] >= start_date) & (df["source_posted_at"] <= end_date)
    ]

    test_df = df[(df["source_posted_at"] > end_date)]

    return train_df, validate_df, test_df


# def get_train_scored_signals():
#     """
#     Get signals from the database for pre-pump scoring.
#     """

#     with open(
#         path.join(PROJECT_ROOT, "data", "old", "train_signals.pkl"), "rb"
#     ) as file:
#         signals = pickle.load(file)
#     result = signals[~signals["telegram_chat_id"].isna()]

#     return result


# def get_valid_scored_signals():
#     """
#     Get signals from the database for pre-pump scoring.
#     """

#     with open(
#         path.join(PROJECT_ROOT, "data", "old", "validate_signals.pkl"), "rb"
#     ) as file:
#         signals = pickle.load(file)
#     result = signals[~signals["telegram_chat_id"].isna()]

#     return result


# def get_test_scored_signals():
#     """
#     Get signals from the database for pre-pump scoring.
#     """

#     with open(path.join(PROJECT_ROOT, "data", "old", "test_signals.pkl"), "rb") as file:
#         signals = pickle.load(file)
#     result = signals[~signals["telegram_chat_id"].isna()]

#     return result


def get_btc_train_scored_signals():
    """
    Get signals from the database for pre-pump scoring.
    """

    with open(path.join(PROJECT_ROOT, "data", "train_signals_btc.pkl"), "rb") as file:
        signals = pickle.load(file)
    result = signals[~signals["telegram_chat_id"].isna()]

    return result


def get_btc_valid_scored_signals():
    """
    Get signals from the database for pre-pump scoring.
    """

    with open(path.join(PROJECT_ROOT, "data", "valid_signals_btc.pkl"), "rb") as file:
        signals = pickle.load(file)
    result = signals[~signals["telegram_chat_id"].isna()]

    return result


def get_btc_test_scored_signals():
    """
    Get signals from the database for pre-pump scoring.
    """

    with open(path.join(PROJECT_ROOT, "data", "test_signals_btc.pkl"), "rb") as file:
        signals = pickle.load(file)
    result = signals[~signals["telegram_chat_id"].isna()]

    return result


def get_new_detection():
    """
    Get signals from the database for pre-pump scoring.
    """

    with open(path.join(PROJECT_ROOT, "data", "new_detection.pkl"), "rb") as file:
        signals = pickle.load(file)
    result = signals[~signals["telegram_chat_id"].isna()]

    return result


if __name__ == "__main__":
    # a = get_new_detection()
    # seperate_train_test_validate()
    # aa = get_train_scored_signals()
    # bb = get_valid_scored_signals()
    # cc = get_test_scored_signals()
    a = get_btc_train_scored_signals()
    b = get_btc_valid_scored_signals()
    c = get_btc_test_scored_signals()
    # d = get_new_detection()
