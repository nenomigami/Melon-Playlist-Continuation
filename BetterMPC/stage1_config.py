# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 22:14:56 2020

@author: ghdbs
"""

# data
import os
WORKING_DIRECTORY = os.path.dirname(os.path.abspath(__file__)) + "/"
RAW_DATA_FOLDER = WORKING_DIRECTORY + "data/raw/"
SPLITED_DATA_FOLDER = WORKING_DIRECTORY + "data/splited/"
PROCESSED_DATA_FOLDER = WORKING_DIRECTORY + "data/processed/"

TRAINING_DATA_FILE = RAW_DATA_FOLDER + "train.json"
VALIDATING_DATA_FILE = RAW_DATA_FOLDER + "val.json"
TESTING_DATA_FILE = RAW_DATA_FOLDER + "test.json"

SPLITED_TRAINING_DATA_FILE = [SPLITED_DATA_FOLDER + "data_I.json",
                              SPLITED_DATA_FOLDER + "data_II_q.json",
                              SPLITED_DATA_FOLDER + "data_II_a.json",
                              SPLITED_DATA_FOLDER + "data_III_q.json",
                              SPLITED_DATA_FOLDER + "data_III_a.json"]

STAGE1_TRAIN = [SPLITED_TRAINING_DATA_FILE[0],
                VALIDATING_DATA_FILE,
                TESTING_DATA_FILE,
                SPLITED_TRAINING_DATA_FILE[1],
                SPLITED_TRAINING_DATA_FILE[3]]

STAGE1_VALID_Q = [SPLITED_TRAINING_DATA_FILE[1],
                  SPLITED_TRAINING_DATA_FILE[3]]

STAGE1_VALID_A = [SPLITED_TRAINING_DATA_FILE[2],
                  SPLITED_TRAINING_DATA_FILE[4]]

STAGE2_TRAIN = PROCESSED_DATA_FOLDER + "stage2_train.pkl"
STAGE2_TEST1 = PROCESSED_DATA_FOLDER + "stage2_test1.pkl"
STAGE2_TEST2 = PROCESSED_DATA_FOLDER + "stage2_test2.pkl"

TARGET_VARS = ["songs", "tags"]

STAGE1_PROCESSED_TRAIN_FILE = PROCESSED_DATA_FOLDER + "processed_train.pkl"

SAVE_FOLDER = WORKING_DIRECTORY + "model/"

# input variables 
FEATURES = ['MSSubClass', 'MSZoning', 'Neighborhood',
            'OverallQual', 'OverallCond', 'YearRemodAdd',
            'RoofStyle', 'MasVnrType', 'BsmtQual', 'BsmtExposure',
            'HeatingQC', 'CentralAir', '1stFlrSF', 'GrLivArea',
            'BsmtFullBath', 'KitchenQual', 'Fireplaces', 'FireplaceQu',
            'GarageType', 'GarageFinish', 'GarageCars', 'PavedDrive',
            'LotFrontage',
            # this one is only to calculate temporal variable:
            'YrSold']

# this variable is to calculate the temporal variable,
# must be dropped afterwards
DROP_FEATURES = 'YrSold'

# numerical variables with NA in train set
NUMERICAL_VARS_WITH_NA = ['LotFrontage']

# categorical variables with NA in train set
CATEGORICAL_VARS_WITH_NA = ['MasVnrType', 'BsmtQual', 'BsmtExposure',
                            'FireplaceQu', 'GarageType', 'GarageFinish']

TEMPORAL_VARS = 'YearRemodAdd'

# variables to log transform
NUMERICALS_LOG_VARS = ['LotFrontage', '1stFlrSF', 'GrLivArea']

# categorical variables to encode
CATEGORICAL_VARS = ['MSZoning', 'Neighborhood', 'RoofStyle', 'MasVnrType',
                    'BsmtQual', 'BsmtExposure', 'HeatingQC', 'CentralAir',
                    'KitchenQual', 'FireplaceQu', 'GarageType', 'GarageFinish',
                    'PavedDrive']
