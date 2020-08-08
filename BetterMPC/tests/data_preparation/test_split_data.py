# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 00:22:21 2020

@author: ghdbs
"""

import pytest
import pandas as pd
from pandas.testing import *
from src.data_preparation.DataGenerator import DataGenerator
import stage1_config
class TestMask(object):
    
    def test_song_only_case_on_one_row(self):
        test_argument = stage1_config.TRAINING_DATA_FILE
        expected = pd.read_json(stage1_config.SPLITED_TRAINING_DATA_FILE[0])
        actual = DataGenerator().split_data(test_argument)
        assert_frame_equal(expected, actual[0])
