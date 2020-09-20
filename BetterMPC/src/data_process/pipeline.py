# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 23:04:23 2020

@author: ghdbs
"""

from sklearn.pipeline import Pipeline
from src.data_process import preprocessor as pp
import stage1_config

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from numpy.testing import *
import dill as pkl

wrmf_data_pipe = Pipeline(
    [
        ('uir_matrix_maker',
            pp.Uir_matrix_maker(variables=stage1_config.TARGET_VARS)),
         
        ('csr_matrix_maker',
            pp.Csr_matrix_maker())]
    )

if __name__ == "__main__":
    pass

