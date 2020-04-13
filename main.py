"""
Author : G Bettsworth
"""

import pandas as pd

# read in all data
# note to self - good read of data
a_wh = pd.read_table(r"../binary_classifiers_comparison_data/a_wh_question_datapoints.txt",
                     sep='\s', header=0)
