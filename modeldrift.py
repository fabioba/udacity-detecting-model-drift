
"""
This module is part of the Udacity - MLOps course.
The aim of this project is to automatically store scoring of the pre-trained model.

"""

import pandas as pd
import logging
import ast
import numpy as np
from ast import literal_eval

newr2=0.3625


logging.basicConfig(format='%(funcName)s %(asctime)s %(message)s',level='INFO')
logger=logging.getLogger(__name__)


def read_data(path):
    """
        Read past scores

        Args:
            path(str): path of the file

        Output:
            list_scores(list): list scores
    """
    try:
        logger.info('START')



        with open(path) as f:
            list_scores = literal_eval(f.read()) 
                
        logger.info('len:{}'.format(len(list_scores)))


        return list_scores

    except Exception as err:
        logger.exception(err)
        raise


if __name__=='__main__':
    DF=read_data('data/previousscores.txt')

