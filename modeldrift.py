
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


def check_past_scores(list_scores):
    """
        Read past scores

        Args:
            list_scores(list): list scores
    """
    try:
        logger.info('START')



        min_past_score=min(list_scores)

        logger.info('min past score: {}'.format(min_past_score))

        if min_past_score<newr2:
            logger.info('current r2 is NOT worse than past scores')
        elif min_past_score>newr2:
            logger.info('current r2 is worse than past scores')


    except Exception as err:
        logger.exception(err)
        raise


if __name__=='__main__':
    LIST_SCORES=read_data('data/previousscores.txt')

    check_past_scores(LIST_SCORES)

