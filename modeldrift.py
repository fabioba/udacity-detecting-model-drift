
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
        Check if all the past scores are worse than current score

        Args:
            list_scores(list): list scores
    """
    try:
        logger.info('START')



        min_past_score=min(list_scores)

        logger.info('min past score: {}'.format(min_past_score))

        if min_past_score<newr2:
            logger.info('RAW TEST: current r2 is NOT worse than past scores')
        elif min_past_score>newr2:
            logger.info('RAW TEST: current r2 is worse than past scores')


    except Exception as err:
        logger.exception(err)
        raise


def check_parametric_past_scores(list_scores):
    """
        Check if all past scores are worse the current one, using parametric test

        Args:
            list_scores(list): list scores
    """
    try:
        logger.info('START')


        # calculate parametric statistics
        mean_past_scores=np.mean(list_scores)
        std_past_scores=np.std(list_scores)

        logger.info('standard deviation past scores: {}'.format(std_past_scores))

        # this value is the upper limit of worse scoring value, if the current value is less than this, so there's model drift
        value_worse_score=mean_past_scores - std_past_scores*2

        logger.info('limit value worse score: {}'.format(value_worse_score))



        if value_worse_score<newr2:
            logger.info('PARAMETRIC TEST: current r2 is NOT worse than past scores')
        elif value_worse_score>newr2:
            logger.info('PARAMETRIC TEST: current r2 is worse than past scores')


    except Exception as err:
        logger.exception(err)
        raise



def check_non_parametric_past_scores(list_scores):
    """
        Check if all past scores are worse the current one, using NON-parametric test

        Args:
            list_scores(list): list scores
    """
    try:
        logger.info('START')

        # calculate interquartile range
        iqr = np.quantile(list_scores,0.75)-np.quantile(list_scores,0.25)

        logger.info('interquartile range: {}'.format(iqr))

        # this value is the upper limit of worse scoring value, if the current value is less than this, so there's model drift
        value_worse_score = iqr<np.quantile(list_scores,0.25)-iqr*1.5


        logger.info('limit value worse score: {}'.format(value_worse_score))



        if value_worse_score<newr2:
            logger.info('NON-PARAMETRIC TEST: current r2 is NOT worse than past scores')
        elif value_worse_score>newr2:
            logger.info('NON-PARAMETRIC TEST: current r2 is worse than past scores')


    except Exception as err:
        logger.exception(err)
        raise



if __name__=='__main__':
    LIST_SCORES=read_data('data/previousscores.txt')

    check_past_scores(LIST_SCORES)

    check_parametric_past_scores(LIST_SCORES)

    check_non_parametric_past_scores(LIST_SCORES)