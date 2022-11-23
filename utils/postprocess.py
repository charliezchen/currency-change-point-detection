import pandas as pd
import numpy as np
from datetime import timedelta,datetime

"""
Descriptions:
    This function resolves the changepoints from the output csv to valid changepoints.
    The strategy is:
    a) only changepoints with score higher than some threshold will be accepted
    b) to avoid multiple count, within each date window, only one changepoint will be accepted

Parameters:
    path: the path to the changepoint csv
    thr: the minimum cp_score to accept the changepoint
    window: if the date difference of two changepoints is less than 'window',
            only one changepoint will be accepted

Return:
    A list of changepoints in the format of [discovery_date, changepoint_date, score]
"""

def resolve_changepoints_from_path(path, thr=0.98, window=5):

    cpd_df = pd.read_csv(path)

    cpd_df = cpd_df[cpd_df.cp_score >= thr]
    cpd_df = cpd_df.sort_values('cp_location')
    cache = [] # date, cp(t), score, date(t) 
    changepoints = [] # date, cp_date, score
    
    # Caller should make sure cache is not empty
    def flush_cache():
        nonlocal cache, changepoints
        changepoints.append([datetime.strptime(cache[0], '%Y-%m-%d'), datetime.strptime(cache[0], '%Y-%m-%d') - timedelta(round(cache[3] - cache[1])), cache[2]])
        cache = []   

    changepoints = []
    for _, row in cpd_df.iterrows():
        if cache and row.cp_location - window > cache[1]:
            flush_cache()
            
        if not cache or row.cp_score > cache[2]:
            cache = ([row.date, row.cp_location, row.cp_score, row.t])

    if cache: flush_cache()

            
    print("Total number of changepoints is:", len(changepoints))
    return changepoints