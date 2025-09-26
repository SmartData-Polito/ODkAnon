import pandas as pd
import os
import glob
import random
import numpy as np
import ast
import json
import os
import re
import itertools
import h3

from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from itertools import combinations
from collections import defaultdict
from collections import Counter
from typing import Dict, Set, List, Optional, Tuple





# compute metrics
# Equivalence classes are origin/destination pairs with count >= k.
# C_DM : sum of the squares of the sizes of the equivalence classes
# C_AVG : (total number of records / number of equivalence classes) / k

def compute_discernability_and_cavg(df: pd.DataFrame, k: int, suppressed_count: int = 0) -> dict:
    """
    compute C_DM e C_AVG for dataset OD generalization (with suppression).
    
    Args:
        df: DataFrame with column ['start_h3', 'end_h3', 'count']
        k: for k-anonimity
        suppressed_count: number of OD pairs suppressed (optional)
    
    """
    counts = df['count'].values
    total_records = counts.sum() + suppressed_count
    total_equiv_classes = len(counts) + suppressed_count
    
    # C_DM: somma dei quadrati dei count >= k
    k_anonymous_counts = counts[counts >= k]
    c_dm_gen = np.sum(k_anonymous_counts**2)
    
    # Penalità per record soppressi
    suppression_penalty = suppressed_count * counts.sum()  # o totale record, a seconda della definizione
    c_dm = c_dm_gen + suppression_penalty
    
    # C_AVG: (total_records / total_equiv_classes) / k
    c_avg = (total_records / total_equiv_classes) / k if total_equiv_classes > 0 else float('inf')
    
    return {
        'C_DM': c_dm,
        'C_AVG': c_avg,
        'total_records': total_records,
        'total_equivalence_classes': total_equiv_classes,
        'k': k}

def compute_discernability_and_cavg_weight(df: pd.DataFrame, k: int, suppressed_count: int = 0) -> dict:
    """
    Args:
        df: DataFrame with ['start_h3', 'end_h3', 'count']
        k: for k-anonimity
        suppressed_count: number of OD pairs suppressed (optional)
    
    Returns:
        dict con C_DM, C_AVG, total number of records and equivalence classes
    """
    counts = df['total_weight'].values
    total_records = counts.sum() + suppressed_count
    total_equiv_classes = len(counts) + suppressed_count
    
    k_anonymous_counts = counts[counts >= k]
    c_dm_gen = np.sum(k_anonymous_counts**2)
    
    # Penalty for suppressed records
    suppression_penalty = suppressed_count * counts.sum()  # o totale record, a seconda della definizione
    c_dm = c_dm_gen + suppression_penalty
    
    # C_AVG: (total_records / total_equiv_classes) / k
    c_avg = (total_records / total_equiv_classes) / k if total_equiv_classes > 0 else float('inf')
    
    return {
        'C_DM': c_dm,
        'C_AVG': c_avg,
        'total_records': total_records,
        'total_equivalence_classes': total_equiv_classes,
        'k': k
    }


