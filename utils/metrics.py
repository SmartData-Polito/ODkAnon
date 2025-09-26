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

def compute_discernability_and_cavg_sparse(dfn: pd.DataFrame, k: int) -> dict:
    
    od_groups = dfn.groupby(['ori_lon', 'ori_lat', 'dst_lon', 'dst_lat'])['count'].sum().reset_index()

    counts = od_groups['count'].values
    total_records = counts.sum()
    total_equiv_classes = len(counts)

    # C_DM: somma dei quadrati dei count >= k
    k_anonymous_counts = counts[counts >= k]
    c_dm = (k_anonymous_counts ** 2).sum()

    # CAVG: ((total_records / total_equiv_classes) / k)    
    c_avg = (total_records / total_equiv_classes) / k if total_equiv_classes > 0 else float('inf')

    return {
        'C_DM': c_dm,
        'C_AVG': c_avg,
        'total_records': total_records,
        'total_equivalence_classes': total_equiv_classes,
        'k': k
    }