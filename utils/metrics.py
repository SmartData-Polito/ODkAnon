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

class GeneralizationMetric:
    """
    Ḡ = (1/V+) × Σ(|o| + |d|) × v_{o→d}
    """
    def __init__(self, k_threshold: int = 10):
        self.k_threshold = k_threshold

    def calculate_generalization_error(self, od_matrix_generalized: pd.DataFrame, od_matrix: pd.DataFrame) -> float:
        # generalized -> number of original cells
        origin_counts = self._build_hexagon_counts(
            od_matrix_generalized, od_matrix, column_gen="start_gen", column_orig="start_h3"
        )
        destination_counts = self._build_hexagon_counts(
            od_matrix_generalized, od_matrix, column_gen="end_gen", column_orig="end_h3"
        )

        total_volume_anonymous = 0
        weighted_count_sum = 0

        for _, row in od_matrix_generalized.iterrows():
            flow_value = row["count"]
            if flow_value >= self.k_threshold:
                origin_h3 = row["start_gen"]
                dest_h3   = row["end_gen"]

                origin_count = origin_counts.get(origin_h3, 1)
                dest_count   = destination_counts.get(dest_h3, 1)

                total_volume_anonymous += flow_value
                weighted_count_sum += (origin_count + dest_count) * flow_value

        return weighted_count_sum / total_volume_anonymous if total_volume_anonymous > 0 else 0.0

    def _build_hexagon_counts(
        self, od_matrix_generalized: pd.DataFrame, od_matrix: pd.DataFrame, 
        column_gen: str, column_orig: str
    ) -> dict:
        """
        Count how many original hexagons belong to each generalized hexagon.
        """
        generalized_hexagons = od_matrix_generalized[column_gen].unique()
        original_hexagons = od_matrix[column_orig].unique()

        counts = {}
        for gen_hex in generalized_hexagons:
            target_res = h3.get_resolution(gen_hex)

            # Find all parents of originals at target resolution
            parent_series = [h3.cell_to_parent(h, target_res) for h in original_hexagons]

            # Count how many times the parent == gen_hex appears
            count = sum(1 for p in parent_series if p == gen_hex)
            counts[gen_hex] = max(count, 1)  # fallback to 1

        return counts
