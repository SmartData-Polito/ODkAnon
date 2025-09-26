import pandas as pd
import os
from datetime import datetime, timedelta
import glob
import pandas as pd
import random
import numpy as np
import seaborn as sns
import ast
import json
import os
import re
import itertools
import folium
import h3
import time

import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.patches as patches
from itertools import combinations
from collections import defaultdict
from collections import Counter
from typing import Dict, Set, List, Optional, Tuple

from utils.h3hierarchy import create_h3_hierarchical_tree
from utils.metrics import compute_discernability_and_cavg_sparse
from utils.metrics import compute_discernability_and_cavg_sparse_ODkAnon
from utils.metrics import GeneralizationMetricODkAnon
from utils.metrics import fast_reconstruction_loss_ODkAnon
from utils.metrics import fast_reconstruction_loss_weight
from utils.visualization import CountAnalyzer
from utils.visualization import plot_count_distributions
from utils.visualization import H3FoliumVisualizerODkAnon

# Data preparation

df_gps = pd.read_csv("C:\\Users\\dmc\\Desktop\\extracted_trips_data.csv")

df_people = pd.read_csv("C:\\Users\\dmc\\Desktop\\individuals_dataset.csv")
df_people = df_people[df_people['GPS_RECORD'] == True]

df_merged = pd.merge(
    df_gps,
    df_people[['ID', 'WEIGHT_INDIV']],
    left_on='person_id',
    right_on='ID',
    how='inner'
).drop(columns='ID')

h3_resolution = 10

df_merged['start_h3'] = df_merged.apply(lambda row: h3.latlng_to_cell(row['start_lat'], row['start_lon'], h3_resolution), axis=1)
df_merged['end_h3'] = df_merged.apply(lambda row: h3.latlng_to_cell(row['end_lat'], row['end_lon'], h3_resolution), axis=1)

media_peso = df_merged['WEIGHT_INDIV'].mean()
print(f"📏 Media del peso individuale: {media_peso:.3f}")

od_matrix_first = df_merged.groupby(['start_h3', 'end_h3']).agg({
    'WEIGHT_INDIV': ['sum', 'count']
}).reset_index()

od_matrix_first.columns = ['start_h3', 'end_h3', 'total_weight', 'count']

parent_hexes = ["861fb4667ffffff", "861fb4677ffffff", "861fb466fffffff", "861fb4647ffffff", "861fb475fffffff"]

target_resolution = 10
start_valid_h3 = set()
end_valid_h3 = set()

for parent in parent_hexes:
    children = h3.cell_to_children(parent, target_resolution)
    for child in children:
        start_valid_h3.add(child)
        end_valid_h3.add(child)

mask = (
    (od_matrix_first["start_h3"].isin(start_valid_h3))
    & (od_matrix_first["end_h3"].isin(end_valid_h3))
)

od_matrix_first = od_matrix_first[mask].copy()
print(f"Numero di righe filtrate: {len(od_matrix_first):,}")

od_matrix = od_matrix_first.copy()
od_matrix

# ODkAnon algorithm
def fast_pre_generalization_filter(
    od_matrix: pd.DataFrame,
    k_threshold: int = 10,
    max_generalization_levels: int = 3,
    suppression_budget_percent: float = 0.1
) -> pd.DataFrame:
    """
    Filter OD couples which are not k-anonymous even after generalizing to the maximum level
    """
    print("⚙️ Inizio filtro veloce pre-generalizzazione...")

    original_size = len(od_matrix)
    max_suppressions = int(original_size * suppression_budget_percent)
    print(f"📊 Budget suppression: {max_suppressions} righe su {original_size} ({suppression_budget_percent*100:.1f}%)")

    mapping_cache = {}

    def generalize(h, level_down):
        res = h3.get_resolution(h)
        target_res = res - level_down
        if target_res < 0:
            target_res = 0
        key = (h, target_res)
        if key not in mapping_cache:
            mapping_cache[key] = h3.cell_to_parent(h, target_res) if res > target_res else h
        return mapping_cache[key]

    od_working = od_matrix.copy()
    for lvl in range(max_generalization_levels + 1):
        od_working[f'start_gen_{lvl}'] = od_working['start_h3'].apply(lambda x: generalize(x, lvl))
        od_working[f'end_gen_{lvl}'] = od_working['end_h3'].apply(lambda x: generalize(x, lvl))

    results = []
    for lvl in range(max_generalization_levels + 1):
        grouped = od_working.groupby([f'start_gen_{lvl}', f'end_gen_{lvl}'])['count'].sum().reset_index()
        grouped.columns = ['start_gen', 'end_gen', 'agg_count']
        grouped['level'] = lvl
        results.append(grouped)

    all_levels = pd.concat(results)
    
    od_with_id = od_working.reset_index().rename(columns={'index': 'row_id'})

    valid_pairs = set()
    for lvl in range(max_generalization_levels + 1):
        merged = od_with_id.merge(
            all_levels[all_levels['level'] == lvl],
            left_on=[f'start_gen_{lvl}', f'end_gen_{lvl}'],
            right_on=['start_gen', 'end_gen'],
            how='left'
        )
        valid = merged[merged['agg_count'] >= k_threshold]
        valid_pairs.update(valid['row_id'].tolist())

    all_row_ids = set(od_with_id['row_id'].tolist())
    problematic_rows = all_row_ids - valid_pairs
    
    print(f"🔍 Righe k-anonime con generalizzazione: {len(valid_pairs)}")
    print(f"⚠️ Righe problematiche: {len(problematic_rows)}")

    if len(problematic_rows) <= max_suppressions:
        print(f"✅ Budget sufficiente: soppressione di tutte le {len(problematic_rows)} righe problematiche")
        rows_to_keep = valid_pairs
        suppressed_count = len(problematic_rows)
    else:
        print(f"🎯 Budget insufficiente: sopprimi prima le righe con count più basso")
        
        problematic_df = od_with_id[od_with_id['row_id'].isin(problematic_rows)].copy()
        
        to_suppress = problematic_df.nsmallest(max_suppressions, 'count')['row_id'].tolist()
        
        rows_to_suppress = set(to_suppress)
        rows_to_keep = valid_pairs | (problematic_rows - rows_to_suppress)
        suppressed_count = len(to_suppress)
        
        print(f"📋 Soppresse {suppressed_count} righe")

    filtered = od_with_id[od_with_id['row_id'].isin(rows_to_keep)].copy()
    filtered = filtered[['start_h3', 'end_h3', 'total_weight', 'count']]

    kept_count = len(filtered)
    
    print(f"📈 Risultati finali:")
    print(f"   • Righe mantenute: {kept_count} / {original_size} ({kept_count/original_size*100:.1f}%)")
    print(f"   • Righe soppresse: {suppressed_count} ({suppressed_count/original_size*100:.1f}%)")
    print(f"   • Budget utilizzato: {suppressed_count} / {max_suppressions}")

    return filtered, suppressed_count

od_matrix, suppressed_count = fast_pre_generalization_filter(od_matrix, k_threshold=10, max_generalization_levels=3, suppression_budget_percent=0.1)

filtered_df = df_merged.merge(
    od_matrix[['start_h3', 'end_h3']],
    on=['start_h3', 'end_h3'],
    how='inner'
)
filtered_df

tree_start = create_h3_hierarchical_tree(od_matrix, target_resolution=10, hex_column='start_h3')
tree_end = create_h3_hierarchical_tree(od_matrix, target_resolution=10, hex_column='end_h3')

class OptimizedH3GeneralizedODMatrix:
    """
    Optimized version for very large OD matrices with k-anonymity based on counts
    Main matrix: count (for k-anonymity)
    Secondary matrix: weights (kept for analysis)
    """
    
    def __init__(self, od_matrix: pd.DataFrame, tree_start, tree_end, k_threshold: int = 10):
        self.original_od_matrix = od_matrix
        self.tree_start = tree_start
        self.tree_end = tree_end
        self.k_threshold = k_threshold
        
        self.current_matrix_sparse = None
        self.current_weights_sparse = None
        
        self.start_to_idx = {}  
        self.end_to_idx = {}    
        self.idx_to_start = {}  
        self.idx_to_end = {}    
        
        self.sibling_groups_cache = {}
        self.parent_cache = {}
        
        self.generalization_history = []
        
    def initialize_optimized_matrix(self):
        """Initialize using sparse matrices and smart pre-processing"""
        print("🔧 Optimized initialization...")
        
        non_zero_od = self.original_od_matrix[self.original_od_matrix['count'] > 0].copy()
        print(f"📊 Filtered data: {len(non_zero_od):,} non-zero cells out of {len(self.original_od_matrix):,}")
        
        used_starts = set(non_zero_od['start_h3'].unique())
        used_ends = set(non_zero_od['end_h3'].unique())
        
        target_starts = self._get_target_resolution_hexagons(used_starts, self.tree_start)
        target_ends = self._get_target_resolution_hexagons(used_ends, self.tree_end)
        
        print(f"🎯 Target hexagons: {len(target_starts)} origins, {len(target_ends)} destinations")
        
        self.start_to_idx = {h3_id: idx for idx, h3_id in enumerate(sorted(target_starts))}
        self.end_to_idx = {h3_id: idx for idx, h3_id in enumerate(sorted(target_ends))}
        self.idx_to_start = {idx: h3_id for h3_id, idx in self.start_to_idx.items()}
        self.idx_to_end = {idx: h3_id for h3_id, idx in self.end_to_idx.items()}
        
        self.current_matrix_sparse, self.current_weights_sparse = self._build_sparse_matrices(
            non_zero_od, target_starts, target_ends
        )
        
        self._precompute_sibling_groups()
        
        print(f"✅ Sparse matrices initialized: {self.current_matrix_sparse.shape} "
              f"({self.current_matrix_sparse.nnz:,} non-zero elements)")
        
        return self
    
    def _get_target_resolution_hexagons(self, hexagons: Set[str], tree) -> Set[str]:
        """Get target resolution hexagons only for those used"""
        target_hexagons = set()
        target_res = tree.target_resolution
        
        for hex_id in hexagons:
            current_res = h3.get_resolution(hex_id)
            
            if current_res == target_res:
                target_hexagons.add(hex_id)
            elif current_res < target_res:
                if hex_id in tree.nodes:
                    children = self._get_children_at_resolution_fast(hex_id, target_res)
                    target_hexagons.update(children)
            else:
                parent = h3.cell_to_parent(hex_id, target_res)
                target_hexagons.add(parent)
        
        return target_hexagons
    
    def _get_children_at_resolution_fast(self, hex_id: str, target_res: int) -> Set[str]:
        """Fast version to get children"""
        current_res = h3.get_resolution(hex_id)
        if current_res == target_res:
            return {hex_id}
        
        cache_key = (hex_id, target_res)
        if cache_key in self.sibling_groups_cache:
            return self.sibling_groups_cache[cache_key]
        
        children = set()
        queue = [hex_id]
        
        while queue:
            current = queue.pop(0)
            current_r = h3.get_resolution(current)
            
            if current_r == target_res:
                children.add(current)
            elif current_r < target_res:
                direct_children = h3.cell_to_children(current, current_r + 1)
                queue.extend(direct_children)
        
        self.sibling_groups_cache[cache_key] = children
        return children
    
    def _build_sparse_matrices(self, od_data: pd.DataFrame, target_starts: Set[str], target_ends: Set[str]) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
        """Builds separate sparse matrices for counts (main) and weights (secondary)"""
        rows_counts, cols_counts, data_counts = [], [], []
        rows_weights, cols_weights, data_weights = [], [], []
        
        print("🔨 Building sparse matrices...")
        
        grouped = od_data.groupby(['start_h3', 'end_h3']).agg({
            'count': 'sum',
            'total_weight': 'sum'
        }).reset_index()
        
        for _, row in grouped.iterrows():
            start_h3 = row['start_h3']
            end_h3 = row['end_h3']
            count = row['count']
            weight = row['total_weight']
            
            mapped_start = self._map_to_target_fast(start_h3, target_starts, self.tree_start)
            mapped_end = self._map_to_target_fast(end_h3, target_ends, self.tree_end)
            
            if mapped_start and mapped_end:
                start_idx = self.start_to_idx[mapped_start]
                end_idx = self.end_to_idx[mapped_end]
                
                rows_counts.append(end_idx)
                cols_counts.append(start_idx)
                data_counts.append(count)
                
                rows_weights.append(end_idx)
                cols_weights.append(start_idx)
                data_weights.append(weight)
        
        shape = (len(target_ends), len(target_starts))
        
        matrix_counts_coo = sp.coo_matrix((data_counts, (rows_counts, cols_counts)), shape=shape)
        matrix_counts_csr = matrix_counts_coo.tocsr()
        matrix_counts_csr.sum_duplicates()
        
        matrix_weights_coo = sp.coo_matrix((data_weights, (rows_weights, cols_weights)), shape=shape)
        matrix_weights_csr = matrix_weights_coo.tocsr()
        matrix_weights_csr.sum_duplicates()
        
        return matrix_counts_csr, matrix_weights_csr
    
    def _map_to_target_fast(self, h3_id: str, target_nodes: Set[str], tree) -> Optional[str]:
        """Fast version of mapping with cache"""
        if h3_id in target_nodes:
            return h3_id
        
        if h3_id in self.parent_cache:
            cached_parent = self.parent_cache[h3_id]
            if cached_parent in target_nodes:
                return cached_parent
        
        current_res = h3.get_resolution(h3_id)
        target_res = tree.target_resolution
        
        if current_res > target_res:
            parent = h3.cell_to_parent(h3_id, target_res)
            self.parent_cache[h3_id] = parent
            return parent if parent in target_nodes else None
        elif current_res < target_res:
            for target in target_nodes:
                if self._is_descendant_fast(target, h3_id):
                    return target
        
        return None
    
    def _is_descendant_fast(self, child_h3: str, parent_h3: str) -> bool:
        """Optimized descendant check"""
        child_res = h3.get_resolution(child_h3)
        parent_res = h3.get_resolution(parent_h3)
        
        if parent_res >= child_res:
            return False
        
        cache_key = (child_h3, parent_h3)
        if cache_key in self.parent_cache:
            return self.parent_cache[cache_key]
        
        current = child_h3
        while h3.get_resolution(current) > parent_res:
            current = h3.cell_to_parent(current, h3.get_resolution(current) - 1)
        
        result = current == parent_h3
        self.parent_cache[cache_key] = result
        return result
    
    def _precompute_sibling_groups(self):
        """Pre-compute all possible sibling groups"""
        print("🧠 Pre-computing sibling groups...")
        
        self.start_sibling_groups = self._compute_sibling_groups(self.start_to_idx.keys(), self.tree_start)
        self.end_sibling_groups = self._compute_sibling_groups(self.end_to_idx.keys(), self.tree_end)
        
        print(f"📋 Found {len(self.start_sibling_groups)} origin groups, {len(self.end_sibling_groups)} destination groups")
    
    def _compute_sibling_groups(self, nodes: Set[str], tree) -> List[Tuple[List[str], str]]:
        """Compute all sibling groups once - INCLUDING single children"""
        groups = []
        processed = set()
        
        for node_id in nodes:
            if node_id in processed or node_id not in tree.nodes:
                continue
            
            node = tree.nodes[node_id]
            if not node.parent:
                continue
            
            siblings = []
            for sibling_id in node.parent.children:
                if sibling_id in nodes and sibling_id not in processed:
                    siblings.append(sibling_id)
            
            if len(siblings) >= 1:
                groups.append((siblings, node.parent.h3_id))
                processed.update(siblings)
        
        return groups
    
    def get_best_generalization_fast(self, axis: str) -> Optional[Tuple[List[str], str, int]]:
        """Find the best generalization based on COUNT (k-anonymity)"""
        if axis == 'columns':
            tree = self.tree_start
            mapping = self.start_to_idx
            matrix = self.current_matrix_sparse.tocsc()
        else:
            tree = self.tree_end
            mapping = self.end_to_idx
            matrix = self.current_matrix_sparse.tocsr()

        best_group, best_parent, best_cost = None, None, float('inf')
        
        for parent_id, parent_node in tree.nodes.items():
            siblings = list(parent_node.children.keys())
            
            if len(siblings) < 1:
                continue
                
            present = [s for s in siblings if s in mapping]
            
            if len(present) == 0:
                continue
            
            if len(siblings) > 1 and len(present) != len(siblings):
               continue

            cost = sum(tree.nodes[sibling_id].count for sibling_id in present)
            
            if cost < best_cost:
                best_cost = cost
                best_group = present
                best_parent = parent_id

        if best_group is None:
            return None
        return best_group, best_parent, best_cost
    
    def apply_sparse_generalization(self, group: List[str], parent_id: str, axis: str):
        """Apply generalization on both matrices - works also with single-element groups"""
        if axis == 'columns':
            indices = [self.start_to_idx[h3_id] for h3_id in group]

            counts_csc = self.current_matrix_sparse.tocsc()
            combined_counts_col = counts_csc[:, indices].sum(axis=1).A1
            
            weights_csc = self.current_weights_sparse.tocsc()
            combined_weights_col = weights_csc[:, indices].sum(axis=1).A1
            
            mask = np.ones(counts_csc.shape[1], dtype=bool)
            mask[indices] = False
            counts_reduced = counts_csc[:, mask]
            weights_reduced = weights_csc[:, mask]

            combined_counts_sparse = sp.csr_matrix(combined_counts_col).T
            combined_weights_sparse = sp.csr_matrix(combined_weights_col).T
            
            self.current_matrix_sparse = sp.hstack([counts_reduced, combined_counts_sparse]).tocsr()
            self.current_weights_sparse = sp.hstack([weights_reduced, combined_weights_sparse]).tocsr()

            new_start_to_idx = {}
            new_idx_to_start = {}
            new_idx = 0

            for old_idx, h3_id in self.idx_to_start.items():
                if h3_id not in group:
                    new_start_to_idx[h3_id] = new_idx
                    new_idx_to_start[new_idx] = h3_id
                    new_idx += 1

            new_start_to_idx[parent_id] = new_idx
            new_idx_to_start[new_idx] = parent_id

            self.start_to_idx = new_start_to_idx
            self.idx_to_start = new_idx_to_start

        else:
            indices = [self.end_to_idx[h3_id] for h3_id in group]

            counts_csr = self.current_matrix_sparse.tocsr()
            combined_counts_row = counts_csr[indices, :].sum(axis=0).A1

            weights_csr = self.current_weights_sparse.tocsr()
            combined_weights_row = weights_csr[indices, :].sum(axis=0).A1

            mask = np.ones(counts_csr.shape[0], dtype=bool)
            mask[indices] = False
            counts_reduced = counts_csr[mask, :]
            weights_reduced = weights_csr[mask, :]

            combined_counts_sparse = sp.csr_matrix(combined_counts_row)
            combined_weights_sparse = sp.csr_matrix(combined_weights_row)
            
            self.current_matrix_sparse = sp.vstack([counts_reduced, combined_counts_sparse]).tocsr()
            self.current_weights_sparse = sp.vstack([weights_reduced, combined_weights_sparse]).tocsr()

            new_end_to_idx = {}
            new_idx_to_end = {}
            new_idx = 0

            for old_idx, h3_id in self.idx_to_end.items():
                if h3_id not in group:
                    new_end_to_idx[h3_id] = new_idx
                    new_idx_to_end[new_idx] = h3_id
                    new_idx += 1

            new_end_to_idx[parent_id] = new_idx
            new_idx_to_end[new_idx] = parent_id

            self.end_to_idx = new_end_to_idx
            self.idx_to_end = new_idx_to_end

        if axis == 'columns':
            self.start_sibling_groups = [
                (sibs, par) for sibs, par in self.start_sibling_groups 
                if not any(s in group for s in sibs)
            ]
        else:
            self.end_sibling_groups = [
                (sibs, par) for sibs, par in self.end_sibling_groups 
                if not any(s in group for s in sibs)
            ]

        if axis == 'columns':
            parent_node = self.tree_start.nodes.get(parent_id)
            if parent_node and parent_node.parent:
                siblings_left = [s for s in parent_node.parent.children.keys() 
                            if s in self.start_to_idx and s != parent_id]
                if len(siblings_left) >= 1:
                    grandparent_id = parent_node.parent.h3_id
                    self.start_sibling_groups.append((siblings_left + [parent_id], grandparent_id))
        else:
            parent_node = self.tree_end.nodes.get(parent_id)
            if parent_node and parent_node.parent:
                siblings_left = [s for s in parent_node.parent.children.keys() 
                            if s in self.end_to_idx and s != parent_id]
                if len(siblings_left) >= 1:
                    grandparent_id = parent_node.parent.h3_id
                    self.end_sibling_groups.append((siblings_left + [parent_id], grandparent_id))
    
    def get_min_value_sparse(self) -> int:
        """Get minimum value from sparse COUNT matrix (for k-anonymity)"""
        if self.current_matrix_sparse.nnz == 0:
            return 0
        return self.current_matrix_sparse.data.min()
    
    def run_optimized_generalization(self) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
        """Run optimized generalization based on COUNT and return both matrices"""
        print(f"🚀 Starting optimized generalization based on COUNT (k={self.k_threshold})")
        print("=" * 60)
        
        start_time = time.time()
        self.initialize_optimized_matrix()
        
        step_count = 0
        self.step_rates = []
        
        while True:
            min_value = self.get_min_value_sparse()
            current_shape = self.current_matrix_sparse.shape
            
            print(f"🔍 Step {step_count+1}: Min Count={min_value:,}, Shape={current_shape[0]}×{current_shape[1]}, NonZero={self.current_matrix_sparse.nnz:,}")
        
            min_after_generalization = self.current_matrix_sparse.data.min() if self.current_matrix_sparse.nnz > 0 else float('inf')
            if min_after_generalization >= self.k_threshold:
                print(f"✅ All counts >= k ({self.k_threshold}), generalization completed.")
                break
            
            if step_count == 0:
                initial_cols = self.current_matrix_sparse.shape[1]
                initial_rows = self.current_matrix_sparse.shape[0]
                self.initial_ratio = initial_cols / initial_rows if initial_rows > 0 else 1.0
                self.tolerance = 0.03  

            current_cols = self.current_matrix_sparse.shape[1]
            current_rows = self.current_matrix_sparse.shape[0]
            current_ratio = current_cols / current_rows if current_rows > 0 else 1.0

            deviation = (current_ratio - self.initial_ratio) / self.initial_ratio

            if deviation > self.tolerance:
                axis = 'columns'
            elif deviation < -self.tolerance:
                axis = 'rows'
            else:
                axis = 'columns' if step_count % 2 == 0 else 'rows'
            
            best_gen = self.get_best_generalization_fast(axis)
            if not best_gen:
                fallback_axis = 'rows' if axis == 'columns' else 'columns'
                best_gen = self.get_best_generalization_fast(fallback_axis)
                if best_gen:
                    axis = fallback_axis
                    print(f"🔄 No generalization for original direction, fallback to: {fallback_axis}")
                else:
                    print(f"⚠️ No generalization possible on {axis} nor on alternative direction ({fallback_axis}).")
                    if min_after_generalization < self.k_threshold:
                        print(f"❌ There are still counts < k ({self.k_threshold}), but no further generalization is possible.")
                    break

            group, parent_id, cost = best_gen
            
            self.apply_sparse_generalization(group, parent_id, axis)

            if current_shape[0] <= 50 or current_shape[1] <= 50:
                print("🧾 Current count matrix (non-zero integer values):")
                print(self.current_matrix_sparse.toarray().astype(int))
                print("🧾 Current weight matrix (non-zero integer values):")
                print(self.current_weights_sparse.toarray().astype(int))
            
            self.generalization_history.append({
                'step': step_count + 1,
                'axis': axis,
                'group_size': len(group),
                'parent': parent_id,
                'cost': cost,
                'min_before': min_value,
                'shape_after': self.current_matrix_sparse.shape
            })
            
            elapsed = time.time() - start_time
            step_rate

            step_rate = step_count / elapsed if elapsed > 0 else 0
            self.step_rates.append(step_rate)
            step_count += 1
            
            if step_count % 10 == 0:
                print(f"⏱️ Elapsed: {elapsed:.1f}s, Step rate: {step_rate:.2f} steps/sec")
        
        total_time = time.time() - start_time
        print(f"\n🎯 Generalization completed in {step_count} steps ({total_time:.2f}s)")
        print(f"📐 Final matrix: {self.current_matrix_sparse.shape}")
        print(f"📊 Threshold to respect: {self.k_threshold}")
        print(f"📊 Final minimum value (count): {self.current_matrix_sparse.data.min() if self.current_matrix_sparse.nnz > 0 else 'N/A'}")
        print(f"📊 Final maximum value (count): {self.current_matrix_sparse.data.max() if self.current_matrix_sparse.nnz > 0 else 'N/A'}")
        print(f"📊 Final minimum value (weight): {self.current_weights_sparse.data.min() if self.current_weights_sparse.nnz > 0 else 'N/A'}")
        print(f"📊 Final maximum value (weight): {self.current_weights_sparse.data.max() if self.current_weights_sparse.nnz > 0 else 'N/A'}")
        print(f"💾 Non-zero elements: {self.current_matrix_sparse.nnz:,}")
        
        return self.current_matrix_sparse, self.current_weights_sparse

def run_optimized_generalization(od_matrix_df: pd.DataFrame, tree_start, tree_end, k_threshold: int = 10):
    """
    Generalization algorithm
    
    Returns:
        Tuple[sp.csr_matrix, sp.csr_matrix, OptimizedH3GeneralizedODMatrix]: 
        (matrice_count, matrice_pesi, generalizer)
    """
    generalizer = OptimizedH3GeneralizedODMatrix(od_matrix_df, tree_start, tree_end, k_threshold)
    count_result, weights_result = generalizer.run_optimized_generalization()
    
    return count_result, weights_result, generalizer

sparse_result, weights_result, generalizer = run_optimized_generalization(od_matrix, tree_start, tree_end, k_threshold=10)


# Results and visualization

k_count = 10
k_weight = 10 * media_peso
analyzer = CountAnalyzer(sparse_result, weights_result, generalizer, k_count, k_weight)
count_stats, weight_stats = analyzer.print_summary_report()

plot_count_distributions(weights_result, k_count=10*media_peso)

visualizer = H3FoliumVisualizerODkAnon(generalizer)
mappa = visualizer.create_base_map(zoom_start=11)
visualizer.add_origin_hexagons(mappa, max_hexagons=14000)
visualizer.add_destination_hexagons(mappa, max_hexagons=14000)
folium.LayerControl(collapsed=False).add_to(mappa)
mappa

metrics = compute_discernability_and_cavg_sparse_ODkAnon(sparse_result, od_matrix, suppressed_count=suppressed_count, k=10)
print("\n📊 Metrics di Discernibilità e CAVG:")
print(f"C_DM: {metrics['C_DM']:,}")
print(f"C_AVG: {metrics['C_AVG']:.4f}")

metric = GeneralizationMetricODkAnon(k_threshold=10)
error = metric.calculate_generalization_error(sparse_result, generalizer)
print(f"Errore di generalizzazione medio Ḡ: {error:.3f}")

loss = fast_reconstruction_loss_ODkAnon(original_od_df=od_matrix_first, generalized_matrix=sparse_result, generalizer=generalizer, tree_start=tree_start, tree_end=tree_end)
print(f"Reconstruction Loss: {loss:.6f}")

metrics = compute_discernability_and_cavg_sparse_ODkAnon(weights_result, od_matrix, suppressed_count=suppressed_count, k=10*media_peso)
print("\n📊 Metrics di Discernibilità e CAVG:")
print(f"C_DM: {metrics['C_DM']:,}")
print(f"C_AVG: {metrics['C_AVG']:.4f}")

metric = GeneralizationMetricODkAnon(k_threshold=10*media_peso)
error = metric.calculate_generalization_error(weights_result, generalizer)
print(f"Errore di generalizzazione medio Ḡ: {error:.3f}")

loss = fast_reconstruction_loss_weight(original_od_df=od_matrix_first, generalized_matrix=weights_result, generalizer=generalizer, tree_start=tree_start, tree_end=tree_end)
print(f"Reconstruction Loss: {loss:.6f}")