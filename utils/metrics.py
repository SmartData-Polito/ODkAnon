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
from geopy.distance import geodesic



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
    

def compute_discernability_and_cavg_ATG(df: pd.DataFrame, k: int, suppressed_count: int = 0) -> dict:
    """
    Calcola C_DM e C_AVG per un dataset OD generalizzato.
    
    Args:
        df: DataFrame con colonne ['start_h3', 'end_h3', 'count']
        k: soglia k-anonimity
        suppressed_count: numero di coppie OD soppressi (facoltativo)
    
    Returns:
        dict con C_DM, C_AVG, numero totale record e classi equivalenza
    """
    counts = df['count'].values
    total_records = counts.sum() + suppressed_count
    total_equiv_classes = len(counts) + suppressed_count
    total_records_avg = counts.sum()
    total_equiv_classes_avg = len(counts)
    
    # C_DM: somma dei quadrati dei count >= k
    k_anonymous_counts = counts[counts >= k]
    c_dm_gen = np.sum(k_anonymous_counts**2)
    
    # Penalità per record soppressi
    suppression_penalty = suppressed_count * counts.sum() # ogni record soppresso "costa" quanto l'intero dataset
    c_dm = c_dm_gen + suppression_penalty
    
    # C_AVG: (total_records / total_equiv_classes) / k
    c_avg = (total_records_avg / total_equiv_classes_avg) / k if total_equiv_classes_avg > 0 else float('inf')
    
    return {
        'C_DM': c_dm,
        'C_AVG': c_avg,
        'total_records': total_records,
        'total_equivalence_classes': total_equiv_classes,
        'k': k
    }

# Esempio di utilizzo
metrics = compute_discernability_and_cavg(od_matrix_agg, k=10, suppressed_count=suppressed_count)
print("\n📊 Metrics di Discernibilità e CAVG:")
print(f"C_DM: {metrics['C_DM']:,}")
print(f"C_AVG: {metrics['C_AVG']:.4f}")



def calculate_generalization_distance_metric_ATG(df: pd.DataFrame, od_matrix_generalized: pd.DataFrame) -> Dict:
   
   print("🔍 Calcolo metrica di distanza post-generalizzazione...")
   
   # 1. Crea mapping da esagoni originali a esagoni generalizzati
   start_original_to_generalized = {}
   end_original_to_generalized = {}
   
   # Ottieni tutti gli esagoni generalizzati unici
   generalized_start_h3 = set(od_matrix_generalized['start_h3'].unique())
   generalized_end_h3 = set(od_matrix_generalized['end_h3'].unique())
   
   # Per ogni esagono originale, trova l'esagono generalizzato corrispondente
   unique_start_h3 = df['start_h3'].unique()
   unique_end_h3 = df['end_h3'].unique()
   
   print(f"📊 Mappatura {len(unique_start_h3)} esagoni origine...")
   for original_h3 in unique_start_h3:
       generalized_h3 = find_generalized_hexagon(original_h3, generalized_start_h3)
       if generalized_h3:
           start_original_to_generalized[original_h3] = generalized_h3
   
   print(f"📊 Mappatura {len(unique_end_h3)} esagoni destinazione...")
   for original_h3 in unique_end_h3:
       generalized_h3 = find_generalized_hexagon(original_h3, generalized_end_h3)
       if generalized_h3:
           end_original_to_generalized[original_h3] = generalized_h3
   
   # 3. Calcola distanze per i punti di partenza
   start_distances = []
   start_coords = []
   
   for idx, row in df.iterrows():
       original_h3 = row['start_h3']
       original_coords = (row['start_lat'], row['start_lon'])
       
       if original_h3 in start_original_to_generalized:
           generalized_h3 = start_original_to_generalized[original_h3]
           generalized_coords = h3.cell_to_latlng(generalized_h3)
           
           distance = geodesic(original_coords, generalized_coords).meters
           
           start_distances.append(distance)
           start_coords.append({
               'original_h3': original_h3,
               'generalized_h3': generalized_h3,
               'original_coords': original_coords,
               'generalized_coords': generalized_coords,
               'distance': distance
           })
   
   # 4. Calcola distanze per i punti di destinazione
   end_distances = []
   end_coords = []
   
   for idx, row in df.iterrows():
       original_h3 = row['end_h3']
       original_coords = (row['end_lat'], row['end_lon'])
       
       if original_h3 in end_original_to_generalized:
           generalized_h3 = end_original_to_generalized[original_h3]
           generalized_coords = h3.cell_to_latlng(generalized_h3)
           
           distance = geodesic(original_coords, generalized_coords).meters
               
           end_distances.append(distance)
           end_coords.append({
               'original_h3': original_h3,
               'generalized_h3': generalized_h3,
               'original_coords': original_coords,
               'generalized_coords': generalized_coords,
               'distance': distance
           })
   
   # 5. Calcola statistiche
   results = {
       'start_distances': {
           'mean': np.mean(start_distances) if start_distances else 0,
           'median': np.median(start_distances) if start_distances else 0,
           'std': np.std(start_distances) if start_distances else 0,
           'min': np.min(start_distances) if start_distances else 0,
           'max': np.max(start_distances) if start_distances else 0,
           'count': len(start_distances)
       },
       'end_distances': {
           'mean': np.mean(end_distances) if end_distances else 0,
           'median': np.median(end_distances) if end_distances else 0,
           'std': np.std(end_distances) if end_distances else 0,
           'min': np.min(end_distances) if end_distances else 0,
           'max': np.max(end_distances) if end_distances else 0,
           'count': len(end_distances)
       },
       'overall': {
           'mean': np.mean(start_distances + end_distances) if (start_distances or end_distances) else 0,
           'median': np.median(start_distances + end_distances) if (start_distances or end_distances) else 0,
           'std': np.std(start_distances + end_distances) if (start_distances or end_distances) else 0,
           'total_points': len(start_distances) + len(end_distances)
       },
       'mappings': {
           'start_original_to_generalized': start_original_to_generalized,
           'end_original_to_generalized': end_original_to_generalized
       },
       'detailed_coords': {
           'start': start_coords,
           'end': end_coords
       }
   }
   
   # 6. Stampa risultati
   print("\n" + "="*60)
   print("📏 METRICHE DI DISTANZA POST-GENERALIZZAZIONE")
   print("="*60)
   
   print(f"\n🎯 PUNTI DI PARTENZA:")
   print(f"   • Distanza media: {results['start_distances']['mean']:.2f} metri")
   print(f"   • Distanza mediana: {results['start_distances']['median']:.2f} metri")
   print(f"   • Deviazione standard: {results['start_distances']['std']:.2f} metri")
   print(f"   • Min-Max: {results['start_distances']['min']:.2f} - {results['start_distances']['max']:.2f} metri")
   print(f"   • Punti analizzati: {results['start_distances']['count']:,}")
   
   print(f"\n🏁 PUNTI DI DESTINAZIONE:")
   print(f"   • Distanza media: {results['end_distances']['mean']:.2f} metri")
   print(f"   • Distanza mediana: {results['end_distances']['median']:.2f} metri")
   print(f"   • Deviazione standard: {results['end_distances']['std']:.2f} metri")
   print(f"   • Min-Max: {results['end_distances']['min']:.2f} - {results['end_distances']['max']:.2f} metri")
   print(f"   • Punti analizzati: {results['end_distances']['count']:,}")
   
   print(f"\n🌍 COMPLESSIVO:")
   print(f"   • Distanza media totale: {results['overall']['mean']:.2f} metri")
   print(f"   • Distanza mediana totale: {results['overall']['median']:.2f} metri")
   print(f"   • Deviazione standard totale: {results['overall']['std']:.2f} metri")
   print(f"   • Punti totali: {results['overall']['total_points']:,}")
   
   return results

def find_generalized_hexagon(original_h3: str, generalized_hexagons: set) -> str:
   """
   Trova l'esagono generalizzato corrispondente a un esagono originale
   """
   # Se l'esagono è già nella lista degli esagoni generalizzati
   if original_h3 in generalized_hexagons:
       return original_h3
   
   # Altrimenti cerca tra tutti gli esagoni generalizzati se l'originale è loro discendente
   for generalized_h3 in generalized_hexagons:
       if is_descendant_of(original_h3, generalized_h3):
           return generalized_h3
   
   return None

def is_descendant_of(child_h3: str, parent_h3: str) -> bool:
   """
   Controlla se child_h3 è discendente di parent_h3
   """
   child_res = h3.get_resolution(child_h3)
   parent_res = h3.get_resolution(parent_h3)
   
   if parent_res >= child_res:
       return False
   
   current = child_h3
   while h3.get_resolution(current) > parent_res:
       current = h3.cell_to_parent(current, h3.get_resolution(current) - 1)
   
   return current == parent_h3


class GeneralizationMetricATG:
    """
    Ḡ = (1/V+) × Σ(|o| + |d|) × v_{o→d}
    """
    def __init__(self, k_threshold: int = 10):
        self.k_threshold = k_threshold

    def calculate_generalization_error(self, od_matrix_generalized: pd.DataFrame, od_matrix: pd.DataFrame) -> float:
        # Costruisci dizionari: generalizzato -> numero di celle originali
        origin_counts = self._build_hexagon_counts(
            od_matrix_generalized, od_matrix, column_gen="start_h3", column_orig="start_h3"
        )
        destination_counts = self._build_hexagon_counts(
            od_matrix_generalized, od_matrix, column_gen="end_h3", column_orig="end_h3"
        )

        total_volume_anonymous = 0
        weighted_count_sum = 0

        for _, row in od_matrix_generalized.iterrows():
            flow_value = row["count"]
            if flow_value >= self.k_threshold:
                origin_h3 = row["start_h3"]
                dest_h3   = row["end_h3"]

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
        Conta quanti esagoni originali appartengono a ciascun esagono generalizzato
        """
        generalized_hexagons = od_matrix_generalized[column_gen].unique()
        original_hexagons = od_matrix[column_orig].unique()

        counts = {}
        for gen_hex in generalized_hexagons:
            target_res = h3.get_resolution(gen_hex)

            # Trova tutti i parent degli originali alla risoluzione target
            parent_series = [h3.cell_to_parent(h, target_res) for h in original_hexagons]

            # Conta quante volte compare il parent == gen_hex
            count = sum(1 for p in parent_series if p == gen_hex)
            counts[gen_hex] = max(count, 1)  # fallback a 1

        return counts
    

def fast_reconstruction_loss_ATG(original_od_df: pd.DataFrame,
                                       od_matrix_generalized: pd.DataFrame) -> float:
    """
    Calcola la reconstruction loss includendo anche le celle con 0 viaggi.
    Versione ottimizzata, evita itertools.product su tutte le foglie.
    """
    # Dizionario dei flussi originali
    original_flows = {(row['start_h3'], row['end_h3']): row['count']
                      for _, row in original_od_df.iterrows()}

    total_volume = sum(original_flows.values())
    if total_volume == 0:
        return 0.0

    # Dizionario dei flussi generalizzati
    generalized_flows = {(row['start_h3'], row['end_h3']): row['count']
                         for _, row in od_matrix_generalized.iterrows()}

    gen_start_hexes = od_matrix_generalized['start_h3'].unique()
    gen_end_hexes   = od_matrix_generalized['end_h3'].unique()

    # Cache: gen_hex → leaves
    leaf_cache = {}

    def get_leaves(gen_hex, target_res):
        key = (gen_hex, target_res)
        if key in leaf_cache:
            return leaf_cache[key]
        res = h3.get_resolution(gen_hex)
        leaves = {gen_hex} if res == target_res else set(h3.cell_to_children(gen_hex, target_res))
        leaf_cache[key] = leaves
        return leaves

    # Risoluzione target
    target_res_start = h3.get_resolution(original_od_df['start_h3'].iloc[0])
    target_res_end   = h3.get_resolution(original_od_df['end_h3'].iloc[0])

    # Precompute mappe: leaf → parent generalized
    start_leaf_to_parent = {}
    for gen in gen_start_hexes:
        for leaf in get_leaves(gen, target_res_start):
            start_leaf_to_parent[leaf] = gen

    end_leaf_to_parent = {}
    for gen in gen_end_hexes:
        for leaf in get_leaves(gen, target_res_end):
            end_leaf_to_parent[leaf] = gen

    total_abs_error = 0.0

    # Itera solo sulle coppie presenti nell'originale
    for (s, d), true_count in original_flows.items():
        gen_s = start_leaf_to_parent.get(s)
        gen_d = end_leaf_to_parent.get(d)

        if gen_s is None or gen_d is None:
            reconstructed_count = 0.0
        else:
            gen_count = generalized_flows.get((gen_s, gen_d), 0.0)
            start_leaves = get_leaves(gen_s, target_res_start)
            end_leaves   = get_leaves(gen_d, target_res_end)
            reconstructed_count = gen_count / (len(start_leaves) * len(end_leaves))

        total_abs_error += abs(reconstructed_count - true_count)

    # Aggiungi errore per le coppie con count=0 nei generalized flows
    for (gen_s, gen_d), gen_count in generalized_flows.items():
        start_leaves = get_leaves(gen_s, target_res_start)
        end_leaves   = get_leaves(gen_d, target_res_end)
        count_per_leaf = gen_count / (len(start_leaves) * len(end_leaves))

        # Sottrai le coppie già contate
        for s in start_leaves:
            for d in end_leaves:
                if (s, d) not in original_flows:
                    total_abs_error += abs(count_per_leaf - 0.0)

    return total_abs_error / total_volume


def compute_discernability_and_cavg_weight_ATG(df: pd.DataFrame, k: int, suppressed_count: int = 0) -> dict:
    """
    Calcola C_DM e C_AVG per un dataset OD generalizzato.
    
    Args:
        df: DataFrame con colonne ['start_h3', 'end_h3', 'count']
        k: soglia k-anonimity
        suppressed_count: numero di coppie OD soppressi (facoltativo)
    
    Returns:
        dict con C_DM, C_AVG, numero totale record e classi equivalenza
    """
    counts = df['weight'].values
    total_records = counts.sum() + suppressed_count
    total_equiv_classes = len(counts) + suppressed_count
    total_records_avg = counts.sum()
    total_equiv_classes_avg = len(counts)
    
    # C_DM: somma dei quadrati dei count >= k
    k_anonymous_counts = counts[counts >= k]
    c_dm_gen = np.sum(k_anonymous_counts**2)
    
    # Penalità per record soppressi
    suppression_penalty = suppressed_count * counts.sum() # ogni record soppresso "costa" quanto l'intero dataset
    c_dm = c_dm_gen + suppression_penalty
    
    # C_AVG: (total_records / total_equiv_classes) / k
    c_avg = (total_records_avg / total_equiv_classes_avg) / k if total_equiv_classes_avg > 0 else float('inf')
    
    return {
        'C_DM': c_dm,
        'C_AVG': c_avg,
        'total_records': total_records,
        'total_equivalence_classes': total_equiv_classes,
        'k': k
    }


class GeneralizationMetricWeightATG:
    """
    Ḡ = (1/V+) × Σ(|o| + |d|) × v_{o→d}
    """
    def __init__(self, k_threshold: int = 10):
        self.k_threshold = k_threshold

    def calculate_generalization_error(self, od_matrix_generalized: pd.DataFrame, od_matrix: pd.DataFrame) -> float:
        # Costruisci dizionari: generalizzato -> numero di celle originali
        origin_counts = self._build_hexagon_counts(
            od_matrix_generalized, od_matrix, column_gen="start_h3", column_orig="start_h3"
        )
        destination_counts = self._build_hexagon_counts(
            od_matrix_generalized, od_matrix, column_gen="end_h3", column_orig="end_h3"
        )

        total_volume_anonymous = 0
        weighted_count_sum = 0

        for _, row in od_matrix_generalized.iterrows():
            flow_value = row["weight"]
            if flow_value >= self.k_threshold:
                origin_h3 = row["start_h3"]
                dest_h3   = row["end_h3"]

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
        Conta quanti esagoni originali appartengono a ciascun esagono generalizzato
        """
        generalized_hexagons = od_matrix_generalized[column_gen].unique()
        original_hexagons = od_matrix[column_orig].unique()

        counts = {}
        for gen_hex in generalized_hexagons:
            target_res = h3.get_resolution(gen_hex)

            # Trova tutti i parent degli originali alla risoluzione target
            parent_series = [h3.cell_to_parent(h, target_res) for h in original_hexagons]

            # Conta quante volte compare il parent == gen_hex
            count = sum(1 for p in parent_series if p == gen_hex)
            counts[gen_hex] = max(count, 1)  # fallback a 1

        return counts



def fast_reconstruction_loss_weight_ATG(original_od_df: pd.DataFrame,
                                       od_matrix_generalized: pd.DataFrame) -> float:
    """
    Calcola la reconstruction loss includendo anche le celle con 0 viaggi.
    Versione ottimizzata, evita itertools.product su tutte le foglie.
    """
    # Dizionario dei flussi originali
    original_flows = {(row['start_h3'], row['end_h3']): row['total_weight']
                      for _, row in original_od_df.iterrows()}

    total_volume = sum(original_flows.values())
    if total_volume == 0:
        return 0.0

    # Dizionario dei flussi generalizzati
    generalized_flows = {(row['start_h3'], row['end_h3']): row['weight']
                         for _, row in od_matrix_generalized.iterrows()}

    gen_start_hexes = od_matrix_generalized['start_h3'].unique()
    gen_end_hexes   = od_matrix_generalized['end_h3'].unique()

    # Cache: gen_hex → leaves
    leaf_cache = {}

    def get_leaves(gen_hex, target_res):
        key = (gen_hex, target_res)
        if key in leaf_cache:
            return leaf_cache[key]
        res = h3.get_resolution(gen_hex)
        leaves = {gen_hex} if res == target_res else set(h3.cell_to_children(gen_hex, target_res))
        leaf_cache[key] = leaves
        return leaves

    # Risoluzione target
    target_res_start = h3.get_resolution(original_od_df['start_h3'].iloc[0])
    target_res_end   = h3.get_resolution(original_od_df['end_h3'].iloc[0])

    # Precompute mappe: leaf → parent generalized
    start_leaf_to_parent = {}
    for gen in gen_start_hexes:
        for leaf in get_leaves(gen, target_res_start):
            start_leaf_to_parent[leaf] = gen

    end_leaf_to_parent = {}
    for gen in gen_end_hexes:
        for leaf in get_leaves(gen, target_res_end):
            end_leaf_to_parent[leaf] = gen

    total_abs_error = 0.0

    # Itera solo sulle coppie presenti nell'originale
    for (s, d), true_count in original_flows.items():
        gen_s = start_leaf_to_parent.get(s)
        gen_d = end_leaf_to_parent.get(d)

        if gen_s is None or gen_d is None:
            reconstructed_count = 0.0
        else:
            gen_count = generalized_flows.get((gen_s, gen_d), 0.0)
            start_leaves = get_leaves(gen_s, target_res_start)
            end_leaves   = get_leaves(gen_d, target_res_end)
            reconstructed_count = gen_count / (len(start_leaves) * len(end_leaves))

        total_abs_error += abs(reconstructed_count - true_count)

    # Aggiungi errore per le coppie con count=0 nei generalized flows
    for (gen_s, gen_d), gen_count in generalized_flows.items():
        start_leaves = get_leaves(gen_s, target_res_start)
        end_leaves   = get_leaves(gen_d, target_res_end)
        count_per_leaf = gen_count / (len(start_leaves) * len(end_leaves))

        # Sottrai le coppie già contate
        for s in start_leaves:
            for d in end_leaves:
                if (s, d) not in original_flows:
                    total_abs_error += abs(count_per_leaf - 0.0)

    return total_abs_error / total_volume


def compute_discernability_and_cavg_sparse_ODkAnon(sparse_matrix: sp.csr_matrix, od_matrix, suppressed_count: int, k: int) -> dict:
    
    counts = sparse_matrix.data
    total_records = counts.sum() + suppressed_count
    total_records_avg = counts.sum()
    total_equiv_classes = sparse_matrix.nnz + suppressed_count
    total_equiv_classes_avg = sparse_matrix.nnz

    # Aggiungo una penalità per i record soppressi
    # Ogni riga soppressa conta come una classe di equivalenza
    # Quindi ogni riga ha una penalità grande quanto la grandezza del dataset
    suppression_penalty = len(od_matrix) * suppressed_count

    # C_DM: somma dei quadrati dei count >= k
    k_anonymous_counts = counts[counts >= k]
    c_dm_gen = (k_anonymous_counts ** 2).sum()
    c_dm = c_dm_gen + suppression_penalty

    # CAVG: ((total_records / total_equiv_classes) / k)    
    c_avg = (total_records_avg / total_equiv_classes_avg) / k if total_equiv_classes_avg > 0 else float('inf')

    return {
        'C_DM': c_dm,
        'C_AVG': c_avg,
        'total_records': total_records,
        'total_equivalence_classes': total_equiv_classes,
        'k': k
    }


from geopy.distance import geodesic

def calculate_generalization_distance_metric_ODkAnon(df_merged: pd.DataFrame, generalizer, tree_start, tree_end) -> Dict:
    
    print("🔍 Calcolo metrica di distanza post-generalizzazione...")
    
    # 1. Ottieni le mappature finali dalla generalizzazione
    final_start_mapping = generalizer.start_to_idx
    final_end_mapping = generalizer.end_to_idx
    
    # 2. Crea mapping da esagoni originali a esagoni generalizzati
    start_original_to_generalized = {}
    end_original_to_generalized = {}
    
    # Per ogni esagono originale, trova l'esagono generalizzato corrispondente
    # In 'start_original_to_generalized' e 'end_original_to_generalized' ci sono gli esagoni generalizzati
    unique_start_h3 = df_merged['start_h3'].unique()
    unique_end_h3 = df_merged['end_h3'].unique()
    
    print(f"📊 Mappatura {len(unique_start_h3)} esagoni origine...")
    for original_h3 in unique_start_h3:
        generalized_h3 = find_generalized_hexagon(original_h3, final_start_mapping, tree_start)
        if generalized_h3:
            start_original_to_generalized[original_h3] = generalized_h3
    
    print(f"📊 Mappatura {len(unique_end_h3)} esagoni destinazione...")
    for original_h3 in unique_end_h3:
        generalized_h3 = find_generalized_hexagon(original_h3, final_end_mapping, tree_end)
        if generalized_h3:
            end_original_to_generalized[original_h3] = generalized_h3
    
    # 3. Calcola distanze per i punti di partenza
    # Recupera le coordinate originali e calcola la distanza tra quel punto e l'esagono generalizzato (il centro dell'esagono)
    start_distances = []
    start_coords = []
    
    for idx, row in df_merged.iterrows():
        original_h3 = row['start_h3']
        original_coords = (row['start_lat'], row['start_lon'])
        
        if original_h3 in start_original_to_generalized:
            generalized_h3 = start_original_to_generalized[original_h3]
            # Funzione utlizzata per trovare il centro di un esagono H3
            generalized_coords = h3.cell_to_latlng(generalized_h3)
            
            # Distanza che tiene conto della curvatura della Terra (più precisa di una distanza euclidea che considera la Terra come piatta)
            distance = geodesic(original_coords, generalized_coords).meters
                
            start_distances.append(distance)
            start_coords.append({
                'original_h3': original_h3,
                'generalized_h3': generalized_h3,
                'original_coords': original_coords,
                'generalized_coords': generalized_coords,
                'distance': distance
            })
    
    # 4. Calcola distanze per i punti di destinazione
    end_distances = []
    end_coords = []
    
    for idx, row in df_merged.iterrows():
        original_h3 = row['end_h3']
        original_coords = (row['end_lat'], row['end_lon'])
        
        if original_h3 in end_original_to_generalized:
            generalized_h3 = end_original_to_generalized[original_h3]
            generalized_coords = h3.cell_to_latlng(generalized_h3)
            
            distance = geodesic(original_coords, generalized_coords).meters
                
            end_distances.append(distance)
            end_coords.append({
                'original_h3': original_h3,
                'generalized_h3': generalized_h3,
                'original_coords': original_coords,
                'generalized_coords': generalized_coords,
                'distance': distance
            })
    
    # 5. Calcola statistiche
    results = {
        'start_distances': {
            'mean': np.mean(start_distances) if start_distances else 0,
            'median': np.median(start_distances) if start_distances else 0,
            'std': np.std(start_distances) if start_distances else 0,
            'min': np.min(start_distances) if start_distances else 0,
            'max': np.max(start_distances) if start_distances else 0,
            'count': len(start_distances)
        },
        'end_distances': {
            'mean': np.mean(end_distances) if end_distances else 0,
            'median': np.median(end_distances) if end_distances else 0,
            'std': np.std(end_distances) if end_distances else 0,
            'min': np.min(end_distances) if end_distances else 0,
            'max': np.max(end_distances) if end_distances else 0,
            'count': len(end_distances)
        },
        'overall': {
            'mean': np.mean(start_distances + end_distances) if (start_distances or end_distances) else 0,
            'median': np.median(start_distances + end_distances) if (start_distances or end_distances) else 0,
            'std': np.std(start_distances + end_distances) if (start_distances or end_distances) else 0,
            'total_points': len(start_distances) + len(end_distances)
        },
        'mappings': {
            'start_original_to_generalized': start_original_to_generalized,
            'end_original_to_generalized': end_original_to_generalized
        },
        'detailed_coords': {
            'start': start_coords,
            'end': end_coords
        }
    }
    
    # 6. Stampa risultati
    print("\n" + "="*60)
    print("📏 METRICHE DI DISTANZA POST-GENERALIZZAZIONE")
    print("="*60)
    
    print(f"\n🎯 PUNTI DI PARTENZA:")
    print(f"   • Distanza media: {results['start_distances']['mean']:.2f} metri")
    print(f"   • Distanza mediana: {results['start_distances']['median']:.2f} metri")
    print(f"   • Deviazione standard: {results['start_distances']['std']:.2f} metri")
    print(f"   • Min-Max: {results['start_distances']['min']:.2f} - {results['start_distances']['max']:.2f} metri")
    print(f"   • Punti analizzati: {results['start_distances']['count']:,}")
    
    print(f"\n🏁 PUNTI DI DESTINAZIONE:")
    print(f"   • Distanza media: {results['end_distances']['mean']:.2f} metri")
    print(f"   • Distanza mediana: {results['end_distances']['median']:.2f} metri")
    print(f"   • Deviazione standard: {results['end_distances']['std']:.2f} metri")
    print(f"   • Min-Max: {results['end_distances']['min']:.2f} - {results['end_distances']['max']:.2f} metri")
    print(f"   • Punti analizzati: {results['end_distances']['count']:,}")
    
    print(f"\n🌍 COMPLESSIVO:")
    print(f"   • Distanza media totale: {results['overall']['mean']:.2f} metri")
    print(f"   • Distanza mediana totale: {results['overall']['median']:.2f} metri")
    print(f"   • Deviazione standard totale: {results['overall']['std']:.2f} metri")
    print(f"   • Punti totali: {results['overall']['total_points']:,}")
    
    return results

def find_generalized_hexagon(original_h3: str, final_mapping: Dict, tree) -> str:
    """
    Trova l'esagono generalizzato corrispondente a un esagono originale
    """
    # Se l'esagono è direttamente presente nella mappatura finale
    if original_h3 in final_mapping:
        return original_h3
    
    # Se non trovato, cerca tra TUTTI gli esagoni finali se l'originale è loro discendente
    for final_h3 in final_mapping.keys():
        if is_descendant_of(original_h3, final_h3):
            return final_h3
    
    return None

def is_descendant_of(child_h3: str, parent_h3: str) -> bool:
    """
    Controlla se child_h3 è discendente di parent_h3
    """
    child_res = h3.get_resolution(child_h3)
    parent_res = h3.get_resolution(parent_h3)
    
    if parent_res >= child_res:
        return False
    
    current = child_h3
    while h3.get_resolution(current) > parent_res:
        current = h3.cell_to_parent(current, h3.get_resolution(current) - 1)
    
    return current == parent_h3



class GeneralizationMetricODkAnon:
    """
    Ḡ = (1/V+) × Σ(|o| + |d|) × v_{o→d}
    """
        
    def __init__(self, k_threshold: int = 10):
        self.k_threshold = k_threshold
        
    def calculate_generalization_error(self, generalized_matrix: sp.csr_matrix, generalizer: 'OptimizedH3GeneralizedODMatrix') -> float:
                
        matrix_dense = generalized_matrix.toarray()
        return self._calculate_dense(matrix_dense, generalizer)
    
    def _calculate_dense(self, matrix: np.ndarray, generalizer) -> float:
        
        # Denominatore sommatoria dei flussi >= k (V+)
        total_volume_anonymous = 0  
        # Numeratore (Σ(|o| + |d|) * v_o→d)
        weighted_count_sum = 0
        
        rows, cols = matrix.shape
        
        # Scorriamo la matrice OD
        for i in range(rows):
            for j in range(cols):
                flow_value = matrix[i, j]
                
                # Considera solo i flussi >= k_threshold
                if flow_value >= self.k_threshold:
                    # Ottieni gli ID H3 dalle mappature
                    origin_h3 = generalizer.idx_to_start.get(j)
                    destination_h3 = generalizer.idx_to_end.get(i)
                    
                    if origin_h3 and destination_h3:
                        # '_get_node_count' dice quante celle originali sono state aggregate dentro ciascun nodo H3 generalizzato
                        origin_count = self._get_node_count(origin_h3, generalizer.tree_start)
                        destination_count = self._get_node_count(destination_h3, generalizer.tree_end)
                        
                        # Sommo i count per ottenere V+
                        total_volume_anonymous += flow_value
                        # Calcola il contributo del numeratore
                        weighted_count_sum += (origin_count + destination_count) * flow_value
        
        # Calcola l'errore medio
        if total_volume_anonymous > 0:
            mean_generalization_error = weighted_count_sum / total_volume_anonymous
        else:
            mean_generalization_error = 0.0
            
        return mean_generalization_error
    
    def _get_node_count(self, h3_id: str, tree) -> int:
        """
        Dato un nodo H3 generalizzato, conta quante celle originali (foglie) rappresenta.
        """
        # Caso limite : il nodo non è nell'albero
        if h3_id not in tree.nodes:
            return 0
        
        node = tree.nodes[h3_id]
        
        # Se è una foglia (livello più basso), conta 1 => l'esagono generalizzato rappresenta una cella originale
        if not node.children:
            return 1
        
        # Altrimenti, chiama _count_terminal_nodes per contare tutte le foglie sotto questo nodo
        return self._count_terminal_nodes(node, tree)
    
    def _count_terminal_nodes(self, node, tree) -> int:
        """
        Conta ricorsivamente tutti i nodi terminali sotto un nodo.
        """
        if not node.children:
            return 1
        
        total = 0
        for child_id in node.children:
            if child_id in tree.nodes:
                child_node = tree.nodes[child_id]
                total += self._count_terminal_nodes(child_node, tree)
            
        return total
    


# Parte dal nodo h3_id
# Se è un nodo foglia (nessn figlio/discnedente), lo aggiunge al set
# Se ha figli, li esplora ricorsivamente
# Il risultato sono tutti gli esagoni foglia che sono stati aggregati in h3_id

def get_leaf_descendants(h3_id: str, tree) -> set:
    """
    Ritorna tutti i nodi foglia discendenti di un nodo H3.
    """
    leaves = set()

    def _collect(node_id):
        node = tree.nodes.get(node_id)
        if not node or not node.children:
            leaves.add(node_id)
        else:
            for child_id in node.children:
                _collect(child_id)

    _collect(h3_id)
    return leaves

# Per ogni nodo generalizzato
# Estrae tutti i suoi discendenti foglia
# E li associa nella forma:
# {
#     foglia_1: generalizzato_A,
#     foglia_2: generalizzato_A,
#     ...
# }

generalizer.start_generalization_map = {
    leaf_h3: gen_h3 for gen_h3 in generalizer.start_to_idx
    for leaf_h3 in get_leaf_descendants(gen_h3, tree_start)
}

generalizer.end_generalization_map = {
    leaf_h3: gen_h3 for gen_h3 in generalizer.end_to_idx
    for leaf_h3 in get_leaf_descendants(gen_h3, tree_end)
}

generalizer.start_idx_map = {h3: idx for h3, idx in generalizer.start_to_idx.items()}
generalizer.end_idx_map = {h3: idx for h3, idx in generalizer.end_to_idx.items()}

# Reconstruction loss : differenza assoluta media tra i flussi originali e quelli ricostruiti, normalizzata per il volume totale

def fast_reconstruction_loss_ODkAnon(original_od_df: pd.DataFrame,
                                  generalized_matrix: sp.csr_matrix,
                                  generalizer,
                                  tree_start,
                                  tree_end) -> float:
    """
    Calcola la reconstruction loss includendo anche le celle con 0 viaggi.
    E = (1/V) * Σ |ṽ_o→d - v_o→d|
    """

    original_dict = {
        (row['start_h3'], row['end_h3']): row['count']
        for _, row in original_od_df.iterrows()
    }

    total_volume = sum(original_dict.values())
    if total_volume == 0:
        return 0.0

    total_abs_error = 0.0

    # Precomputo foglie per ogni nodo generalizzato
    start_descendants = {gen: get_leaf_descendants(gen, tree_start) for gen in generalizer.start_to_idx}
    end_descendants   = {gen: get_leaf_descendants(gen, tree_end) for gen in generalizer.end_to_idx}

    # Lista di foglie
    start_leaves = [n for n in tree_start.nodes if not tree_start.nodes[n].children]
    end_leaves   = [n for n in tree_end.nodes if not tree_end.nodes[n].children]

    for s, d in itertools.product(start_leaves, end_leaves):
        true_count = original_dict.get((s, d), 0)

        gen_start = generalizer.start_generalization_map.get(s)
        gen_end   = generalizer.end_generalization_map.get(d)

        if gen_start is None or gen_end is None:
            reconstructed_count = 0
        else:
            row_idx = generalizer.end_idx_map.get(gen_end)
            col_idx = generalizer.start_idx_map.get(gen_start)

            if row_idx is None or col_idx is None:
                reconstructed_count = 0
            else:
                gen_count = generalized_matrix[row_idx, col_idx]
                total_combinations = len(start_descendants[gen_start]) * len(end_descendants[gen_end])
                reconstructed_count = gen_count / total_combinations if total_combinations > 0 else 0

        total_abs_error += abs(reconstructed_count - true_count)

    return total_abs_error / total_volume


def fast_reconstruction_loss_weight(original_od_df: pd.DataFrame,
                                  generalized_matrix: sp.csr_matrix,
                                  generalizer,
                                  tree_start,
                                  tree_end) -> float:
    """
    Calcola la reconstruction loss includendo anche le celle con 0 viaggi.
    E = (1/V) * Σ |ṽ_o→d - v_o→d|
    """

    original_dict = {
        (row['start_h3'], row['end_h3']): row['total_weight']
        for _, row in original_od_df.iterrows()
    }

    total_volume = sum(original_dict.values())
    if total_volume == 0:
        return 0.0

    total_abs_error = 0.0

    # Precomputo foglie per ogni nodo generalizzato
    start_descendants = {gen: get_leaf_descendants(gen, tree_start) for gen in generalizer.start_to_idx}
    end_descendants   = {gen: get_leaf_descendants(gen, tree_end) for gen in generalizer.end_to_idx}

    # Lista di foglie
    start_leaves = [n for n in tree_start.nodes if not tree_start.nodes[n].children]
    end_leaves   = [n for n in tree_end.nodes if not tree_end.nodes[n].children]

    for s, d in itertools.product(start_leaves, end_leaves):
        true_count = original_dict.get((s, d), 0)

        gen_start = generalizer.start_generalization_map.get(s)
        gen_end   = generalizer.end_generalization_map.get(d)

        if gen_start is None or gen_end is None:
            reconstructed_count = 0
        else:
            row_idx = generalizer.end_idx_map.get(gen_end)
            col_idx = generalizer.start_idx_map.get(gen_start)

            if row_idx is None or col_idx is None:
                reconstructed_count = 0
            else:
                gen_count = generalized_matrix[row_idx, col_idx]
                total_combinations = len(start_descendants[gen_start]) * len(end_descendants[gen_end])
                reconstructed_count = gen_count / total_combinations if total_combinations > 0 else 0

        total_abs_error += abs(reconstructed_count - true_count)

    return total_abs_error / total_volume