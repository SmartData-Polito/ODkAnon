import pandas as pd
import os
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import ast
import json
import os
import re
import itertools
import folium
import h3

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.patches as patches
from itertools import combinations
from collections import defaultdict
from collections import Counter
from typing import Dict, Set, List, Optional, Tuple

from utils.h3hierarchy import create_h3_hierarchical_tree
from utils.metrics import compute_discernability_and_cavg_sparse
from utils.metrics import calculate_generalization_distance_metric_ATG
from utils.metrics import GeneralizationMetricATG
from utils.metrics import fast_reconstruction_loss_ATG
from utils.metrics import compute_discernability_and_cavg_weight_ATG
from utils.metrics import GeneralizationMetricWeightATG
from utils.metrics import fast_reconstruction_loss_weight_ATG
from utils.visualization import H3FoliumODVisualizerATG

### Data preparation

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

# Taking only hexagons in the center of Paris
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
filtered_df = df_merged.merge(
    od_matrix[['start_h3', 'end_h3']],
    on=['start_h3', 'end_h3'],
    how='inner'
)

# Creating hierarchical trees
tree_start = create_h3_hierarchical_tree(od_matrix, target_resolution=10, hex_column='start_h3')
tree_end = create_h3_hierarchical_tree(od_matrix, target_resolution=10, hex_column='end_h3')


# ATG-Soft algorithm
import numpy as np
import pandas as pd
import weakref
from math import *

class H3SbaAggregator:
    def __init__(self, od_matrix, tree_start, tree_end, param):
        self.od_matrix = od_matrix.copy()
        self.od_matrix = self.od_matrix.rename(columns={
            'start_h3': 'oi', 
            'end_h3': 'di', 
            'count': 'vol',
            'total_weight': 'weight'
        })
        
        self.tree_start = tree_start
        self.tree_end = tree_end
        self.param = param

        if param.get('target_vol_o') is None:
            raise ValueError('Must set a target_vol_o for tree aggregation')
        self.target_vol_o = param['target_vol_o']
        
        self.anon_thres = param['anon_thres']
        self.suppr_thres_frac = param['suppr_thres_frac']

        if 't' in self.od_matrix.columns:
            vol_ori_df = self.od_matrix.groupby('t')['vol'].sum().reset_index()
        else:
            self.od_matrix['t'] = 0
            vol_ori_df = pd.DataFrame({'t': [0], 'vol': [self.od_matrix['vol'].sum()]})

        self.eval_df = pd.DataFrame(columns=['t', 'vol_ori', 'vol_kept', 'nb_flows'])
        self.od_matrix_agg = pd.DataFrame(columns=['coi', 'cdi', 'vol', 'weight', 't', 'censored'])
        self.od_matrix_agg['censored'] = self.od_matrix_agg['censored'].astype(bool)

        for _, vol_ori_row in vol_ori_df.iterrows():
            timestep = vol_ori_row['t']
            od_matrix_agg_timestep = self.solve_timestep(timestep)

            total_vol = od_matrix_agg_timestep['vol'].sum()
            if total_vol != vol_ori_row['vol']:
                if total_vol >= self.anon_thres:
                    print(f"Warning: missing some volumes! ({od_matrix_agg_timestep['vol'].sum()} vs {vol_ori_row['vol']})")
                else:
                    print('Censored everything as total volume < anon thres')

            self.od_matrix_agg = pd.concat([self.od_matrix_agg, od_matrix_agg_timestep])

            eval_row = {
                't': timestep,
                'vol_ori': vol_ori_row['vol'],
                'vol_kept': (od_matrix_agg_timestep['vol']*(~od_matrix_agg_timestep['censored'])).sum(),
                'nb_flows': len(self.od_matrix_timestep),
            }
            self.eval_df = pd.concat([self.eval_df, pd.DataFrame(eval_row, index=[len(self.eval_df)])])

    def solve_timestep(self, timestep):
        ori_df = self.get_ori_df(timestep)
        total_vol = ori_df.sum()
        
        if total_vol < self.anon_thres:
            return pd.DataFrame({'coi': [self.tree_start.root.h3_id],
                                 'cdi': [self.tree_end.root.h3_id],
                                 'vol': 0,
                                 'weight': 0,
                                 't': timestep,
                                 'censored': total_vol != 0})
        elif total_vol == self.anon_thres:
            return pd.DataFrame({'coi': [self.tree_start.root.h3_id],
                                 'cdi': [self.tree_end.root.h3_id],
                                 'vol': total_vol,
                                 'weight': self.od_matrix_timestep['weight'].sum(),
                                 't': timestep,
                                 'censored': False})
        
        qo = H3VQuadtree(pop_df=ori_df,
                         h3_root=self.tree_start.root,
                         target_vol=self.target_vol_o,
                         ci_col='oi', vol_col='vol')
        o_partition = qo.flat_leaves

        od_matrix_agg_timestep = self.get_dest_agg(o_partition)
        od_matrix_agg_timestep['t'] = timestep
        od_matrix_agg_timestep['censored'] = (od_matrix_agg_timestep['vol'] < self.param['anon_thres']) & (od_matrix_agg_timestep['vol'] != 0)

        return od_matrix_agg_timestep

    def get_dest_agg(self, clusters_o):
        dest_df = self.get_dest_df(clusters_o)

        tree = H3SbaTree(pop_df=dest_df,
                         h3_root=self.tree_end.root,
                         clusters_o=clusters_o,
                         anon_thres=self.anon_thres)

        OD_report = tree.sba_solve(self.suppr_thres_frac * sum([co.vol for co in clusters_o]))
        od_matrix_agg_timestep = pd.DataFrame(OD_report, columns=['coi', 'cdi', 'vol', 'weight'])

        return od_matrix_agg_timestep

    def get_ori_df(self, timestep):
        self.od_matrix_timestep = self.od_matrix[self.od_matrix['t'] == timestep]
        ori_df = self.od_matrix_timestep.groupby('oi')['vol'].sum()
        return ori_df

    def get_dest_df(self, clusters_o):
        nb_leaves_by_cluster_o = [len(cluster_o.flat_leaves_name) for cluster_o in clusters_o]
        oicoidf = pd.DataFrame({
             'oi': np.concatenate([cluster_o.flat_leaves_name for cluster_o in clusters_o]),
             'coi': np.repeat([cluster_o.h3_node.h3_id for cluster_o in clusters_o], nb_leaves_by_cluster_o)
        })

        dest_df = self.od_matrix_timestep.merge(oicoidf, on='oi', how='outer').drop(columns=['oi'])
        dest_df = dest_df.groupby(['coi', 'di']).agg({'vol': 'sum', 'weight': 'sum'}).reset_index()
        dest_df = dest_df.groupby('di').agg(list)

        return dest_df


class H3SbaTree:
    def __init__(self, pop_df, h3_root, clusters_o, anon_thres=0):
        self.pop_df = pop_df
        self.anon_thres = anon_thres
        
        self.clusters_o = clusters_o
        self.cluster_order = {cluster.h3_node.h3_id: i for i, cluster in enumerate(clusters_o)}
        
        self.area = h3_root.resolution
        self.areas_o = np.array([co.h3_node.resolution for co in self.clusters_o])
        
        self.root = H3SbaNode(h3_root, weakref.ref(self))

    def sba_solve(self, S):
        left_delta = 0
        self.root.activate(left_delta)
        left_slope = self.root.best_vol_suppr.sum() - S
        left_score = self.root.best_score.sum() - left_delta*S
        if left_slope < 0:
            return self.get_leaves_arr()

        right_delta = self.area
        self.root.activate(right_delta)
        right_slope = self.root.best_vol_suppr.sum() - S
        right_score = self.root.best_score.sum() - right_delta*S
        
        while right_slope > 0:
            left_delta = right_delta
            left_slope = right_slope
            left_score = right_score
            right_delta += self.area
            self.root.activate(right_delta)
            right_slope = self.root.best_vol_suppr.sum() - S
            right_score = self.root.best_score.sum() - right_delta*S
        
        while True:
            mid_delta = (left_slope*left_delta - right_slope*right_delta + right_score - left_score)/(left_slope - right_slope)
            self.root.activate(mid_delta)
            mid_slope = self.root.best_vol_suppr.sum() - S
            mid_score = self.root.best_score.sum() - mid_delta*S
            
            if round(mid_delta-left_delta, 5) == 0 or round(mid_delta-right_delta, 5) == 0:
                return self.get_leaves_arr()
            else:
                if mid_slope > 0:
                    left_delta = mid_delta
                    left_slope = mid_slope
                    left_score = mid_score
                else:
                    right_delta = mid_delta
                    right_slope = mid_slope
                    right_score = mid_score

    def get_leaves_arr(self):
        leaves_arr = []
        for ori_counter in range(len(self.cluster_order)):
            self.root.get_leaves_arr_rec(leaves_arr, ori_counter)
        return leaves_arr


class H3SbaNode:
    def __init__(self, h3_tree_node, tree_weakref, parent=None):
        self.tree_weakref = tree_weakref
        self.h3_node = h3_tree_node
        self.k = self.tree_weakref().anon_thres
        
        self.children = [H3SbaNode(child, tree_weakref, self) for child in h3_tree_node.children.values()]
        cluster_order = self.tree_weakref().cluster_order
        self.vol_raw = np.zeros(len(cluster_order))
        self.weight_raw = np.zeros(len(cluster_order))
        
        if len(self.children) == 0:
            row = None
            try:
                row = self.tree_weakref().pop_df.loc[self.h3_node.h3_id]
            except KeyError:
                pass
            if row is not None:
                vols = row['vol']
                weights = row['weight']
                cois = row['coi']
                for i in range(len(cois)):
                    self.vol_raw[cluster_order[cois[i]]] = vols[i]
                    self.weight_raw[cluster_order[cois[i]]] = weights[i]
        elif len(self.children) > 0:
            self.vol_raw = np.sum([c.vol_raw for c in self.children], axis=0)
            self.weight_raw = np.sum([c.weight_raw for c in self.children], axis=0)
        
        self.censored = (self.vol_raw > 0) & (self.vol_raw < self.k)
        self.agg_vol_suppr = self.vol_raw * self.censored
        
        self.area = self.h3_node.resolution
        self.agg_error = self.vol_raw * (~self.censored) * (self.tree_weakref().areas_o + self.area)

    def activate(self, delta):
        for c in self.children:
            c.activate(delta)

        self.split_this = np.zeros(len(self.tree_weakref().clusters_o), dtype=bool)
        self.best_vol_suppr = self.agg_vol_suppr.copy()
        self.best_score = self.agg_error + self.agg_vol_suppr * delta

        if len(self.children) > 0:
            split_score = np.sum([c.best_score for c in self.children], axis=0)
            split_mask = (split_score < self.best_score) & (self.vol_raw > self.k)
            self.split_this[split_mask] = True
            self.best_score[split_mask] = split_score[split_mask]
            split_vol_suppr = np.sum([c.best_vol_suppr for c in self.children], axis=0)
            self.best_vol_suppr[split_mask] = split_vol_suppr[split_mask]

    def summary(self, ori_counter):
        cluster_h3_id = self.tree_weakref().clusters_o[ori_counter].h3_node.h3_id
        return [cluster_h3_id, self.h3_node.h3_id, self.vol_raw[ori_counter], self.weight_raw[ori_counter]]

    def get_leaves_arr_rec(self, leaves_arr, ori_counter):
        if not self.split_this[ori_counter]:
            leaves_arr += [self.summary(ori_counter)]
        else:
            for c in self.children:
                c.get_leaves_arr_rec(leaves_arr, ori_counter)


class H3VQuadtree:
    def __init__(self, pop_df, h3_root, target_vol, ci_col, vol_col):
        self.h3_root = h3_root
        self.pop_df = pop_df
        self.target_vol = target_vol
        self.flat_leaves = []
        self.ci_col = ci_col
        self.vol_col = vol_col
        self.grow()

    def grow(self):
        self.root = H3VQuadNode(self.h3_root, self)
        self.root.compute_best_split()
        self.root.keep_subtree(keep_condition=lambda x: x.split_this)
        self.flat_leaves = self.flatten_leaves()

    def flatten_leaves(self):
        flat_leaves = []
        self.root.flatten_leaves_rec(flat_leaves)
        return flat_leaves

    def get_total_error(self):
        return sum([leaf.error for leaf in self.flat_leaves])


class H3VQuadNode:
    def __init__(self, h3_tree_node, tree):
        self.tree = tree
        self.h3_node = h3_tree_node
        self.children = [H3VQuadNode(child, tree) for child in h3_tree_node.children.values()]
        if len(self.children) == 0:
            try:
                self.vol = self.tree.pop_df.loc[self.h3_node.h3_id]
            except KeyError:
                self.vol = 0
        else:
            self.vol = sum([c.vol for c in self.children])
        self.h3_node.vol = self.vol
        if len(self.children) == 0:
            self.flat_leaves_name = [self.h3_node.h3_id]
        else:
            self.flat_leaves_name = []
            for child in self.children:
                self.flat_leaves_name.extend(child.flat_leaves_name)
        self.error = self.compute_error()

    def compute_error(self):
        if self.vol == 0:
            return 0
        return (self.tree.target_vol - self.vol)**2

    def keep_subtree(self, keep_condition):
        if keep_condition(self):
            for c in self.children:
                c.keep_subtree(keep_condition)
        else:
            self.children = []

    def compute_best_split(self):
        self.split_this = False
        self.best_score = self.error
        if self.vol > 0:
            if len(self.children) == 0:
                self.split_this = False
            else:
                for c in self.children:
                    c.compute_best_split()
                split_score = sum([c.best_score for c in self.children])
                if split_score <= self.error:
                    self.split_this = True
                    self.best_score = split_score

    def flatten_leaves_rec(self, flat_leaves):
        if len(self.children) == 0:
            flat_leaves += [self]
        else:
            for c in self.children:
                c.flatten_leaves_rec(flat_leaves)


param = {
    'anon_thres': 10,
    'suppr_thres_frac': 0.1,
    'target_vol_o': 1000,
}

aggregator = H3SbaAggregator(
    od_matrix=od_matrix,
    tree_start=tree_start,
    tree_end=tree_end,
    param=param
)

od_matrix_agg = aggregator.od_matrix_agg

censored_rows = od_matrix_agg[od_matrix_agg["censored"] == True]
suppressed_count = od_matrix_agg[od_matrix_agg["censored"] == True]["vol"].sum()

od_matrix_agg = od_matrix_agg[od_matrix_agg["censored"] == False]
od_matrix_agg = od_matrix_agg.rename(columns={
    "coi": "start_h3",
    "cdi": "end_h3",
    "vol": "count"
})

od_matrix_agg = od_matrix_agg.drop(columns=["t", "censored"], errors="ignore")
od_matrix_agg = od_matrix_agg[od_matrix_agg['count'] > 0.0]
od_matrix_agg


# Results and visualization

visualizer = H3FoliumODVisualizerATG(od_matrix_agg)

mappa = visualizer.create_base_map(zoom_start=11)
visualizer.add_origin_hexagons(mappa)
visualizer.add_destination_hexagons(mappa)
folium.LayerControl(collapsed=False).add_to(mappa)
mappa


distance_results = calculate_generalization_distance_metric_ATG(
   df=filtered_df, 
   od_matrix_generalized=od_matrix_agg
)


metric = GeneralizationMetricATG(k_threshold=10)
error = metric.calculate_generalization_error(od_matrix_agg, od_matrix)
print(f"Errore di generalizzazione medio Ḡ: {error:.3f}")


loss = fast_reconstruction_loss_ATG(
    original_od_df=od_matrix,
    od_matrix_generalized=od_matrix_agg
)
print(f"Reconstruction Loss: {loss:.6f}")


metrics = compute_discernability_and_cavg_weight_ATG(od_matrix_agg, k=10*media_peso, suppressed_count=suppressed_count)
print("\n📊 Metrics di Discernibilità e CAVG:")
print(f"C_DM: {metrics['C_DM']:,}")
print(f"C_AVG: {metrics['C_AVG']:.4f}")


metric = GeneralizationMetricWeightATG(k_threshold=10*media_peso)
error = metric.calculate_generalization_error(od_matrix_agg, od_matrix)
print(f"Errore di generalizzazione medio Ḡ: {error:.3f}")


loss = fast_reconstruction_loss_weight_ATG(
    original_od_df=od_matrix,
    od_matrix_generalized=od_matrix_agg
)
print(f"Reconstruction Loss: {loss:.6f}")