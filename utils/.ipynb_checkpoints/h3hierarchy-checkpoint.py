import pandas as pd
import numpy as np
import h3
from collections import defaultdict
from typing import Dict, Set, List, Optional, Tuple

class H3TreeNode:
    """H3 hierarchy tree node"""
    def __init__(self, h3_id: str, resolution: int, total_weight: int = 0, count: int = 0):
        self.h3_id = h3_id
        self.resolution = resolution
        self.total_weight = total_weight
        self.count = count
        self.children: Dict[str, 'H3TreeNode'] = {}
        self.parent: Optional['H3TreeNode'] = None
    
    def add_child(self, child: 'H3TreeNode'):
        """Adds a child to the node"""
        self.children[child.h3_id] = child
        child.parent = self
    
    def add_weight(self, weight: int):
        self.total_weight += weight
        if self.parent:
            self.parent.add_weight(weight)
    
    def add_count(self, count: int):
        self.count += count
        if self.parent:
            self.parent.add_count(count)
    
    def __repr__(self):
        return (f"H3Node(id={self.h3_id}, res={self.resolution}, "
                f"total_weight={self.total_weight}, count={self.count}, children={len(self.children)})")

class H3HierarchicalTree:
    def __init__(self, od_matrix: pd.DataFrame, target_resolution: int = 11, hex_column: str = 'start_h3'):
        self.od_matrix = od_matrix
        self.target_resolution = target_resolution
        self.hex_column = hex_column  # 'start_h3' o 'end_h3'
        self.nodes: Dict[str, H3TreeNode] = {}
        self.root = None
        self.min_resolution = None  # Sarà calcolato dinamicamente
        
    def get_all_hexagons(self) -> Set[str]:
        """Extracts all unique hexagons from the specified column"""
        return set(self.od_matrix[self.hex_column].unique())
    
    def get_resolution_coverage(self, hexagons: Set[str], target_res: int) -> Set[str]:
        """
        Obtains all target resolution hexagons covering the area defined by the input hexagons.
        """
        coverage_hexagons = set()
        
        for hex_id in hexagons:
            current_res = h3.get_resolution(hex_id)
            
            if current_res == target_res:
                coverage_hexagons.add(hex_id)
            elif current_res < target_res:
                # Dobbiamo espandere a risoluzione più alta
                children = self._get_all_children_at_resolution(hex_id, target_res)
                coverage_hexagons.update(children)
            else:
                # Dobbiamo salire a risoluzione più bassa
                parent = h3.cell_to_parent(hex_id, target_res)
                coverage_hexagons.add(parent)
        
        return coverage_hexagons
    
    def find_optimal_min_resolution(self, hexagons: Set[str]) -> int:
        """
        Find the highest resolution where there is still only one node covering all hexagons.
        """        
        # For each resolution from 0 to the target, count how many nodes are needed to cover all hexagons.
        resolution_stats = {}
        
        for resolution in range(0, self.target_resolution + 1):
            ancestors = set()
            for hex_id in hexagons:
                current_res = h3.get_resolution(hex_id)
                if current_res >= resolution:
                    ancestor = h3.cell_to_parent(hex_id, resolution)
                    ancestors.add(ancestor)
                else:
                    # add the hexagon directly if already with a lower resolution, 
                    ancestors.add(hex_id)
            
            resolution_stats[resolution] = len(ancestors)
            print(f"Resolution {resolution}: {len(ancestors)} nodes")
        
        # Find the highest resolution with count = 1
        optimal_resolution = 0
        for resolution in range(self.target_resolution, -1, -1):
            if resolution_stats[resolution] == 1:
                optimal_resolution = resolution
                break
        
        # print(f"optimal resolution: {optimal_resolution}")
        return optimal_resolution
    
    def get_siblings(self, node_id: str) -> List[str]:
        """Returns the h3_ids of the sibling nodes of node_id (other hexagons within the same parent)"""
        if node_id not in self.nodes:
            return []

        node = self.nodes[node_id]
        parent = node.parent
        if parent is None:
            return []

        siblings = [child.h3_id for child in parent.children.values() if child.h3_id != node_id]
        return siblings
    
    def get_parent(self, node_id: str) -> Optional[str]:
        """Returns the h3_id of the parent node, or None for root node."""
        if node_id not in self.nodes:
            return None
        node = self.nodes[node_id]
        if node.parent is None:
            return None
        return node.parent.h3_id
    
    def _get_all_children_at_resolution(self, hex_id: str, target_res: int) -> Set[str]:
        """Recursively obtains all children at a specific resolution"""
        current_res = h3.get_resolution(hex_id)
        
        if current_res == target_res:
            return {hex_id}
        elif current_res > target_res:
            return set()
        
        children = set()
        direct_children = h3.cell_to_children(hex_id, current_res + 1)
        
        for child in direct_children:
            children.update(self._get_all_children_at_resolution(child, target_res))
        
        return children
    
    def build_hierarchy_path(self, hex_id: str, min_resolution: int) -> List[str]:
        """Builds the hierarchical path from a hexagon to the minimum resolution"""
        path = [hex_id]
        current = hex_id
        current_res = h3.get_resolution(current)
        
        while current_res > min_resolution:
            parent = h3.cell_to_parent(current, current_res - 1)
            path.append(parent)
            current = parent
            current_res -= 1
        
        return path
    
    def create_tree_structure(self):
        """Create the optimized tree structure"""
        target_hexagons = self.get_all_hexagons()
        
        # Get full coverage at the target resolution
        coverage_hexagons = self.get_resolution_coverage(target_hexagons, self.target_resolution)
        
        # Find the optimal minimum resolution (the highest one with count=1)
        self.min_resolution = self.find_optimal_min_resolution(coverage_hexagons)
        
        # print(f"resolution tree from {self.min_resolution} to {self.target_resolution}")
        
        # Build all hierarchical paths
        all_paths = []
        for hex_id in coverage_hexagons:
            path = self.build_hierarchy_path(hex_id, self.min_resolution)
            all_paths.append(path)
        
        for path in all_paths:
            for hex_id in path:
                if hex_id not in self.nodes:
                    resolution = h3.get_resolution(hex_id)
                    self.nodes[hex_id] = H3TreeNode(hex_id, resolution)
        
        # parent-child relation
        for path in all_paths:
            for i in range(len(path) - 1):
                child_id = path[i]
                parent_id = path[i + 1]
                self.nodes[parent_id].add_child(self.nodes[child_id])
        
        # Identify the root
        self.root = self.nodes[self._find_root_hexagon(coverage_hexagons, self.min_resolution)]
        
        return self
    
    def _find_root_hexagon(self, hexagons: Set[str], min_resolution: int) -> str:
        sample_hex = next(iter(hexagons))
        return h3.cell_to_parent(sample_hex, min_resolution)
    
    def populate_counts(self):
        # Group the two metrics by hexagon
        agg_df = self.od_matrix.groupby(self.hex_column).agg({'total_weight': 'sum', 'count': 'sum'}).reset_index()
        
        for _, row in agg_df.iterrows():
            hex_id = row[self.hex_column]
            weight = int(row['total_weight'])
            count = int(row['count'])
            
            target_hex = self._map_to_target_resolution(hex_id)
            
            if target_hex in self.nodes:
                self.nodes[target_hex].add_weight(weight)
                self.nodes[target_hex].add_count(count)
            else:
                print(f"hexagon {target_hex} not found")
        
        return self
    
    def _map_to_target_resolution(self, hex_id: str) -> str:
        """Map a hexagon to the target resolution"""
        current_res = h3.get_resolution(hex_id)
        
        if current_res == self.target_resolution:
            return hex_id
        elif current_res < self.target_resolution:
            # Take the first child available
            children = self._get_all_children_at_resolution(hex_id, self.target_resolution)
            return next(iter(children)) if children else hex_id
        else:
            return h3.cell_to_parent(hex_id, self.target_resolution)
    
    def get_tree_statistics(self) -> Dict:
        if not self.root:
            return {}
        
        stats = {
            'total_nodes': len(self.nodes),
            'root_resolution': self.root.resolution,
            'min_resolution': self.min_resolution,
            'target_resolution': self.target_resolution,
            'total_weight': self.root.total_weight,
            'nodes_by_resolution': defaultdict(int),
            'resolution_range': f"{self.min_resolution} → {self.target_resolution}"
        }
        
        for node in self.nodes.values():
            stats['nodes_by_resolution'][node.resolution] += 1
        
        return stats
    
    def print_tree(self, max_children_per_level: int = 10):
        if not self.root:
            print("Tree not built")
            return
        
        print(f"OPTIMIZED TREE STRUCTURE: resolution range {self.min_resolution} → {self.target_resolution}")
        
        def _print_node(node: H3TreeNode, depth: int = 0, is_last: bool = True, prefix: str = ""):
            connector = "└─ " if is_last else "├─ "
            print(f"{prefix}{connector}{node.h3_id} (res:{node.resolution}, total_weight:{node.total_weight}, count:{node.count}, children:{len(node.children)})")
            
            if is_last:
                child_prefix = prefix + "   "
            else:
                child_prefix = prefix + "│  "
            
            children_list = list(node.children.values())
            
            if len(children_list) <= max_children_per_level:
                for i, child in enumerate(children_list):
                    is_last_child = (i == len(children_list) - 1)
                    _print_node(child, depth + 1, is_last_child, child_prefix)
            else:
                for i in range(max_children_per_level):
                    child = children_list[i]
                    is_last_child = (i == max_children_per_level - 1) and (len(children_list) == max_children_per_level)
                    _print_node(child, depth + 1, is_last_child, child_prefix)
                
                remaining = len(children_list) - max_children_per_level
                print(f"{child_prefix}└─ ... and other {remaining} children with the same pattern")
        
        _print_node(self.root, 0, True, "")

def create_h3_hierarchical_tree(od_matrix_df: pd.DataFrame, target_resolution: int = 10, hex_column: str = 'start_h3',verbose: bool=False):
    """
    Create an optimized H3 hierarchical tree from the OD matrix dataset
    
    Args:
        od_matrix_df: DataFrame with column 'start_h3', 'end_h3', 'count'
        target_resolution: Target resolution for tree leaves
        hex_column: Column to be analyzed ('start_h3' o 'end_h3')
    
    Returns:
        H3HierarchicalTree: Constructed and optimized hierarchical tree
    """
    
    tree = H3HierarchicalTree(od_matrix_df, target_resolution, hex_column)
    tree.create_tree_structure()
    tree.populate_counts()
    
    stats = tree.get_tree_statistics()
    print(f"OPTIMIZED TREE STATISTICS ({hex_column.upper()})")
    print(stats)
    
    # Calculate savings in nodes
    total_resolutions_possible = target_resolution + 1  # da 0 a target
    resolutions_used = len(stats['nodes_by_resolution'])
    resolutions_saved = total_resolutions_possible - resolutions_used
    
    print(f"OPTIMIZATIONS:")
    print(f"Saved resolutions: {resolutions_saved}")
    print(f"Tree efficiency: {resolutions_used}/{total_resolutions_possible} levels used")
    if verbose:
        tree.print_tree()
    
    return tree