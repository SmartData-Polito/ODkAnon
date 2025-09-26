import pandas as pd
import numpy as np
import h3

def filter_by_h3_hexagons(df, target_hexagons, resolution=6):
    """
    Filter the dataset, keeping only the rows where both start and end
    are located in the specified H3 hexagons.
    
    Parameters:
    - df: DataFrame with columns ori_lat, ori_lon, dst_lat, dst_lon	
    - target_hexagons: set of target H3 hexagons
    - resolution: H3 resolution (default 6, based on target hexagons)
    
    Returns:
    - Filtered DataFrame
    """
    
    # map latitude and longitude to h3 hexagon
    df['ori_h3'] = df.apply(lambda row: h3.latlng_to_cell(row['start_lat'], row['start_lon'], resolution), axis=1)
    df['dst_h3'] = df.apply(lambda row: h3.latlng_to_cell(row['end_lat'], row['end_lon'], resolution), axis=1)
    
    filtered_df = df[
        (df['ori_h3'].isin(target_hexagons)) & 
        (df['dst_h3'].isin(target_hexagons))
    ].copy()
    
    return filtered_df