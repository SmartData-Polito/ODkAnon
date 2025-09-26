import h3
import folium
import numpy as np
import pandas as pd


class GeneralizedH3Visualizer:
    def __init__(self, od_matrix, center_lat=48.8566, center_lon=2.3522):
        """
        Visualize the generalized OD matrix H3 on Folium..

        Args:
            od_matrix: DataFrame with columns ['start_h3', 'end_h3', 'count']
            center_lat, center_lon: map center coordinates
        """
        self.od_matrix = od_matrix
        self.center_lat = center_lat
        self.center_lon = center_lon
        
        # Totali per origine e destinazione
        self.origin_flows = od_matrix.groupby('start_gen')['count'].sum().to_dict()
        self.dest_flows = od_matrix.groupby('end_gen')['count'].sum().to_dict()
    
    def _h3_to_geojson(self, h3_id):
        """Convert H3 to GeoJSON"""
        boundary = h3.cell_to_boundary(h3_id)
        coords = [[[lon, lat] for lat, lon in boundary]]  # GeoJSON vuole lon, lat
        return {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": coords},
            "properties": {"h3_id": h3_id, "resolution": h3.get_resolution(h3_id)}
        }
    
    def create_map(self, max_hexagons=100, alpha=0.6, zoom_start=10):
        """Create the Folium map with origin and destination hexagons"""
        m = folium.Map(location=[self.center_lat, self.center_lon], zoom_start=zoom_start, tiles='OpenStreetMap')
        
        # Layer Origin (blue)
        origins_sorted = sorted(self.origin_flows.items(), key=lambda x: x[1], reverse=True)[:max_hexagons]
        origin_layer = folium.FeatureGroup(name="Origin (blue)", show=True)
        if origins_sorted:
            min_flow, max_flow = min(v for _, v in origins_sorted), max(v for _, v in origins_sorted)
            for h3_id, count in origins_sorted:
                geojson = self._h3_to_geojson(h3_id)
                intensity = (count - min_flow) / (max_flow - min_flow) if max_flow > min_flow else 1.0
                blue_intensity = int(255 * (0.3 + 0.7*intensity))
                fill_color = f"#{0:02x}{0:02x}{blue_intensity:02x}"
                folium.GeoJson(
                    geojson,
                    style_function=lambda x, fill_color=fill_color: {
                        'fillColor': fill_color,
                        'color': 'darkblue',
                        'weight': 1,
                        'fillOpacity': alpha,
                        'opacity': 0.8
                    },
                    tooltip=f"{count} viaggi"
                ).add_to(origin_layer)
        origin_layer.add_to(m)
        
        # Layer Destination (red)
        dest_sorted = sorted(self.dest_flows.items(), key=lambda x: x[1], reverse=True)[:max_hexagons]
        dest_layer = folium.FeatureGroup(name="Destination (red)", show=True)
        if dest_sorted:
            min_flow, max_flow = min(v for _, v in dest_sorted), max(v for _, v in dest_sorted)
            for h3_id, count in dest_sorted:
                geojson = self._h3_to_geojson(h3_id)
                intensity = (count - min_flow) / (max_flow - min_flow) if max_flow > min_flow else 1.0
                red_intensity = int(255 * (0.3 + 0.7*intensity))
                fill_color = f"#{red_intensity:02x}{0:02x}{0:02x}"
                folium.GeoJson(
                    geojson,
                    style_function=lambda x, fill_color=fill_color: {
                        'fillColor': fill_color,
                        'color': 'darkred',
                        'weight': 1,
                        'fillOpacity': alpha,
                        'opacity': 0.8
                    },
                    tooltip=f"{count} viaggi"
                ).add_to(dest_layer)
        dest_layer.add_to(m)
        
        folium.LayerControl().add_to(m)
        return m
    

class GeneralizedH3VisualizerATG:
    def __init__(self, od_matrix, center_lat=48.8566, center_lon=2.3522):
        """
        Visualize the generalized OD matrix H3 on Folium..

        Args:
            od_matrix: DataFrame with columns ['start_h3', 'end_h3', 'count']
            center_lat, center_lon: map center coordinates
        """
        self.od_matrix = od_matrix
        self.center_lat = center_lat
        self.center_lon = center_lon
        
        self.origin_flows = od_matrix.groupby('start_gen')['count'].sum().to_dict()
        self.dest_flows = od_matrix.groupby('end_gen')['count'].sum().to_dict()
    
    def _h3_to_geojson(self, h3_id):
        """Convert H3 to GeoJSON"""
        boundary = h3.cell_to_boundary(h3_id)
        coords = [[[lon, lat] for lat, lon in boundary]] 
        return {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": coords},
            "properties": {"h3_id": h3_id, "resolution": h3.get_resolution(h3_id)}
        }
    
    def create_map(self, max_hexagons=100, alpha=0.6, zoom_start=10):
        """Create the Folium map with origin and destination hexagons"""
        m = folium.Map(location=[self.center_lat, self.center_lon], zoom_start=zoom_start, tiles='OpenStreetMap')
        
        origins_sorted = sorted(self.origin_flows.items(), key=lambda x: x[1], reverse=True)[:max_hexagons]
        origin_layer = folium.FeatureGroup(name="Origin (blue)", show=True)
        if origins_sorted:
            min_flow, max_flow = min(v for _, v in origins_sorted), max(v for _, v in origins_sorted)
            for h3_id, count in origins_sorted:
                geojson = self._h3_to_geojson(h3_id)
                intensity = (count - min_flow) / (max_flow - min_flow) if max_flow > min_flow else 1.0
                blue_intensity = int(255 * (0.3 + 0.7*intensity))
                fill_color = f"#{0:02x}{0:02x}{blue_intensity:02x}"
                folium.GeoJson(
                    geojson,
                    style_function=lambda x, fill_color=fill_color: {
                        'fillColor': fill_color,
                        'color': 'darkblue',
                        'weight': 1,
                        'fillOpacity': alpha,
                        'opacity': 0.8
                    },
                    tooltip=f"{count} viaggi"
                ).add_to(origin_layer)
        origin_layer.add_to(m)
        
        dest_sorted = sorted(self.dest_flows.items(), key=lambda x: x[1], reverse=True)[:max_hexagons]
        dest_layer = folium.FeatureGroup(name="Destination (red)", show=True)
        if dest_sorted:
            min_flow, max_flow = min(v for _, v in dest_sorted), max(v for _, v in dest_sorted)
            for h3_id, count in dest_sorted:
                geojson = self._h3_to_geojson(h3_id)
                intensity = (count - min_flow) / (max_flow - min_flow) if max_flow > min_flow else 1.0
                red_intensity = int(255 * (0.3 + 0.7*intensity))
                fill_color = f"#{red_intensity:02x}{0:02x}{0:02x}"
                folium.GeoJson(
                    geojson,
                    style_function=lambda x, fill_color=fill_color: {
                        'fillColor': fill_color,
                        'color': 'darkred',
                        'weight': 1,
                        'fillOpacity': alpha,
                        'opacity': 0.8
                    },
                    tooltip=f"{count} viaggi"
                ).add_to(dest_layer)
        dest_layer.add_to(m)
        
        folium.LayerControl().add_to(m)
        return m
    

class CountAnalyzer:
    def __init__(self, counts_matrix: sp.csr_matrix, weights_matrix: sp.csr_matrix, 
                 generalizer, k_count: int, k_weight: float):
        self.weights_matrix = weights_matrix
        self.counts_matrix = counts_matrix
        self.generalizer = generalizer
        self.k_count = k_count
        self.k_weight = k_weight

    def analyze_count_anonymity(self) -> Dict:
        counts_coo = self.counts_matrix.tocoo()
        count_values = counts_coo.data
        stats = {
            'total_non_zero_cells': len(count_values),
            'min_count': int(count_values.min()) if len(count_values) > 0 else 0,
            'max_count': int(count_values.max()) if len(count_values) > 0 else 0,
            'mean_count': float(count_values.mean()) if len(count_values) > 0 else 0,
            'median_count': float(np.median(count_values)) if len(count_values) > 0 else 0,
            'std_count': float(count_values.std()) if len(count_values) > 0 else 0
        }
        below_k = count_values[count_values < self.k_count]
        stats['cells_below_k'] = len(below_k)
        stats['percent_below_k'] = (len(below_k) / len(count_values) * 100) if len(count_values) > 0 else 0
        stats['is_k_anonymous'] = len(below_k) == 0
        unique_counts, frequencies = np.unique(count_values, return_counts=True)
        stats['unique_count_values'] = len(unique_counts)
        stats['most_common_count'] = int(unique_counts[np.argmax(frequencies)])
        stats['most_common_frequency'] = int(frequencies.max())
        return stats

    def analyze_weight_anonymity(self) -> Dict:
        weights_coo = self.weights_matrix.tocoo()
        weight_values = weights_coo.data
        stats = {
            'total_non_zero_cells': len(weight_values),
            'min_weight': float(weight_values.min()) if len(weight_values) > 0 else 0,
            'max_weight': float(weight_values.max()) if len(weight_values) > 0 else 0,
            'mean_weight': float(weight_values.mean()) if len(weight_values) > 0 else 0,
            'median_weight': float(np.median(weight_values)) if len(weight_values) > 0 else 0,
            'std_weight': float(weight_values.std()) if len(weight_values) > 0 else 0
        }
        below_k = weight_values[weight_values < self.k_weight]
        stats['cells_below_k'] = len(below_k)
        stats['percent_below_k'] = (len(below_k) / len(weight_values) * 100) if len(weight_values) > 0 else 0
        stats['is_k_anonymous'] = len(below_k) == 0
        return stats

    def print_summary_report(self):
        count_stats = self.analyze_count_anonymity()
        print(f"\n🔢 COUNT (k={self.k_count})")
        print(f"   Celle sotto soglia k: {count_stats['cells_below_k']:,}")
        print(f"   Percentuale sotto k: {count_stats['percent_below_k']:.2f}%")
        print(f"   È k-anonima? {'✅ SÌ' if count_stats['is_k_anonymous'] else '❌ NO'}")
        weight_stats = self.analyze_weight_anonymity()
        print(f"\n⚖️  PESI (k={self.k_weight:.2f})")
        print(f"   Celle sotto soglia k: {weight_stats['cells_below_k']:,}")
        print(f"   Percentuale sotto k: {weight_stats['percent_below_k']:.2f}%")
        print(f"   È k-anonima? {'✅ SÌ' if weight_stats['is_k_anonymous'] else '❌ NO'}")
        return count_stats, weight_stats
    

from scipy.sparse import coo_matrix

def plot_count_distributions(weights_result, k_count=10*media_peso):
    counts_coo = coo_matrix(weights_result)
    counts = counts_coo.data

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(counts, bins=50, color='skyblue', edgecolor='black', log=True)
    axes[0].axvline(k_count, color='red', linestyle='--', linewidth=2, label=f'k = {k_count}')
    axes[0].set_title('Distribuzione completa dei Pesi')
    axes[0].set_xlabel('Valore di pesi')
    axes[0].set_ylabel('Frequenza (log)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    counts_zoom = counts[counts <= k_count*1.2]
    axes[1].hist(counts_zoom, bins=40, color='mediumseagreen', edgecolor='black')
    axes[1].axvline(k_count, color='red', linestyle='--', linewidth=2, label=f'k = {k_count}')
    axes[1].set_title('Zoom dei pesi')
    axes[1].set_xlabel('Valore di pesi')
    axes[1].set_ylabel('Frequenza')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


import folium
import h3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import scipy.sparse as sp
from branca.colormap import linear
import json
from folium.plugins import HeatMap
import matplotlib.cm as cm
import matplotlib.colors as mcolors

class H3FoliumVisualizerODkAnon:
    """
    Classe per visualizzare i risultati della generalizzazione H3 con Folium
    """
    
    def __init__(self, generalizer, center_lat=48.8566, center_lon=2.3522):
        """
        Inizializza il visualizzatore
        
        Args:
            generalizer: Istanza di OptimizedH3GeneralizedODMatrix
            center_lat, center_lon: Coordinate del centro della mappa (default: Torino)
        """
        self.generalizer = generalizer
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.sparse_matrix = generalizer.current_matrix_sparse
        
        # Estrai dati dalla matrice sparse per visualizzazione
        self._extract_visualization_data()
    
    def _extract_visualization_data(self):
        print("📊 Estrazione dati per visualizzazione...")
        coo = self.sparse_matrix.tocoo()

        # Prepara liste di mapping con None per indici mancanti
        list_idx_to_start = [None] * self.sparse_matrix.shape[1]
        for idx, h3_id in self.generalizer.idx_to_start.items():
            list_idx_to_start[idx] = h3_id

        list_idx_to_end = [None] * self.sparse_matrix.shape[0]
        for idx, h3_id in self.generalizer.idx_to_end.items():
            list_idx_to_end[idx] = h3_id

        self.origin_data = {}
        self.destination_data = {}
        self.od_pairs = []

        # Calcola flussi totali per origine
        for start_idx, h3_id in enumerate(list_idx_to_start):
            if h3_id is None:
                continue
            total_flow = self.sparse_matrix[:, start_idx].sum()
            if total_flow > 0:
                self.origin_data[h3_id] = total_flow

        # Calcola flussi totali per destinazione
        for end_idx, h3_id in enumerate(list_idx_to_end):
            if h3_id is None:
                continue
            total_flow = self.sparse_matrix[end_idx, :].sum()
            if total_flow > 0:
                self.destination_data[h3_id] = total_flow

        # Estrai coppie OD
        for i, j, data in zip(coo.row, coo.col, coo.data):
            if data > 0:
                if j < len(list_idx_to_start) and i < len(list_idx_to_end):
                    origin_h3 = list_idx_to_start[j]
                    dest_h3 = list_idx_to_end[i]
                    if origin_h3 is not None and dest_h3 is not None:
                        self.od_pairs.append((origin_h3, dest_h3, data))

        print(f"✅ Estratti {len(self.origin_data)} origini, {len(self.destination_data)} destinazioni, {len(self.od_pairs)} coppie OD")

    def _h3_to_geojson(self, h3_id: str) -> Dict:
        """Converte un esagono H3 in GeoJSON"""
        boundary = h3.cell_to_boundary(h3_id)
        # H3 restituisce (lat, lon), GeoJSON vuole (lon, lat)
        coords = [[[lon, lat] for lat, lon in boundary]]
        
        return {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": coords
            },
            "properties": {
                "h3_id": h3_id,
                "resolution": h3.get_resolution(h3_id)
            }
        }
    
    def create_base_map(self, zoom_start=10) -> folium.Map:
        """Crea la mappa base"""
        m = folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=zoom_start,
            tiles='OpenStreetMap'
        )
        
        # Aggiungi controlli layer
        folium.plugins.Fullscreen().add_to(m)
        
        return m
    
    def add_origin_hexagons(self, m: folium.Map, max_hexagons=100, alpha=0.6):
        """Aggiunge gli esagoni di origine alla mappa"""
        print(f"🔵 Aggiunta esagoni origine (max {max_hexagons})...")
        
        # Ordina per flusso e prendi i top
        sorted_origins = sorted(self.origin_data.items(), key=lambda x: x[1], reverse=True)

        # k = self.generalizer.k_threshold
        # sorted_origins = [(h3_id, flow) for h3_id, flow in sorted_origins if flow >= k]

        top_origins = sorted_origins[:max_hexagons]
        
        if not top_origins:
            print("⚠️ Nessun esagono origine da visualizzare")
            return
        
        # Calcola range per normalizzazione colori
        flows = [flow for _, flow in top_origins]
        min_flow, max_flow = min(flows), max(flows)
        
        # Gruppo layer per le origini
        origin_group = folium.FeatureGroup(name=f'Origini (Top {len(top_origins)})', show=True)
        
        for h3_id, flow in top_origins:
            try:
                # Ottieni geometria
                geojson = self._h3_to_geojson(h3_id)
                
                # Calcola intensità colore normalizzata
                if max_flow > min_flow:
                    intensity = (flow - min_flow) / (max_flow - min_flow)
                else:
                    intensity = 1.0
                
                # Usa colori esadecimali invece di rgba per evitare problemi
                # Calcola il blu con intensità variabile
                blue_intensity = int(255 * (0.3 + intensity * 0.7))
                fill_color = f"#{0:02x}{0:02x}{blue_intensity:02x}"
                
                # Aggiungi esagono con style_function semplificata
                folium.GeoJson(
                    geojson,
                    style_function=lambda x: {
                        'fillColor': fill_color,
                        'color': 'darkblue',
                        'weight': 1,
                        'fillOpacity': alpha,
                        'opacity': 0.8
                    },
                    popup=folium.Popup(
                        f"""
                        <b>Origine H3</b><br>
                        ID: {h3_id}<br>
                        Risoluzione: {h3.get_resolution(h3_id)}<br>
                        Flusso totale: {flow:,}<br>
                        Rank: {sorted_origins.index((h3_id, flow)) + 1}
                        """,
                        max_width=200
                    ),
                    tooltip=f"Origine: {flow:,} viaggi"
                ).add_to(origin_group)
                
            except Exception as e:
                print(f"⚠️ Errore nell'aggiungere esagono origine {h3_id}: {e}")
        
        origin_group.add_to(m)
        print(f"✅ Aggiunti {len(top_origins)} esagoni origine")
    
    def add_destination_hexagons(self, m: folium.Map, max_hexagons=100, alpha=0.6):
        """Aggiunge gli esagoni di destinazione alla mappa"""
        print(f"🔴 Aggiunta esagoni destinazione (max {max_hexagons})...")
        
        # Ordina per flusso e prendi i top
        sorted_destinations = sorted(self.destination_data.items(), key=lambda x: x[1], reverse=True)

        # k = self.generalizer.k_threshold
        # sorted_destinations = [(h3_id, flow) for h3_id, flow in sorted_destinations if flow >= k]

        top_destinations = sorted_destinations[:max_hexagons]
        
        if not top_destinations:
            print("⚠️ Nessun esagono destinazione da visualizzare")
            return
        
        # Gruppo layer per le destinazioni
        dest_group = folium.FeatureGroup(name=f'Destinazioni (Top {len(top_destinations)})', show=True)
        
        flows = [flow for _, flow in top_destinations]
        min_flow, max_flow = min(flows), max(flows)
        
        for h3_id, flow in top_destinations:
            try:
                geojson = self._h3_to_geojson(h3_id)
                
                # Calcola intensità colore
                if max_flow > min_flow:
                    intensity = (flow - min_flow) / (max_flow - min_flow)
                else:
                    intensity = 1.0
                
                # Usa colori esadecimali per il rosso
                red_intensity = int(255 * (0.3 + intensity * 0.7))
                fill_color = f"#{red_intensity:02x}{0:02x}{0:02x}"
                
                folium.GeoJson(
                    geojson,
                    style_function=lambda x: {
                        'fillColor': fill_color,
                        'color': 'darkred',
                        'weight': 1,
                        'fillOpacity': alpha,
                        'opacity': 0.8
                    },
                    popup=folium.Popup(
                        f"""
                        <b>Destinazione H3</b><br>
                        ID: {h3_id}<br>
                        Risoluzione: {h3.get_resolution(h3_id)}<br>
                        Flusso totale: {flow:,}<br>
                        Rank: {sorted_destinations.index((h3_id, flow)) + 1}
                        """,
                        max_width=200
                    ),
                    tooltip=f"Destinazione: {flow:,} viaggi"
                ).add_to(dest_group)
                
            except Exception as e:
                print(f"⚠️ Errore nell'aggiungere esagono destinazione {h3_id}: {e}")
        
        dest_group.add_to(m)
        print(f"✅ Aggiunti {len(top_destinations)} esagoni destinazione")