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