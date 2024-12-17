import pyproj
from pyproj import Transformer
import utm
import folium

#%%
def UTM2WGS(easting, northing, zone_number = 36, zone_letter='U'):
    # Convert UTM coordinates to (latitude, longitude)
    lat, lon = utm.to_latlon(easting, northing, zone_number, zone_letter)
    return lat, lon

def ITM2WGS(itm_x, itm_y):
    # Create a transformer from ITM (EPSG:2039) to WGS84 (EPSG:4326)
    transformer = Transformer.from_crs("EPSG:2039", "EPSG:4326")    
    # Convert to WGS84 (latitude, longitude)
    wgs84_lon, wgs84_lat = transformer.transform(itm_x, itm_y)
    
    return wgs84_lon, wgs84_lat
    # print(f"ITM coordinates ({itm_x}, {itm_y}) correspond to WGS84 coordinates ({wgs84_lat}, {wgs84_lon})")

#%% plot the points on the map
def createFoliumMap(points, map_center, labels, zoom_start=10, marker_radius = 0.5):
    # Example latitude and longitude points
    # points = [(32.0853, 34.7818), (31.7683, 35.2137), (32.1093, 34.8555)]
    
    # Create a map centered around Tel Aviv
    # map_center = (32.0853, 34.7818)
    my_map = folium.Map(location=map_center, zoom_start=zoom_start)
    
    # Add markers for each point
    label_counter = 0
    for lat, lon in points:
        folium.CircleMarker(location=(lat, lon), radius=marker_radius, 
                            popup=labels[label_counter]).add_to(my_map)
        label_counter = label_counter + 1 
    
    
    # Save the map to an HTML file
    my_map.save("points_map.html")
    print("Map saved as points_map.html")

    return my_map

#%% Show the map
import webbrowser
import os
def showMap(map_filename = "points_map.html"):
    # Use the absolute path to the HTML file
    file_path = os.path.abspath(map_filename)
    # Opens the default browser
    webbrowser.get().open_new(f"file://{file_path}")
    
