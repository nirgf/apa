import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
from xml.etree import ElementTree as ET
import tqdm
import numpy as np
import os
from datetime import datetime

## Make plots interactive
import matplotlib
#matplotlib.use('TkAgg')

def parse_kml(kml_file = "Pavement_Condition.kml", save_csv = True):

    tree = ET.parse(kml_file)
    root = tree.getroot()

    # Namespace for KML
    namespace = {"kml": "http://www.opengis.net/kml/2.2"}

    # Initialize lists to hold data
    data = []

    # Find all Placemarks
    placemarks = root.findall(".//kml:Placemark", namespace)

    # Loop through Placemarks and extract data
    for placemark in tqdm.tqdm(placemarks):
        entry = {}
        extended_data = placemark.find("kml:ExtendedData/kml:SchemaData", namespace)

        # Extract SimpleData fields
        if extended_data is not None:
            for simple_data in extended_data.findall("kml:SimpleData", namespace):
                field_name = simple_data.get("name")
                field_value = simple_data.text
                entry[field_name] = field_value

        # Extract coordinates
        coordinates = placemark.find(".//kml:coordinates", namespace)
        if coordinates is not None:
            entry["coordinates"] = coordinates.text.strip()

        data.append(entry)

    # Convert to Pandas DataFrame
    df = pd.DataFrame(data)

    # Get Only Asphalt Roads. use only 2023 data
    filtered_df = df[(df['evalyear'] == '2023') & (df['surface'] == 'Asphalt')]

    fin_df = filtered_df[['cond', 'coordinates','evalyear']]
    point_df = split_coordinates_to_points(fin_df)
    if save_csv:
        point_df.to_csv("Detroit/Pavement_Condition.csv", index=False)
        # Display the DataFrame
        print('Parsed kml data and saved to cvs')

    # Extract all latitude and longitude values
    lat_ls = []
    lon_ls = []
    for geom in fin_df['coordinates']:
        geom_ls = geom.split(' ')
        for points in geom_ls:
            curr_lat = points.split(',')[0]
            curr_lon = points.split(',')[1]

            lat_ls += [float(curr_lat)]
            lon_ls += [float(curr_lon)]
    np.max(lat_ls)
    roi = [np.min(lat_ls), np.max(lat_ls), np.min(lon_ls), np.max(lon_ls)]
    return point_df, roi

### Plot on map ###
def plot_df_coords(fin_df):
    import geopandas as gpd
    from shapely.geometry import Point, MultiPoint

    # Step 1: Parse the coordinates column
    def parse_coordinates(coord_string):
        points = [
            Point(float(lon), float(lat))
            for lon, lat in (pair.split(",") for pair in coord_string.split())
        ]
        return MultiPoint(points) if len(points) > 1 else points[0]

    # Overwrite the 'geometry' column with zeros
    fin_df['geometry'] = None
    fin_df.loc[:, "geometry"] = fin_df["coordinates"].apply(parse_coordinates)

    # Step 2: Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(fin_df, geometry="geometry", crs="EPSG:4326")  # WGS 84
    gdf = gdf.to_crs(epsg=3857)

    # Step 3: Plot on map
    ax = gdf.iloc[0:2000].plot(figsize=(10, 8), color="blue", markersize=10)
    ax.set_title("Pavement Condition Map", fontsize=15)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)

    # Add OpenStreetMap basemap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # Show the plot
    plt.show()

def split_coordinates_to_points(df):
    # Split coordinates into individual points
    expanded_rows = []
    seg_num = 0
    for _, row in df.iterrows():
        seg_num = seg_num + 1
        condition_str = row["cond"]
        coordinate_pairs = row["coordinates"].split(" ")
        for pair in coordinate_pairs:
            if condition_str == 'Poor' : condition = 1
            elif condition_str == 'Fair' : condition = 2
            elif condition_str == 'Good' : condition = 3

            lon, lat = map(float, pair.split(","))
            evalyear = pd.Timestamp(row['evalyear'])
            expanded_rows.append({"PCI": condition, "latitude": lat, "longitude": lon,'S_Date':evalyear, 'seg_id': seg_num})

    # Create a new DataFrame
    expanded_df = pd.DataFrame(expanded_rows)
    return expanded_df


if __name__ == "__main__":
    # %% Get venus data
    print(__name__)
