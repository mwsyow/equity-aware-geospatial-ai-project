import geopandas as gpd
from shapely.geometry import Point
import os
from ..index_travel_accessibility.travel_time_and_centroid import (
    get_hospital_df,
)

def get_existing_hospitals_gdf(crs: str='EPSG:4326'):# Modifying the existing hospitals DataFrame to Prediction DataFrame format
    """
    Get existing hospitals data as a GeoDataFrame with standardized format.
    
    This function processes existing hospital data by:
    1. Loading hospital data and standardizing column names
    2. Creating geometry points from latitude/longitude coordinates
    3. Performing spatial join with NUTS level 3 regions to get district information
    4. Mapping district names to official Saarland district codes
    
    Args:
        crs (str, optional): Coordinate reference system to use. Defaults to 'EPSG:4326'.
            This is the standard WGS84 coordinate system used for GPS coordinates.
    
    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing existing hospitals with columns:
            - geometry: Point geometry of hospital location
            - district_code: Official Saarland district code (e.g., '10041' for Saarbrücken)
            - node: Hospital identifier
            - bed_allocation: Number of beds allocated to the hospital
            - Lat: Latitude coordinate (rounded to 6 decimal places)
            - Lon: Longitude coordinate (rounded to 6 decimal places)
    
    Note:
        The function assumes the existence of a NUTS level 3 shapefile in the
        index_travel_accessibility/data/raw directory for spatial joining.
    """
    existing_hospitals = get_hospital_df()

    existing_hospitals.rename(columns={
        'MaxBeds': 'bed_allocation',
        'SiteID': 'node'
    }, inplace=True)


    # Adding Geometry column to existing_hospitals
    existing_hospitals['Lat'] = existing_hospitals['Lat'].round(6)
    existing_hospitals['Lon'] = existing_hospitals['Lon'].round(6)
    existing_hospitals['geometry'] = existing_hospitals.apply(lambda row: Point(row['Lon'], row['Lat']), axis=1)

    # Getting district codes from the existing hospitals

    # Load Saarland NUTS level 3 shapefile
    nuts = gpd.read_file(os.path.join(os.path.dirname(__file__), '..', 'index_travel_accessibility', 'data', 'raw', 'NUTS_RG_01M_2021_4326_LEVL_3.shp'))
    nuts = nuts.to_crs(crs)


    # Create geometry column from Lat/Lon
    # existing_hospitals['geometry'] = existing_hospitals.apply(lambda row: Point(row['Lon'], row['Lat']), axis=1)
    # existing_hospitals = gpd.GeoDataFrame(existing_hospitals, geometry='geometry', crs='EPSG:4326')

    existing_hospitals = gpd.GeoDataFrame(existing_hospitals, geometry='geometry', crs=crs)

    # Spatial join hospitals with NUTS polygons (only needed columns)
    existing_hospitals = gpd.sjoin(
        existing_hospitals,
        nuts[['geometry', 'NUTS_NAME']],
        how='left',
        predicate='within'
    )

    # Map NUTS_NAME to your official Saarland district codes
    name_to_code = {
        "Regionalverband Saarbrücken": "10041",
        "Merzig-Wadern": "10042",
        "Neunkirchen": "10043",
        "Saarlouis": "10044",
        "Saarpfalz-Kreis": "10045",
        "St. Wendel": "10046"
    }

    # Assign district_code based on district_name
    existing_hospitals['district_code'] = existing_hospitals['NUTS_NAME'].map(name_to_code)

    # Rename NUTS_NAME to district_name
    existing_hospitals.rename(columns={'NUTS_NAME': 'district_name'}, inplace=True)

    # Select and reorder columns as needed
    existing_hospitals = existing_hospitals[['geometry', 'district_code', 'node', 'bed_allocation', 'Lat', 'Lon']]
    return existing_hospitals
