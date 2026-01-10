import streamlit as st
import folium
from streamlit_folium import st_folium
import os
# Set PROJ_LIB to the conda environment's PROJ data directory
os.environ['PROJ_LIB'] = r'C:\Users\mearg\.conda\envs\watershed_env\Library\share\proj'
from pysheds.grid import Grid
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import colors
from folium import plugins
import geopandas as gpd
from shapely.geometry import shape, mapping, LineString, Point, Polygon, box
import tempfile
from datetime import datetime
import pandas as pd
import io
import zipfile
from branca.colormap import linear, LinearColormap
import rasterio
from rasterio.warp import transform_bounds
import pyproj
from pyproj import Transformer

# --- 1. CONFIGURATION ---
st.set_page_config(
    layout="wide", 
    page_title="MEARGE WebGIS Watershed Delineation System",
    page_icon="🌊",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1e3c72;
    }
    .stButton>button {
        background-color: #1e3c72;
        color: white;
        border-radius: 5px;
        padding: 8px 16px;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

DEM_PATH = "dem.tif"  # Ensure this file exists in the same folder

# --- 2. SIDEBAR CONTROLS ---
st.sidebar.image("https://img.icons8.com/fluency/96/water.png", width=80)
st.sidebar.title("⚙️ Control Panel")

# Basemap Selection
st.sidebar.header("🗺️ Basemap")
basemap_option = st.sidebar.selectbox(
    "Select Basemap",
    ["OpenStreetMap", "Satellite", "Terrain", "Topo Map", "Dark Mode"]
)

basemap_dict = {
    "OpenStreetMap": "OpenStreetMap",
    "Satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "Terrain": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}",
    "Topo Map": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
    "Dark Mode": "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
}

st.sidebar.header("🎛️ Analysis Settings")
snap_threshold = st.sidebar.slider("Snap Threshold (Accumulation)", 100, 5000, 500, 
                                   help="Minimum accumulation value to snap to stream")
opacity = st.sidebar.slider("Layer Opacity", 0.0, 1.0, 0.6)

st.sidebar.header("👁️ Display Layers")
show_dem = st.sidebar.checkbox("DEM (Elevation)", value=False, help="Show elevation colors")
show_hillshade = st.sidebar.checkbox("Hillshade", value=False, help="Terrain shading for 3D effect")
show_streams = st.sidebar.checkbox("Stream Network", value=True)
show_stream_order = st.sidebar.checkbox("Stream Order (Strahler)", value=False, help="Color-coded stream hierarchy")
show_flow_direction = st.sidebar.checkbox("Flow Direction", value=False, help="D8 flow direction arrows")
show_accumulation = st.sidebar.checkbox("Flow Accumulation", value=False, help="Drainage area upstream")
show_slope = st.sidebar.checkbox("Slope Analysis", value=False)
show_watershed_boundary = st.sidebar.checkbox("Watershed Boundary Only", value=False, 
                                           help="Show only the red boundary without fill")

# Stream threshold for visibility
stream_threshold = st.sidebar.slider("Stream Visibility Threshold", 50, 1000, 200,
                                     help="Minimum accumulation to show as stream")

# Zoom to DEM button
if st.sidebar.button("🔍 Zoom to DEM Extent"):
    st.session_state.zoom_to_bounds = 'dem'

st.sidebar.header("🛠️ Tools")
enable_measurement = st.sidebar.checkbox("Measurement Tools", value=True)
enable_drawing = st.sidebar.checkbox("Drawing Tools", value=True)
show_coordinates = st.sidebar.checkbox("Mouse Coordinates", value=True)
enable_fullscreen = st.sidebar.checkbox("Fullscreen Button", value=True)

# --- VECTOR DATA MANAGEMENT SECTION ---
st.sidebar.header("📁 Vector Data")

# Initialize session state for vector layers
if 'vector_layers' not in st.session_state:
    st.session_state.vector_layers = {}
if 'pending_crs_layer' not in st.session_state:
    st.session_state.pending_crs_layer = None
if 'zoom_to_bounds' not in st.session_state:
    st.session_state.zoom_to_bounds = None

# Common CRS options for manual assignment
CRS_OPTIONS = {
    "Auto-detect": None,
    "WGS 84 (EPSG:4326)": "EPSG:4326",
    "UTM Zone 37N (EPSG:32637)": "EPSG:32637",
    "UTM Zone 37S (EPSG:32737)": "EPSG:32737",
    "UTM Zone 36N (EPSG:32636)": "EPSG:32636",
    "UTM Zone 36S (EPSG:32736)": "EPSG:32736",
    "UTM Zone 38N (EPSG:32638)": "EPSG:32638",
    "UTM Zone 38S (EPSG:32738)": "EPSG:32738",
    "Web Mercator (EPSG:3857)": "EPSG:3857",
    "Adindan UTM 37N (EPSG:20137)": "EPSG:20137",
    "Custom EPSG...": "custom"
}

# CRS selection for uploaded data
selected_crs_option = st.sidebar.selectbox(
    "Source CRS (if not auto-detected)",
    options=list(CRS_OPTIONS.keys()),
    index=0,
    help="Select the coordinate system of your data. Use 'Auto-detect' if your file has CRS info."
)

# Custom EPSG input
custom_epsg = None
if selected_crs_option == "Custom EPSG...":
    custom_epsg = st.sidebar.text_input(
        "Enter EPSG code",
        value="32637",
        help="Enter just the number, e.g., 32637 for UTM Zone 37N"
    )

# File uploader for vector data
uploaded_file = st.sidebar.file_uploader(
    "Upload Vector Data",
    type=['geojson', 'json', 'zip', 'gpkg', 'kml'],
    help="Supports GeoJSON, Shapefile (zipped), GeoPackage, KML. UTM and other projections are automatically converted to WGS84 for display."
)

def process_gdf_crs(gdf, layer_name, selected_crs_option, custom_epsg):
    """
    Process GeoDataFrame CRS - detect, assign if missing, and reproject to WGS84.
    Returns the processed GeoDataFrame and original CRS info.
    """
    original_crs = gdf.crs
    
    # If CRS is missing and user specified one
    if gdf.crs is None:
        if selected_crs_option == "Custom EPSG..." and custom_epsg:
            try:
                gdf = gdf.set_crs(f"EPSG:{custom_epsg}")
                st.sidebar.info(f"📍 Assigned CRS: EPSG:{custom_epsg}")
            except Exception as e:
                st.sidebar.error(f"Invalid EPSG code: {e}")
                return None, None
        elif CRS_OPTIONS.get(selected_crs_option):
            gdf = gdf.set_crs(CRS_OPTIONS[selected_crs_option])
            st.sidebar.info(f"📍 Assigned CRS: {CRS_OPTIONS[selected_crs_option]}")
        else:
            st.sidebar.warning(f"⚠️ No CRS detected for {layer_name}. Please select the source CRS and re-upload.")
            return None, None
    
    original_crs = gdf.crs
    
    # Reproject to WGS84 if needed
    if gdf.crs and str(gdf.crs) != 'EPSG:4326':
        try:
            gdf_wgs84 = gdf.to_crs('EPSG:4326')
            st.sidebar.success(f"🔄 Reprojected from {original_crs} to WGS84")
            return gdf_wgs84, original_crs
        except Exception as e:
            st.sidebar.error(f"Reprojection error: {e}")
            return None, None
    
    return gdf, original_crs

# Process uploaded vector file
if uploaded_file is not None:
    try:
        file_name = uploaded_file.name
        layer_name = os.path.splitext(file_name)[0]
        
        if file_name.endswith('.geojson') or file_name.endswith('.json'):
            # Read GeoJSON
            gdf = gpd.read_file(uploaded_file)
            gdf_processed, original_crs = process_gdf_crs(gdf, layer_name, selected_crs_option, custom_epsg)
            if gdf_processed is not None:
                st.session_state.vector_layers[layer_name] = {
                    'gdf': gdf_processed,
                    'original_crs': str(original_crs) if original_crs else 'WGS84',
                    'visible': True,
                    'color': '#3388ff',
                    'opacity': 0.7,
                    'weight': 2
                }
                st.sidebar.success(f"✅ Loaded: {layer_name}")
            
        elif file_name.endswith('.zip'):
            # Read zipped Shapefile
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                    zip_ref.extractall(tmpdir)
                # Find .shp file
                shp_files = [f for f in os.listdir(tmpdir) if f.endswith('.shp')]
                if shp_files:
                    gdf = gpd.read_file(os.path.join(tmpdir, shp_files[0]))
                    layer_name = os.path.splitext(shp_files[0])[0]
                    
                    # Check for .prj file
                    prj_files = [f for f in os.listdir(tmpdir) if f.endswith('.prj')]
                    if not prj_files and gdf.crs is None:
                        st.sidebar.warning(f"⚠️ No .prj file found. CRS unknown.")
                    
                    gdf_processed, original_crs = process_gdf_crs(gdf, layer_name, selected_crs_option, custom_epsg)
                    if gdf_processed is not None:
                        st.session_state.vector_layers[layer_name] = {
                            'gdf': gdf_processed,
                            'original_crs': str(original_crs) if original_crs else 'Unknown',
                            'visible': True,
                            'color': '#3388ff',
                            'opacity': 0.7,
                            'weight': 2
                        }
                        st.sidebar.success(f"✅ Loaded: {layer_name}")
                else:
                    st.sidebar.error("No .shp file found in zip")
                    
        elif file_name.endswith('.gpkg'):
            # Read GeoPackage
            with tempfile.NamedTemporaryFile(delete=False, suffix='.gpkg') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            gdf = gpd.read_file(tmp_path)
            os.unlink(tmp_path)
            gdf_processed, original_crs = process_gdf_crs(gdf, layer_name, selected_crs_option, custom_epsg)
            if gdf_processed is not None:
                st.session_state.vector_layers[layer_name] = {
                    'gdf': gdf_processed,
                    'original_crs': str(original_crs) if original_crs else 'WGS84',
                    'visible': True,
                    'color': '#3388ff',
                    'opacity': 0.7,
                    'weight': 2
                }
                st.sidebar.success(f"✅ Loaded: {layer_name}")
            
        elif file_name.endswith('.kml'):
            # Read KML (KML is always WGS84)
            gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
            with tempfile.NamedTemporaryFile(delete=False, suffix='.kml') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            gdf = gpd.read_file(tmp_path, driver='KML')
            os.unlink(tmp_path)
            # KML is always in WGS84
            if gdf.crs is None:
                gdf = gdf.set_crs('EPSG:4326')
            st.session_state.vector_layers[layer_name] = {
                'gdf': gdf,
                'original_crs': 'EPSG:4326 (KML)',
                'visible': True,
                'color': '#3388ff',
                'opacity': 0.7,
                'weight': 2
            }
            st.sidebar.success(f"✅ Loaded: {layer_name}")
            
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")

# Layer Management UI in sidebar
if st.session_state.vector_layers:
    st.sidebar.subheader("🗂️ Loaded Layers")
    layers_to_remove = []
    
    for layer_name, layer_data in st.session_state.vector_layers.items():
        with st.sidebar.expander(f"📍 {layer_name}", expanded=False):
            # Visibility toggle
            layer_data['visible'] = st.checkbox(
                "Show Layer", 
                value=layer_data['visible'],
                key=f"vis_{layer_name}"
            )
            
            # Color picker
            layer_data['color'] = st.color_picker(
                "Color", 
                value=layer_data['color'],
                key=f"color_{layer_name}"
            )
            
            # Opacity slider
            layer_data['opacity'] = st.slider(
                "Opacity", 0.0, 1.0, 
                value=layer_data['opacity'],
                key=f"opacity_{layer_name}"
            )
            
            # Line weight
            layer_data['weight'] = st.slider(
                "Line Weight", 1, 10, 
                value=layer_data['weight'],
                key=f"weight_{layer_name}"
            )
            
            # Layer info
            gdf = layer_data['gdf']
            st.caption(f"Features: {len(gdf)} | Type: {gdf.geom_type.iloc[0] if len(gdf) > 0 else 'N/A'}")
            original_crs = layer_data.get('original_crs', 'Unknown')
            st.caption(f"Original CRS: {original_crs}")
            st.caption(f"Display CRS: WGS84 (EPSG:4326)")
            
            # Zoom to Layer button
            if st.button(f"🔍 Zoom to Layer", key=f"zoom_{layer_name}"):
                # Get layer bounds
                layer_bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
                st.session_state.zoom_to_bounds = {
                    'min_x': layer_bounds[0],
                    'min_y': layer_bounds[1],
                    'max_x': layer_bounds[2],
                    'max_y': layer_bounds[3],
                    'layer_name': layer_name
                }
                st.rerun()
            
            # Remove button
            if st.button(f"🗑️ Remove", key=f"remove_{layer_name}"):
                layers_to_remove.append(layer_name)
    
    # Remove marked layers
    for layer_name in layers_to_remove:
        del st.session_state.vector_layers[layer_name]
        st.rerun()

# --- POUR POINT ANALYSIS SECTION ---
st.sidebar.header("📍 Pour Point Analysis")

# Initialize session state for pour point analysis
if 'pour_point_results' not in st.session_state:
    st.session_state.pour_point_results = None
if 'selected_pour_point_layer' not in st.session_state:
    st.session_state.selected_pour_point_layer = None

# Check for point layers in vector_layers
point_layers = {}
if st.session_state.vector_layers:
    for layer_name, layer_data in st.session_state.vector_layers.items():
        gdf = layer_data['gdf']
        if len(gdf) > 0:
            geom_type = gdf.geom_type.iloc[0].lower() if hasattr(gdf.geom_type.iloc[0], 'lower') else str(gdf.geom_type.iloc[0]).lower()
            if 'point' in geom_type:
                point_layers[layer_name] = layer_data

if point_layers:
    st.sidebar.success(f"✅ {len(point_layers)} point layer(s) available")
    
    # Select pour point layer
    selected_pour_layer = st.sidebar.selectbox(
        "Select Pour Point Layer",
        options=list(point_layers.keys()),
        key="pour_point_selector"
    )
    st.session_state.selected_pour_point_layer = selected_pour_layer
    
    if selected_pour_layer:
        pp_gdf = point_layers[selected_pour_layer]['gdf']
        st.sidebar.caption(f"📌 {len(pp_gdf)} pour point(s) found")
        
        # Button to run batch analysis
        if st.sidebar.button("🚀 Run Batch Watershed Analysis", type="primary"):
            st.session_state.run_batch_analysis = True
        else:
            st.session_state.run_batch_analysis = False
else:
    st.sidebar.info("📤 Upload a point shapefile with pour points to enable batch watershed analysis")
    st.session_state.run_batch_analysis = False

# Help Section
with st.sidebar.expander("ℹ️ Help & Documentation"):
    st.markdown("""
    **How to Use:**
    1. **Setup**: Ensure `dem.tif` is in the same folder
    2. **Configure**: Select basemap and enable desired layers
    3. **Analyze**: Click near a stream on the map
    4. **Results**: View watershed with red boundary and statistics
    5. **Export**: Download results in multiple formats
    
    **Layers:**
    - **Stream Network**: Blue overlay showing rivers
    - **Stream Order**: Hierarchical river classification
    - **Flow Accumulation**: Water flow intensity
    - **Slope**: Terrain gradient analysis
    
    **Tools:**
    - **Measurement**: Click to measure distances/areas
    - **Drawing**: Draw custom features on map
    """)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Click on the map near a stream to delineate the watershed automatically!")

# --- 3. BACKEND FUNCTIONS (Cached for Performance) ---

def add_vector_layer_to_map(m, layer_name, layer_data):
    """
    Add a vector layer to a Folium map with styling and popups.
    The GeoDataFrame is already in WGS84 from the loading process.
    """
    gdf = layer_data['gdf']
    color = layer_data['color']
    opacity = layer_data['opacity']
    weight = layer_data['weight']
    
    # Data should already be in WGS84, but double-check
    if gdf.crs is None:
        gdf = gdf.set_crs('EPSG:4326')
    elif str(gdf.crs) != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')
    
    # Create style function
    style_function = lambda x: {
        'fillColor': color,
        'color': color,
        'weight': weight,
        'fillOpacity': opacity * 0.5,
        'opacity': opacity
    }
    
    highlight_function = lambda x: {
        'fillColor': '#ffff00',
        'color': '#ffff00',
        'weight': weight + 2,
        'fillOpacity': 0.7
    }
    
    # Create GeoJson layer with popup showing all attributes
    def create_popup(feature):
        props = feature.get('properties', {})
        html = f"<div style='max-height: 200px; overflow-y: auto;'><b>{layer_name}</b><hr>"
        for key, value in props.items():
            if value is not None:
                html += f"<b>{key}:</b> {value}<br>"
        html += "</div>"
        return folium.Popup(html, max_width=300)
    
    geojson_layer = folium.GeoJson(
        gdf.to_json(),
        name=layer_name,
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=list(gdf.columns.drop('geometry'))[:5],  # Show first 5 fields
            aliases=[f"{col}:" for col in list(gdf.columns.drop('geometry'))[:5]],
            sticky=True
        ) if len(gdf.columns) > 1 else None
    )
    
    geojson_layer.add_to(m)
    return m

@st.cache_resource
def load_and_process_dem(path):
    """
    Loads DEM, fills pits (depressions), and calculates flow direction.
    Handles UTM and other projected coordinate systems by transforming to WGS84 for display.
    This runs ONLY ONCE when the app starts.
    """
    if not os.path.exists(path):
        return None, None, None, None, None, None, None, None

    # Get CRS information and data using rasterio FIRST
    with rasterio.open(path) as src:
        dem_crs = src.crs
        original_bounds = src.bounds  # left, bottom, right, top
        dem_transform = src.transform
        dem_data = src.read(1)
        dem_nodata = src.nodata
    
    # Load grid using pysheds - with error handling for CRS issues
    try:
        grid = Grid.from_raster(path)
        dem = grid.read_raster(path)
    except Exception as crs_error:
        # If pysheds has CRS issues, create grid from the rasterio data
        st.warning(f"CRS handling fallback activated. Using alternative loading method.")
        
        # Create a temporary file without ANY CRS
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_path = tmp.name
        
        # Write data WITHOUT CRS - pysheds will handle it
        with rasterio.open(tmp_path, 'w', 
                          driver='GTiff',
                          height=dem_data.shape[0],
                          width=dem_data.shape[1],
                          count=1,
                          dtype=dem_data.dtype,
                          transform=dem_transform,
                          nodata=dem_nodata) as dst:
            dst.write(dem_data, 1)
        
        # Now load with pysheds from the temp file
        grid = Grid.from_raster(tmp_path)
        dem = grid.read_raster(tmp_path)
        
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass
    
    # Transform bounds to WGS84 if needed
    if dem_crs is not None and dem_crs != 'EPSG:4326':
        try:
            # Transform bounds from source CRS to WGS84
            wgs84_bounds = transform_bounds(dem_crs, 'EPSG:4326', 
                                           original_bounds.left, 
                                           original_bounds.bottom, 
                                           original_bounds.right, 
                                           original_bounds.top)
            # wgs84_bounds is (left, bottom, right, top) = (min_x, min_y, max_x, max_y)
            bounds_wgs84 = {
                'min_x': wgs84_bounds[0],
                'min_y': wgs84_bounds[1],
                'max_x': wgs84_bounds[2],
                'max_y': wgs84_bounds[3]
            }
            
            # Create transformer for click coordinate conversion (WGS84 to source CRS)
            transformer_to_utm = Transformer.from_crs('EPSG:4326', dem_crs, always_xy=True)
            transformer_to_wgs84 = Transformer.from_crs(dem_crs, 'EPSG:4326', always_xy=True)
            
        except Exception as e:
            st.warning(f"CRS transformation warning: {e}. Using original bounds.")
            bounds_wgs84 = {
                'min_x': original_bounds.left,
                'min_y': original_bounds.bottom,
                'max_x': original_bounds.right,
                'max_y': original_bounds.top
            }
            transformer_to_utm = None
            transformer_to_wgs84 = None
    else:
        # Already in WGS84 or no CRS
        bounds_wgs84 = {
            'min_x': original_bounds.left,
            'min_y': original_bounds.bottom,
            'max_x': original_bounds.right,
            'max_y': original_bounds.top
        }
        transformer_to_utm = None
        transformer_to_wgs84 = None

    # Condition the DEM (Fill Pits/Sinks)
    pit_filled_dem = grid.fill_pits(dem)
    
    # Calculate Flow Direction
    fdir = grid.flowdir(pit_filled_dem)
    
    # Calculate Flow Accumulation
    acc = grid.accumulation(fdir)
    
    # Calculate Stream Order (Strahler method)
    stream_order = grid.stream_order(fdir, acc > 100)
    
    # Calculate Slope (in degrees)
    slope = np.degrees(np.arctan(np.gradient(pit_filled_dem)[0]))
    
    # Store CRS info for later use
    crs_info = {
        'original_crs': str(dem_crs) if dem_crs else 'Unknown',
        'is_projected': dem_crs.is_projected if dem_crs else False,
        'transformer_to_utm': transformer_to_utm,
        'transformer_to_wgs84': transformer_to_wgs84
    }
    
    return grid, pit_filled_dem, fdir, acc, stream_order, slope, bounds_wgs84, crs_info

# --- 4. MAIN APP LOGIC ---
st.markdown('<div class="main-header"><h1>🌊 Professional WebGIS Watershed Delineation System</h1><p>Advanced Hydrological Analysis & Watershed Management Tool(dev:Mearg k)</p></div>', unsafe_allow_html=True)

# Info boxes
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="info-box"><b>📍 Step 1:</b> Wait for DEM processing</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="info-box"><b>🖱️ Step 2:</b> Click on map near a stream</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="info-box"><b>📊 Step 3:</b> View & export results</div>', unsafe_allow_html=True)

# Load Data
with st.spinner("🔄 Initializing Hydrological Model (Loading DEM... this may take 10s)..."):
    grid, dem, fdir, acc, stream_order, slope, bounds_wgs84, crs_info = load_and_process_dem(DEM_PATH)

# Check if DEM exists
if grid is None:
    st.error(f"❌ File '{DEM_PATH}' not found! Please place a GeoTIFF named 'dem.tif' in this folder.")
    st.stop()

# Display DEM CRS info
if crs_info:
    with st.expander("🌍 DEM Coordinate System Info", expanded=False):
        st.write(f"**Original CRS:** {crs_info['original_crs']}")
        st.write(f"**Projected CRS:** {'Yes' if crs_info['is_projected'] else 'No (Geographic)'}")
        st.write(f"**Display CRS:** WGS84 (EPSG:4326)")
        if crs_info['is_projected']:
            st.info("✅ UTM/Projected coordinates detected. Automatically transformed to WGS84 for web display.")

# Get Map Center from WGS84 bounds
center_lat = (bounds_wgs84['min_y'] + bounds_wgs84['max_y']) / 2
center_lon = (bounds_wgs84['min_x'] + bounds_wgs84['max_x']) / 2

# --- 5. MAP INTERFACE ---
st.write("---")
st.subheader("🗺️ Interactive Map Viewer")

# Calculate the bounds in WGS84 for Folium [[south, west], [north, east]]
bounds = [[bounds_wgs84['min_y'], bounds_wgs84['min_x']], 
          [bounds_wgs84['max_y'], bounds_wgs84['max_x']]]

# Create the Base Map with selected basemap
if basemap_option == "OpenStreetMap":
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap")
elif basemap_option in ["Satellite", "Terrain", "Topo Map", "Dark Mode"]:
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles=basemap_dict[basemap_option],
                   attr='Esri' if basemap_option != "Dark Mode" else "CartoDB")

# --- IMPROVED VISUALIZATION LAYERS ---

# Helper function to convert pysheds Raster to numpy array
def to_numpy_array(raster_data):
    """Convert pysheds Raster or masked array to regular numpy array."""
    if hasattr(raster_data, 'data'):
        # It's a masked array or Raster object
        arr = np.array(raster_data)
    else:
        arr = raster_data
    
    # Handle masked arrays
    if isinstance(arr, np.ma.MaskedArray):
        arr = arr.filled(np.nan)
    
    return arr.astype(np.float64)

# Helper function to create a hillshade
def create_hillshade(elevation, azimuth=315, altitude=45):
    """Create hillshade from elevation data."""
    elevation = to_numpy_array(elevation)
    x, y = np.gradient(elevation)
    azimuth_rad = np.radians(azimuth)
    altitude_rad = np.radians(altitude)
    slope_rad = np.arctan(np.sqrt(x*x + y*y))
    aspect_rad = np.arctan2(-x, y)
    shaded = np.sin(altitude_rad) * np.cos(slope_rad) + \
             np.cos(altitude_rad) * np.sin(slope_rad) * \
             np.cos(azimuth_rad - aspect_rad)
    return np.clip(shaded, 0, 1)

# Convert all raster data to numpy arrays
dem_arr = to_numpy_array(dem)
acc_arr = to_numpy_array(acc)
fdir_arr = to_numpy_array(fdir)
stream_order_arr = to_numpy_array(stream_order)
slope_arr = to_numpy_array(slope)

# Debug info in expander
with st.expander("🔍 Debug: Raster Statistics", expanded=False):
    st.write(f"**DEM:** min={np.nanmin(dem_arr):.1f}, max={np.nanmax(dem_arr):.1f}, shape={dem_arr.shape}")
    st.write(f"**Flow Accumulation:** min={np.nanmin(acc_arr):.0f}, max={np.nanmax(acc_arr):.0f}")
    st.write(f"**Stream Order:** min={np.nanmin(stream_order_arr[stream_order_arr > 0]) if np.any(stream_order_arr > 0) else 0}, max={np.nanmax(stream_order_arr):.0f}")
    st.write(f"**Slope:** min={np.nanmin(slope_arr):.1f}°, max={np.nanmax(slope_arr):.1f}°")
    st.write(f"**Bounds (WGS84):** {bounds}")

# Add DEM (Elevation) Layer
if show_dem:
    # Set visualization thresholds (max 3000m)
    MAX_ELEVATION = 3000.0
    dem_min = np.nanmin(dem_arr)
    dem_max_raw = np.nanmax(dem_arr)
    dem_max = min(dem_max_raw, MAX_ELEVATION)  # Cap at 3000m
    
    st.sidebar.caption(f"DEM Range: {dem_min:.0f}m - {dem_max_raw:.0f}m (capped at {MAX_ELEVATION:.0f}m)")
    
    # Clip DEM values to threshold and normalize
    dem_clipped = np.clip(dem_arr, dem_min, MAX_ELEVATION)
    dem_range = dem_max - dem_min
    
    if dem_range > 0:
        dem_norm = (dem_clipped - dem_min) / dem_range
    else:
        dem_norm = np.zeros_like(dem_arr)
    
    # Use matplotlib colormap for proper terrain colors (vectorized - much faster)
    from matplotlib.colors import LinearSegmentedColormap
    
    # Define terrain colormap: green -> yellow -> brown -> white
    terrain_colors = [
        (0.0, '#1a472a'),   # Dark green (low)
        (0.15, '#2d6a4f'),  # Forest green
        (0.3, '#52b788'),   # Light green
        (0.45, '#b7e4c7'),  # Pale green
        (0.55, '#f4e285'),  # Yellow
        (0.7, '#dda15e'),   # Tan/orange
        (0.85, '#bc6c25'),  # Brown
        (1.0, '#ffffff')    # White (high)
    ]
    
    # Create colormap
    colors_list = [c[1] for c in terrain_colors]
    positions = [c[0] for c in terrain_colors]
    cmap_terrain = LinearSegmentedColormap.from_list('terrain_custom', list(zip(positions, colors_list)))
    
    # Apply colormap (returns RGBA)
    dem_colored = cmap_terrain(dem_norm)
    
    # Set nodata to transparent
    dem_colored[np.isnan(dem_arr), 3] = 0
    
    # Ensure float32 for folium
    dem_colored = dem_colored.astype(np.float32)
    
    folium.raster_layers.ImageOverlay(
        image=dem_colored,
        bounds=bounds,
        opacity=opacity,
        name="⛰️ DEM Elevation"
    ).add_to(m)
    
    # Add elevation legend
    legend_html = f'''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white;
                padding: 10px; border-radius: 5px; border: 2px solid #333; font-family: Arial; font-size: 12px;">
        <b>⛰️ Elevation</b><br>
        <div style="display: flex; align-items: center; margin-top: 5px;">
            <div style="width: 150px; height: 20px; background: linear-gradient(to right, #1a472a, #2d6a4f, #52b788, #f4e285, #dda15e, #bc6c25, #ffffff);"></div>
        </div>
        <div style="display: flex; justify-content: space-between; width: 150px; font-size: 10px;">
            <span>{dem_min:.0f}m</span><span>{(dem_min+dem_max)/2:.0f}m</span><span>{dem_max:.0f}m</span>
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

# Add Hillshade Layer
if show_hillshade:
    hillshade = create_hillshade(dem_arr)
    hillshade_colored = np.zeros((*hillshade.shape, 4), dtype=np.float32)
    hillshade_colored[:,:,0] = hillshade
    hillshade_colored[:,:,1] = hillshade
    hillshade_colored[:,:,2] = hillshade
    hillshade_colored[:,:,3] = np.where(np.isnan(dem_arr), 0, 0.7)
    
    folium.raster_layers.ImageOverlay(
        image=hillshade_colored,
        bounds=bounds,
        opacity=0.8,
        name="🌄 Hillshade"
    ).add_to(m)

# Add Stream Network Layer
if show_streams:
    # Create stream mask based on threshold - use lower threshold for visibility
    effective_threshold = max(stream_threshold, 50)  # At least 50
    stream_mask = acc_arr > effective_threshold
    
    st.sidebar.caption(f"Streams visible: {np.sum(stream_mask):,} cells")
    
    if np.any(stream_mask):
        # Logarithmic scaling for better visualization
        acc_log = np.log10(np.maximum(acc_arr, 1))
        acc_max_log = np.log10(np.nanmax(acc_arr) + 1)
        
        if acc_max_log > 0:
            acc_norm = np.clip(acc_log / acc_max_log, 0, 1)
        else:
            acc_norm = np.zeros_like(acc_arr)
        
        # Create BRIGHT BLUE stream network
        stream_colored = np.zeros((*acc_arr.shape, 4), dtype=np.float32)
        
        # Only color where streams exist
        stream_colored[:,:,0] = 0.0                    # R = 0
        stream_colored[:,:,1] = 0.3 * acc_norm         # G - slight green tint
        stream_colored[:,:,2] = 1.0                    # B = full blue
        stream_colored[:,:,3] = np.where(stream_mask, 1.0, 0.0)  # Full opacity where streams exist
        
        folium.raster_layers.ImageOverlay(
            image=stream_colored,
            bounds=bounds,
            opacity=1.0,
            name="🌊 Stream Network"
        ).add_to(m)
    else:
        st.sidebar.warning("No streams found with current threshold")

# Add Stream Order Layer (Strahler)
if show_stream_order:
    valid_orders = stream_order_arr[stream_order_arr > 0]
    
    if len(valid_orders) > 0:
        max_order = int(np.nanmax(stream_order_arr))
        
        # Distinct colors for each stream order
        order_colors = {
            1: [0.5, 0.8, 1.0],    # Light cyan
            2: [0.2, 0.6, 1.0],    # Light blue
            3: [0.0, 0.4, 1.0],    # Blue
            4: [0.0, 0.2, 0.8],    # Dark blue
            5: [0.4, 0.0, 0.8],    # Purple
            6: [0.8, 0.0, 0.6],    # Magenta
            7: [1.0, 0.0, 0.0],    # Red
        }
        
        order_colored = np.zeros((*stream_order_arr.shape, 4), dtype=np.float32)
        
        for order_val in range(1, max_order + 1):
            mask = stream_order_arr == order_val
            color_key = min(order_val, 7)
            color = order_colors.get(color_key, [1.0, 0.0, 0.0])
            order_colored[mask, 0] = color[0]
            order_colored[mask, 1] = color[1]
            order_colored[mask, 2] = color[2]
            order_colored[mask, 3] = 0.9
        
        # Transparent where no streams
        order_colored[stream_order_arr <= 0, 3] = 0
        
        folium.raster_layers.ImageOverlay(
            image=order_colored,
            bounds=bounds,
            opacity=1.0,
            name="📊 Stream Order"
        ).add_to(m)
        
        # Legend
        order_legend = '''
        <div style="position: fixed; bottom: 50px; right: 50px; z-index: 1000; background-color: white;
                    padding: 10px; border-radius: 5px; border: 2px solid #333; font-family: Arial; font-size: 11px;">
            <b>📊 Stream Order</b><br>
            <div style="margin-top: 5px;">
                <span style="display:inline-block;width:20px;height:12px;background:rgb(128,204,255);border:1px solid #333;"></span> 1 - Headwater<br>
                <span style="display:inline-block;width:20px;height:12px;background:rgb(51,153,255);border:1px solid #333;"></span> 2<br>
                <span style="display:inline-block;width:20px;height:12px;background:rgb(0,102,255);border:1px solid #333;"></span> 3<br>
                <span style="display:inline-block;width:20px;height:12px;background:rgb(0,51,204);border:1px solid #333;"></span> 4<br>
                <span style="display:inline-block;width:20px;height:12px;background:rgb(102,0,204);border:1px solid #333;"></span> 5<br>
                <span style="display:inline-block;width:20px;height:12px;background:rgb(204,0,153);border:1px solid #333;"></span> 6<br>
                <span style="display:inline-block;width:20px;height:12px;background:rgb(255,0,0);border:1px solid #333;"></span> 7+ - Main River<br>
            </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(order_legend))

# Add Flow Direction Layer
if show_flow_direction:
    # D8 flow direction: 1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE
    dir_colors = {
        1: [0.0, 1.0, 1.0],     # E - Cyan
        2: [0.0, 0.5, 1.0],     # SE - Blue-cyan
        4: [0.0, 0.0, 1.0],     # S - Blue
        8: [0.5, 0.0, 1.0],     # SW - Purple
        16: [1.0, 0.0, 1.0],    # W - Magenta
        32: [1.0, 0.0, 0.5],    # NW - Pink
        64: [1.0, 0.0, 0.0],    # N - Red
        128: [1.0, 0.5, 0.0],   # NE - Orange
    }
    
    fdir_colored = np.zeros((*fdir_arr.shape, 4), dtype=np.float32)
    
    for dir_val, color in dir_colors.items():
        mask = fdir_arr == dir_val
        fdir_colored[mask, 0] = color[0]
        fdir_colored[mask, 1] = color[1]
        fdir_colored[mask, 2] = color[2]
        fdir_colored[mask, 3] = 0.6
    
    # Make no-data transparent
    fdir_colored[fdir_arr <= 0, 3] = 0
    fdir_colored[np.isnan(fdir_arr), 3] = 0
    
    folium.raster_layers.ImageOverlay(
        image=fdir_colored,
        bounds=bounds,
        opacity=opacity,
        name="➡️ Flow Direction"
    ).add_to(m)
    
    # Legend
    fdir_legend = '''
    <div style="position: fixed; bottom: 180px; right: 50px; z-index: 1000; background-color: white;
                padding: 10px; border-radius: 5px; border: 2px solid #333; font-family: Arial; font-size: 11px;">
        <b>➡️ Flow Direction</b><br>
        <div style="margin-top: 5px;">
            <span style="display:inline-block;width:20px;height:12px;background:rgb(255,0,0);"></span> N ↑<br>
            <span style="display:inline-block;width:20px;height:12px;background:rgb(255,128,0);"></span> NE ↗<br>
            <span style="display:inline-block;width:20px;height:12px;background:rgb(0,255,255);"></span> E →<br>
            <span style="display:inline-block;width:20px;height:12px;background:rgb(0,128,255);"></span> SE ↘<br>
            <span style="display:inline-block;width:20px;height:12px;background:rgb(0,0,255);"></span> S ↓<br>
            <span style="display:inline-block;width:20px;height:12px;background:rgb(128,0,255);"></span> SW ↙<br>
            <span style="display:inline-block;width:20px;height:12px;background:rgb(255,0,255);"></span> W ←<br>
            <span style="display:inline-block;width:20px;height:12px;background:rgb(255,0,128);"></span> NW ↖<br>
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(fdir_legend))

# Add Flow Accumulation Layer
if show_accumulation:
    # Use log scale for better visualization
    acc_log = np.where(acc_arr > 1, np.log10(acc_arr), 0)
    acc_max_log = np.nanmax(acc_log)
    
    if acc_max_log > 0:
        acc_norm = acc_log / acc_max_log
    else:
        acc_norm = np.zeros_like(acc_log)
    
    # Yellow -> Orange -> Red -> Dark Red colormap
    acc_colored = np.zeros((*acc_arr.shape, 4), dtype=np.float32)
    
    # Apply color gradient
    acc_colored[:,:,0] = np.clip(1.0, 0, 1)  # R stays high
    acc_colored[:,:,1] = np.clip(1.0 - acc_norm * 1.2, 0, 1)  # G decreases
    acc_colored[:,:,2] = np.clip(0.2 - acc_norm * 0.2, 0, 1)  # B low
    
    # Alpha based on accumulation (low acc = transparent)
    acc_colored[:,:,3] = np.where(acc_arr > 10, 
                                   np.clip(0.2 + acc_norm * 0.8, 0, 1), 
                                   0)
    
    folium.raster_layers.ImageOverlay(
        image=acc_colored,
        bounds=bounds,
        opacity=1.0,
        name="💧 Flow Accumulation"
    ).add_to(m)
    
    # Legend
    max_acc_val = int(np.nanmax(acc_arr))
    acc_legend = f'''
    <div style="position: fixed; bottom: 130px; left: 50px; z-index: 1000; background-color: white;
                padding: 10px; border-radius: 5px; border: 2px solid #333; font-family: Arial; font-size: 12px;">
        <b>💧 Flow Accumulation</b><br>
        <div style="display: flex; align-items: center; margin-top: 5px;">
            <div style="width: 150px; height: 20px; background: linear-gradient(to right, #ffff66, #ffcc00, #ff6600, #cc0000, #660000);"></div>
        </div>
        <div style="display: flex; justify-content: space-between; width: 150px; font-size: 10px;">
            <span>Low</span><span>High ({max_acc_val:,})</span>
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(acc_legend))

# Add Slope Layer
if show_slope:
    slope_abs = np.abs(slope_arr)
    slope_max = np.nanmax(slope_abs)
    slope_clipped = np.clip(slope_abs, 0, 60)  # Clip to 60 degrees
    
    if slope_max > 0:
        slope_norm = slope_clipped / 60.0  # Normalize to 0-60 degree range
    else:
        slope_norm = np.zeros_like(slope_arr)
    
    # Green (flat) -> Yellow -> Orange -> Red (steep) colormap
    slope_colored = np.zeros((*slope_arr.shape, 4), dtype=np.float32)
    
    # Color gradient: green -> yellow -> red
    slope_colored[:,:,0] = np.clip(slope_norm * 2, 0, 1)  # R increases
    slope_colored[:,:,1] = np.clip(1.0 - slope_norm, 0, 1)  # G decreases from 1
    slope_colored[:,:,2] = 0.1  # B stays low
    
    # Full opacity except NaN
    slope_colored[:,:,3] = np.where(np.isnan(slope_arr), 0, 0.7)
    
    folium.raster_layers.ImageOverlay(
        image=slope_colored,
        bounds=bounds,
        opacity=opacity,
        name="📐 Slope"
    ).add_to(m)
    
    # Legend
    slope_legend = f'''
    <div style="position: fixed; bottom: 210px; left: 50px; z-index: 1000; background-color: white;
                padding: 10px; border-radius: 5px; border: 2px solid #333; font-family: Arial; font-size: 12px;">
        <b>📐 Slope (degrees)</b><br>
        <div style="display: flex; align-items: center; margin-top: 5px;">
            <div style="width: 150px; height: 20px; background: linear-gradient(to right, #00cc00, #66cc00, #cccc00, #ff9900, #ff0000);"></div>
        </div>
        <div style="display: flex; justify-content: space-between; width: 150px; font-size: 10px;">
            <span>0° Flat</span><span>30°</span><span>60°+ Steep</span>
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(slope_legend))

# Add Measurement Tools
if enable_measurement:
    plugins.MeasureControl(
        position='topleft',
        primary_length_unit='meters',
        secondary_length_unit='kilometers',
        primary_area_unit='sqmeters',
        secondary_area_unit='sqkilometers'
    ).add_to(m)

# Add Drawing Tools
if enable_drawing:
    draw = plugins.Draw(
        export=True,
        filename='drawn_features.geojson',
        position='topleft',
        draw_options={
            'polyline': {'shapeOptions': {'color': '#ff0000', 'weight': 3}},
            'polygon': {'shapeOptions': {'color': '#0000ff', 'weight': 3}},
            'circle': True,
            'rectangle': True,
            'marker': True,
            'circlemarker': False
        },
        edit_options={'edit': True, 'remove': True}
    )
    draw.add_to(m)

# Add Mouse Position Plugin
if show_coordinates:
    plugins.MousePosition(
        position='bottomright',
        separator=' | ',
        prefix='Coordinates: ',
        lat_formatter="function(num) {return L.Util.formatNum(num, 5) + ' °N';}",
        lng_formatter="function(num) {return L.Util.formatNum(num, 5) + ' °E';}"
    ).add_to(m)

# Add Fullscreen Button
if enable_fullscreen:
    plugins.Fullscreen(
        position='topright',
        title='Fullscreen',
        title_cancel='Exit Fullscreen',
        force_separate_button=True
    ).add_to(m)

# Add uploaded vector layers to map
for layer_name, layer_data in st.session_state.vector_layers.items():
    if layer_data['visible']:
        try:
            m = add_vector_layer_to_map(m, layer_name, layer_data)
        except Exception as e:
            st.warning(f"Could not add layer {layer_name}: {e}")

# Add batch watershed polygons from pour point analysis
if st.session_state.get('watershed_polygons') and len(st.session_state.watershed_polygons) > 0:
    ws_group = folium.FeatureGroup(name="📍 Batch Watersheds", show=True)
    
    # Color palette for multiple watersheds
    watershed_colors = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#FFFF00', '#00FFFF', 
                       '#FFA500', '#800080', '#008000', '#FF1493']
    
    for i, ws_data in enumerate(st.session_state.watershed_polygons):
        color = watershed_colors[i % len(watershed_colors)]
        ws_geom = ws_data['geometry']
        ws_json = mapping(ws_geom)
        
        folium.GeoJson(
            ws_json,
            style_function=lambda x, color=color: {
                'fillColor': color,
                'color': '#000000',
                'weight': 3,
                'fillOpacity': 0.3,
                'opacity': 1.0
            },
            popup=folium.Popup(
                f"<b>Watershed {ws_data['point_id']}</b><br>"
                f"Area: {ws_data['area_km2']:.2f} km²",
                max_width=200
            )
        ).add_to(ws_group)
    
    ws_group.add_to(m)

# Add MiniMap
plugins.MiniMap(toggle_display=True).add_to(m)

# Add Layer Control
folium.LayerControl(position='topright', collapsed=False).add_to(m)

# Apply zoom to layer bounds if requested
if st.session_state.zoom_to_bounds:
    zoom_bounds = st.session_state.zoom_to_bounds
    
    if zoom_bounds == 'dem':
        # Zoom to DEM extent
        m.fit_bounds([
            [bounds_wgs84['min_y'], bounds_wgs84['min_x']],
            [bounds_wgs84['max_y'], bounds_wgs84['max_x']]
        ])
        st.info("🔍 Zoomed to DEM extent")
    elif isinstance(zoom_bounds, dict):
        # Zoom to vector layer bounds
        m.fit_bounds([
            [zoom_bounds['min_y'], zoom_bounds['min_x']],
            [zoom_bounds['max_y'], zoom_bounds['max_x']]
        ])
        st.info(f"🔍 Zoomed to layer: {zoom_bounds['layer_name']}")
    
    # Clear the zoom state after applying
    st.session_state.zoom_to_bounds = None

# Add Click Handling - Larger map view
output = st_folium(m, width=None, height=850, use_container_width=True, key="main_map")

# --- 5.5 ATTRIBUTE TABLE VIEWER ---
if st.session_state.vector_layers:
    st.write("---")
    st.subheader("📋 Attribute Table Viewer")
    
    # Layer selector for attribute table
    layer_names = list(st.session_state.vector_layers.keys())
    selected_layer = st.selectbox(
        "Select Layer to View Attributes",
        options=layer_names,
        key="attr_table_layer"
    )
    
    if selected_layer:
        gdf = st.session_state.vector_layers[selected_layer]['gdf']
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Table", "📈 Statistics", "🔍 Filter & Query", "💾 Export"])
        
        with tab1:
            # Display attribute table without geometry column
            df_display = gdf.drop(columns=['geometry']).copy()
            
            # Pagination
            page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=0)
            total_pages = max(1, len(df_display) // page_size + (1 if len(df_display) % page_size > 0 else 0))
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
            
            start_idx = (page - 1) * page_size
            end_idx = min(start_idx + page_size, len(df_display))
            
            st.dataframe(
                df_display.iloc[start_idx:end_idx],
                use_container_width=True,
                height=400
            )
            st.caption(f"Showing {start_idx + 1} to {end_idx} of {len(df_display)} features")
        
        with tab2:
            # Statistics for numeric columns
            numeric_cols = gdf.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.markdown("**Numeric Column Statistics:**")
                stats_df = gdf[numeric_cols].describe()
                st.dataframe(stats_df, use_container_width=True)
                
                # Column selector for histogram
                if len(numeric_cols) > 0:
                    selected_col = st.selectbox("Select column for histogram", numeric_cols)
                    if selected_col:
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(10, 4))
                        gdf[selected_col].hist(ax=ax, bins=20, color='#1e3c72', edgecolor='white')
                        ax.set_xlabel(selected_col)
                        ax.set_ylabel('Frequency')
                        ax.set_title(f'Distribution of {selected_col}')
                        st.pyplot(fig)
            else:
                st.info("No numeric columns available for statistics.")
            
            # Geometry info
            st.markdown("**Geometry Information:**")
            geom_info = {
                'Total Features': len(gdf),
                'Geometry Type': gdf.geom_type.unique().tolist(),
                'CRS': str(gdf.crs),
                'Bounds': gdf.total_bounds.tolist()
            }
            for key, value in geom_info.items():
                st.write(f"**{key}:** {value}")
        
        with tab3:
            st.markdown("**Filter Data:**")
            
            # Column selector for filtering
            filter_col = st.selectbox(
                "Select column to filter",
                options=['-- Select --'] + list(gdf.drop(columns=['geometry']).columns)
            )
            
            if filter_col != '-- Select --':
                col_dtype = gdf[filter_col].dtype
                
                if pd.api.types.is_numeric_dtype(col_dtype):
                    # Numeric filter
                    min_val = float(gdf[filter_col].min())
                    max_val = float(gdf[filter_col].max())
                    filter_range = st.slider(
                        f"Filter {filter_col} range",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val)
                    )
                    filtered_gdf = gdf[(gdf[filter_col] >= filter_range[0]) & (gdf[filter_col] <= filter_range[1])]
                else:
                    # Categorical filter
                    unique_vals = gdf[filter_col].dropna().unique().tolist()
                    selected_vals = st.multiselect(
                        f"Select {filter_col} values",
                        options=unique_vals,
                        default=unique_vals[:5] if len(unique_vals) > 5 else unique_vals
                    )
                    filtered_gdf = gdf[gdf[filter_col].isin(selected_vals)]
                
                st.write(f"**Filtered Results:** {len(filtered_gdf)} features")
                st.dataframe(
                    filtered_gdf.drop(columns=['geometry']),
                    use_container_width=True,
                    height=300
                )
                
                # Option to save filtered data as new layer
                if st.button("➕ Add Filtered as New Layer"):
                    new_layer_name = f"{selected_layer}_filtered"
                    st.session_state.vector_layers[new_layer_name] = {
                        'gdf': filtered_gdf.copy(),
                        'visible': True,
                        'color': '#ff6b6b',
                        'opacity': 0.7,
                        'weight': 2
                    }
                    st.success(f"Created new layer: {new_layer_name}")
                    st.rerun()
        
        with tab4:
            st.markdown("**Export Layer Data:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Export as GeoJSON
                geojson_str = gdf.to_json()
                st.download_button(
                    label="📥 Download GeoJSON",
                    data=geojson_str,
                    file_name=f"{selected_layer}.geojson",
                    mime="application/json"
                )
            
            with col2:
                # Export as CSV (without geometry)
                csv_data = gdf.drop(columns=['geometry']).to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"{selected_layer}.csv",
                    mime="text/csv"
                )
            
            with col3:
                # Export as Shapefile
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        shp_path = os.path.join(tmpdir, f'{selected_layer}.shp')
                        gdf.to_file(shp_path)
                        
                        zip_path = os.path.join(tmpdir, f'{selected_layer}.zip')
                        with zipfile.ZipFile(zip_path, 'w') as zipf:
                            for file in os.listdir(tmpdir):
                                if file.startswith(selected_layer) and not file.endswith('.zip'):
                                    zipf.write(os.path.join(tmpdir, file), file)
                        
                        with open(zip_path, 'rb') as f:
                            zip_data = f.read()
                        
                        st.download_button(
                            label="📥 Download Shapefile",
                            data=zip_data,
                            file_name=f"{selected_layer}.zip",
                            mime="application/zip"
                        )
                except Exception as e:
                    st.error(f"Shapefile export error: {e}")

# --- 6A. BATCH POUR POINT WATERSHED ANALYSIS ---
if st.session_state.get('run_batch_analysis', False) and st.session_state.get('selected_pour_point_layer'):
    st.markdown("---")
    st.markdown("## 📍 Batch Pour Point Watershed Analysis")
    
    pour_layer_name = st.session_state.selected_pour_point_layer
    if pour_layer_name in st.session_state.vector_layers:
        pp_gdf = st.session_state.vector_layers[pour_layer_name]['gdf'].copy()
        
        # Ensure WGS84 for display but we'll convert to DEM CRS for analysis
        if pp_gdf.crs is None:
            pp_gdf = pp_gdf.set_crs('EPSG:4326')
        elif str(pp_gdf.crs) != 'EPSG:4326':
            pp_gdf = pp_gdf.to_crs('EPSG:4326')
        
        st.info(f"🔄 Processing {len(pp_gdf)} pour point(s)...")
        
        # Results storage
        results = []
        watershed_polygons = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, row in pp_gdf.iterrows():
            progress = (idx + 1) / len(pp_gdf)
            progress_bar.progress(progress)
            status_text.text(f"Processing point {idx + 1} of {len(pp_gdf)}...")
            
            try:
                # Get point coordinates (already in WGS84)
                point_geom = row.geometry
                pp_lat = point_geom.y
                pp_lng = point_geom.x
                
                # Convert to DEM CRS if needed
                if crs_info and crs_info['transformer_to_utm'] is not None:
                    pp_x, pp_y = crs_info['transformer_to_utm'].transform(pp_lng, pp_lat)
                else:
                    pp_x, pp_y = pp_lng, pp_lat
                
                # Snap to stream
                try:
                    x_snap, y_snap = grid.snap_to_mask(acc > snap_threshold, (pp_x, pp_y))
                except Exception:
                    # If snap fails, use original coordinates
                    x_snap, y_snap = pp_x, pp_y
                
                # Get flow accumulation at pour point
                try:
                    # Find nearest grid cell for accumulation value
                    row_idx = int((grid.extent[3] - y_snap) / grid.cellsize)
                    col_idx = int((x_snap - grid.extent[0]) / grid.cellsize)
                    
                    row_idx = max(0, min(row_idx, acc_arr.shape[0] - 1))
                    col_idx = max(0, min(col_idx, acc_arr.shape[1] - 1))
                    
                    flow_acc_value = float(acc_arr[row_idx, col_idx])
                except Exception:
                    flow_acc_value = None
                
                # Delineate catchment
                catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, xytype='coordinate')
                
                # Clip and polygonize
                grid.clip_to(catch)
                shapes_gen = grid.polygonize()
                
                watershed_geom = None
                watershed_geom_wgs84 = None
                watershed_area_km2 = 0
                
                for shape_geom, value in shapes_gen:
                    watershed_geom = shape(shape_geom)
                    
                    # Transform to WGS84 if needed
                    if crs_info and crs_info['transformer_to_wgs84'] is not None:
                        from shapely.ops import transform as shapely_transform
                        watershed_geom_wgs84 = shapely_transform(
                            crs_info['transformer_to_wgs84'].transform, 
                            watershed_geom
                        )
                        watershed_area_km2 = watershed_geom.area / 1_000_000
                    else:
                        watershed_geom_wgs84 = watershed_geom
                        geod = pyproj.Geod(ellps='WGS84')
                        area_m2 = abs(geod.geometry_area_perimeter(watershed_geom)[0])
                        watershed_area_km2 = area_m2 / 1_000_000
                    break
                
                # Get watershed statistics
                if watershed_geom is not None:
                    catch_dem_vals = dem[catch]
                    min_elev = float(np.nanmin(catch_dem_vals))
                    max_elev = float(np.nanmax(catch_dem_vals))
                    mean_elev = float(np.nanmean(catch_dem_vals))
                    
                    catch_acc_vals = acc[catch]
                    max_acc = float(np.nanmax(catch_acc_vals)) if np.any(catch_acc_vals > 0) else 0
                    
                    # Store result
                    result = {
                        'point_id': idx,
                        'latitude': pp_lat,
                        'longitude': pp_lng,
                        'flow_accumulation': flow_acc_value,
                        'area_km2': watershed_area_km2,
                        'min_elevation_m': min_elev,
                        'max_elevation_m': max_elev,
                        'mean_elevation_m': mean_elev,
                        'max_accumulation': max_acc,
                        'status': 'Success'
                    }
                    
                    # Copy original attributes
                    for col in pp_gdf.columns:
                        if col != 'geometry' and col not in result:
                            result[col] = row[col]
                    
                    results.append(result)
                    
                    if watershed_geom_wgs84 is not None:
                        watershed_polygons.append({
                            'point_id': idx,
                            'geometry': watershed_geom_wgs84,
                            'area_km2': watershed_area_km2
                        })
                
            except Exception as e:
                results.append({
                    'point_id': idx,
                    'latitude': point_geom.y if 'point_geom' in dir() else None,
                    'longitude': point_geom.x if 'point_geom' in dir() else None,
                    'flow_accumulation': None,
                    'area_km2': None,
                    'status': f'Error: {str(e)}'
                })
        
        progress_bar.progress(1.0)
        status_text.text("✅ Batch analysis complete!")
        
        # Store results in session state
        st.session_state.pour_point_results = results
        st.session_state.watershed_polygons = watershed_polygons
        
        # Display results
        if results:
            results_df = pd.DataFrame(results)
            
            st.success(f"✅ Successfully processed {len([r for r in results if r['status'] == 'Success'])} of {len(results)} pour points")
            
            # Summary statistics
            successful_results = [r for r in results if r['status'] == 'Success']
            if successful_results:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total_area = sum([r['area_km2'] for r in successful_results if r['area_km2']])
                    st.metric("Total Watershed Area", f"{total_area:.2f} km²")
                with col2:
                    avg_acc = np.mean([r['flow_accumulation'] for r in successful_results if r['flow_accumulation']])
                    st.metric("Avg Flow Accumulation", f"{avg_acc:,.0f}")
                with col3:
                    max_acc_all = max([r['flow_accumulation'] for r in successful_results if r['flow_accumulation']])
                    st.metric("Max Flow Accumulation", f"{max_acc_all:,.0f}")
                with col4:
                    avg_area = np.mean([r['area_km2'] for r in successful_results if r['area_km2']])
                    st.metric("Avg Watershed Area", f"{avg_area:.2f} km²")
            
            # Results table
            st.markdown("### 📊 Pour Point Analysis Results")
            st.dataframe(results_df, use_container_width=True)
            
            # Export options
            st.markdown("### 📥 Export Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv_data,
                    file_name="pour_point_analysis_results.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Export watersheds as GeoJSON
                if watershed_polygons:
                    ws_gdf = gpd.GeoDataFrame(watershed_polygons, crs='EPSG:4326')
                    geojson_str = ws_gdf.to_json()
                    st.download_button(
                        label="📥 Download Watersheds GeoJSON",
                        data=geojson_str,
                        file_name="watersheds.geojson",
                        mime="application/json"
                    )
            
            with col3:
                # Export as shapefile
                if watershed_polygons:
                    try:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            ws_gdf = gpd.GeoDataFrame(watershed_polygons, crs='EPSG:4326')
                            shp_path = os.path.join(tmpdir, 'watersheds.shp')
                            ws_gdf.to_file(shp_path)
                            
                            zip_path = os.path.join(tmpdir, 'watersheds.zip')
                            with zipfile.ZipFile(zip_path, 'w') as zipf:
                                for file in os.listdir(tmpdir):
                                    if file.startswith('watersheds') and not file.endswith('.zip'):
                                        zipf.write(os.path.join(tmpdir, file), file)
                            
                            with open(zip_path, 'rb') as f:
                                zip_data = f.read()
                            
                            st.download_button(
                                label="📥 Download Watersheds Shapefile",
                                data=zip_data,
                                file_name="watersheds.zip",
                                mime="application/zip"
                            )
                    except Exception as e:
                        st.error(f"Shapefile export error: {e}")

# --- 6B. EVENT HANDLING (THE DELINEATION) ---
if output['last_clicked']:
    click_lat = output['last_clicked']['lat']
    click_lng = output['last_clicked']['lng']

    st.write(f"📍 **Processing Point (WGS84):** {click_lat:.4f}, {click_lng:.4f}")

    try:
        # Convert click coordinates from WGS84 to DEM's CRS if needed
        if crs_info and crs_info['transformer_to_utm'] is not None:
            # Transform from WGS84 to UTM (or whatever the DEM CRS is)
            click_x_utm, click_y_utm = crs_info['transformer_to_utm'].transform(click_lng, click_lat)
            st.write(f"📍 **Converted to DEM CRS:** X={click_x_utm:.2f}, Y={click_y_utm:.2f}")
            click_coords = (click_x_utm, click_y_utm)
        else:
            # DEM is already in WGS84
            click_coords = (click_lng, click_lat)
        
        # Step A: Snap the click to the nearest high-accumulation cell (Stream)
        # This fixes user error if they click slightly off the river
        x_snap, y_snap = grid.snap_to_mask(acc > snap_threshold, click_coords)

        # Step B: Delineate Catchment
        catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, xytype='coordinate')

        # Step C: Clip and Polygonize
        grid.clip_to(catch)
        shapes = grid.polygonize()

        # Step D: Calculate Watershed Properties
        watershed_geom = None
        watershed_geom_wgs84 = None
        watershed_area_km2 = 0
        
        for shape_geom, value in shapes:
            watershed_geom = shape(shape_geom)
            
            # If DEM is in UTM, transform the watershed polygon to WGS84 for display
            if crs_info and crs_info['transformer_to_wgs84'] is not None:
                from shapely.ops import transform as shapely_transform
                watershed_geom_wgs84 = shapely_transform(
                    crs_info['transformer_to_wgs84'].transform, 
                    watershed_geom
                )
                # Calculate area using the UTM geometry (more accurate for projected CRS)
                watershed_area_km2 = watershed_geom.area / 1_000_000  # UTM is in meters
            else:
                watershed_geom_wgs84 = watershed_geom
                # Calculate area using geodetic method for WGS84
                geod = pyproj.Geod(ellps='WGS84')
                area_m2 = abs(geod.geometry_area_perimeter(watershed_geom)[0])
                watershed_area_km2 = area_m2 / 1_000_000
            break
        
        # Get watershed statistics
        catch_acc = acc[catch]
        catch_dem = dem[catch]
        catch_slope = slope[catch] if slope is not None else None
        
        max_acc = np.max(catch_acc[catch_acc > 0]) if np.any(catch_acc > 0) else 0
        min_elev = np.min(catch_dem[catch])
        max_elev = np.max(catch_dem[catch])
        mean_elev = np.mean(catch_dem[catch])
        mean_slope = np.mean(np.abs(catch_slope[catch])) if catch_slope is not None else 0
        
        # Calculate perimeter (use UTM geometry if available for accuracy)
        if crs_info and crs_info['is_projected'] and watershed_geom is not None:
            # UTM/projected - perimeter is in meters directly
            perimeter_m = watershed_geom.length
        else:
            # WGS84 - use geodetic calculation
            geod = pyproj.Geod(ellps='WGS84')
            perimeter_m = abs(geod.geometry_area_perimeter(watershed_geom_wgs84)[1])
        perimeter_km = perimeter_m / 1000
        
        # Convert snap point to WGS84 for display
        if crs_info and crs_info['transformer_to_wgs84'] is not None:
            x_snap_wgs84, y_snap_wgs84 = crs_info['transformer_to_wgs84'].transform(x_snap, y_snap)
        else:
            x_snap_wgs84, y_snap_wgs84 = x_snap, y_snap
        
        # Display Results with Professional Styling
        st.markdown("### 📊 Watershed Analysis Results")
        st.success("✅ Watershed Successfully Delineated!")
        
        # Enhanced Metrics Display
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🏞️ Area</h4>
                <h2>{watershed_area_km2:.2f} km²</h2>
                <p>{watershed_area_km2*100:.0f} hectares</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📏 Perimeter</h4>
                <h2>{perimeter_km:.2f} km</h2>
                <p>{perimeter_m:.0f} meters</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>⛰️ Elevation</h4>
                <h2>{mean_elev:.0f} m</h2>
                <p>Range: {min_elev:.0f}-{max_elev:.0f}m</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📐 Slope</h4>
                <h2>{mean_slope:.1f}°</h2>
                <p>Average gradient</p>
            </div>
            """, unsafe_allow_html=True)
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <h4>💧 Drainage</h4>
                <h2>{int(max_acc):,}</h2>
                <p>Max accumulation</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Create a new map with the result
        if basemap_option == "OpenStreetMap":
            result_map = folium.Map(location=[click_lat, click_lng], zoom_start=13, tiles="OpenStreetMap")
        else:
            result_map = folium.Map(location=[click_lat, click_lng], zoom_start=13, 
                                   tiles=basemap_dict[basemap_option],
                                   attr='Esri' if basemap_option != "Dark Mode" else "CartoDB")
        
        # Add base layers
        if show_streams:
            folium.raster_layers.ImageOverlay(
                image=acc_colored,
                bounds=bounds,
                opacity=0.4,
                name="Stream Network"
            ).add_to(result_map)
        
        # Add the watershed polygon with RED BOUNDARY and semi-transparent fill
        # Use the WGS84 geometry for display
        watershed_json = None
        watershed_boundary_layer = None
        
        if watershed_geom_wgs84 is not None:
            # Convert WGS84 geometry to GeoJSON for display
            watershed_json = mapping(watershed_geom_wgs84)
            
            # Create a feature group for better control
            watershed_layer = folium.FeatureGroup(name="Watershed Boundary", show=True)
            
            # Add filled polygon with prominent RED boundary
            if show_watershed_boundary:
                # Boundary-only mode: no fill, thick red boundary
                boundary_style = {
                    'fillColor': 'transparent',  # No fill
                    'color': '#FF0000',          # RED BOUNDARY
                    'weight': 6,                 # Very thick red line
                    'fillOpacity': 0.0,          # No fill opacity
                    'opacity': 1.0,              # Solid red boundary
                    'dashArray': None            # Solid line
                }
                highlight_style = {
                    'fillColor': 'transparent',  # No fill on hover
                    'color': '#B71C1C',          # Dark red boundary on hover
                    'weight': 8,                 # Even thicker on hover
                    'fillOpacity': 0.0,
                }
            else:
                # Normal mode: semi-transparent fill with red boundary
                boundary_style = {
                    'fillColor': '#4CAF50',      # Green fill
                    'color': '#FF0000',          # RED BOUNDARY
                    'weight': 5,                 # Thick red line
                    'fillOpacity': 0.2,          # Semi-transparent fill
                    'opacity': 1.0,              # Solid red boundary
                    'dashArray': None            # Solid line
                }
                highlight_style = {
                    'fillColor': '#FF5722',      # Orange highlight on hover
                    'color': '#B71C1C',          # Dark red boundary on hover
                    'weight': 6,                 # Thicker on hover
                    'fillOpacity': 0.4,
                }
            
            watershed_boundary = folium.GeoJson(
                watershed_json,
                style_function=lambda x: boundary_style,
                highlight_function=lambda x: highlight_style,
                tooltip=folium.Tooltip(
                    f"""<div style='font-family: Arial; font-size: 12px;'>
                    <b>🏞️ Watershed Boundary</b><br>
                    <b>Area:</b> {watershed_area_km2:.2f} km² ({watershed_area_km2*100:.0f} ha)<br>
                    <b>Perimeter:</b> {perimeter_km:.2f} km<br>
                    <b>Elevation:</b> {min_elev:.0f} - {max_elev:.0f} m (avg: {mean_elev:.0f} m)<br>
                    <b>Mean Slope:</b> {mean_slope:.1f}°<br>
                    <b>Max Drainage:</b> {int(max_acc):,} cells
                    </div>""",
                    sticky=True
                ),
                popup=folium.Popup(
                    f"""<div style='width: 280px; font-family: Arial;'>
                    <h3 style='color: #B71C1C; margin-bottom: 10px; border-bottom: 2px solid #FF0000; padding-bottom: 5px;'>🔴 Watershed Boundary</h3>
                    <table style='width: 100%; font-size: 12px; border-collapse: collapse;'>
                    <tr style='background-color: #ffebee;'><td style='padding: 5px; border: 1px solid #ffcdd2;'><b>Area:</b></td><td style='padding: 5px; border: 1px solid #ffcdd2;'>{watershed_area_km2:.2f} km²</td></tr>
                    <tr><td style='padding: 5px; border: 1px solid #ffcdd2;'><b>Perimeter:</b></td><td style='padding: 5px; border: 1px solid #ffcdd2;'>{perimeter_km:.2f} km</td></tr>
                    <tr style='background-color: #ffebee;'><td style='padding: 5px; border: 1px solid #ffcdd2;'><b>Min Elevation:</b></td><td style='padding: 5px; border: 1px solid #ffcdd2;'>{min_elev:.0f} m</td></tr>
                    <tr><td style='padding: 5px; border: 1px solid #ffcdd2;'><b>Max Elevation:</b></td><td style='padding: 5px; border: 1px solid #ffcdd2;'>{max_elev:.0f} m</td></tr>
                    <tr style='background-color: #ffebee;'><td style='padding: 5px; border: 1px solid #ffcdd2;'><b>Mean Elevation:</b></td><td style='padding: 5px; border: 1px solid #ffcdd2;'>{mean_elev:.0f} m</td></tr>
                    <tr><td style='padding: 5px; border: 1px solid #ffcdd2;'><b>Mean Slope:</b></td><td style='padding: 5px; border: 1px solid #ffcdd2;'>{mean_slope:.1f}°</td></tr>
                    <tr style='background-color: #ffebee;'><td style='padding: 5px; border: 1px solid #ffcdd2;'><b>Drainage Density:</b></td><td style='padding: 5px; border: 1px solid #ffcdd2;'>{int(max_acc):,} cells</td></tr>
                    </table>
                    <p style='margin-top: 10px; font-size: 11px; color: #666;'><i>Click on the boundary to see detailed information</i></p>
                    </div>""",
                    max_width=320
                )
            ).add_to(watershed_layer)
            
            watershed_layer.add_to(result_map)
            watershed_boundary_layer = watershed_layer
            
        # Add area label in the center of watershed (use WGS84 geometry for centroid)
        if watershed_geom_wgs84:
            centroid = watershed_geom_wgs84.centroid
            folium.Marker(
                [centroid.y, centroid.x],
                icon=folium.DivIcon(html=f"""
                    <div style="
                        background-color: white;
                        border: 2px solid #FF0000;
                        border-radius: 5px;
                        padding: 5px 10px;
                        font-weight: bold;
                        font-size: 14px;
                        color: #1e3c72;
                        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
                    ">
                        📍 {watershed_area_km2:.2f} km²
                    </div>
                """)
            ).add_to(result_map)
        
        # Add a marker for the snap point (use WGS84 coordinates)
        folium.Marker([y_snap_wgs84, x_snap_wgs84], 
                     popup=f"""<b>Outlet Point</b><br>
                     Lat: {y_snap_wgs84:.4f}<br>
                     Lon: {x_snap_wgs84:.4f}<br>
                     """,
                     icon=folium.Icon(color='red', icon='tint')).add_to(result_map)
        
        # Add measurement and drawing tools to result map
        if enable_measurement:
            plugins.MeasureControl(position='topleft').add_to(result_map)
        if enable_drawing:
            plugins.Draw(export=True).add_to(result_map)
        
        folium.LayerControl().add_to(result_map)
        st_folium(result_map, width=None, height=750, use_container_width=True)
        
        # --- EXPORT SECTION ---
        st.write("---")
        st.write("### Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        # Export as GeoJSON
        with col1:
            if watershed_json:
                geojson_str = json.dumps(watershed_json, indent=2)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="📥 Download GeoJSON",
                    data=geojson_str,
                    file_name=f"watershed_{timestamp}.geojson",
                    mime="application/json"
                )
        
        # Export as Shapefile (zipped)
        with col2:
            if watershed_geom_wgs84:
                try:
                    # Create temporary directory
                    with tempfile.TemporaryDirectory() as tmpdir:
                        # Create GeoDataFrame with WGS84 geometry
                        gdf = gpd.GeoDataFrame(
                            {'area_km2': [watershed_area_km2],
                             'perim_km': [perimeter_km],
                             'max_acc': [int(max_acc)],
                             'min_elev': [min_elev],
                             'max_elev': [max_elev],
                             'mean_elev': [mean_elev],
                             'mean_slope': [mean_slope]},
                            geometry=[watershed_geom_wgs84],
                            crs='EPSG:4326'
                        )
                        
                        # Save as shapefile
                        shp_path = os.path.join(tmpdir, 'watershed.shp')
                        gdf.to_file(shp_path)
                        
                        # Zip all shapefile components
                        import zipfile
                        zip_path = os.path.join(tmpdir, 'watershed.zip')
                        with zipfile.ZipFile(zip_path, 'w') as zipf:
                            for file in os.listdir(tmpdir):
                                if file.startswith('watershed.'):
                                    zipf.write(os.path.join(tmpdir, file), file)
                        
                        # Read zip file
                        with open(zip_path, 'rb') as f:
                            zip_data = f.read()
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.download_button(
                            label="📥 Download Shapefile",
                            data=zip_data,
                            file_name=f"watershed_{timestamp}.zip",
                            mime="application/zip"
                        )
                except Exception as e:
                    st.error(f"Shapefile export error: {e}")
        
        # Export statistics as CSV
        with col3:
            stats_data = {
                'Parameter': ['Watershed Area (km²)', 'Perimeter (km)', 'Max Accumulation (cells)', 
                            'Min Elevation (m)', 'Max Elevation (m)', 'Mean Elevation (m)', 'Mean Slope (°)',
                            'Outlet Latitude (WGS84)', 'Outlet Longitude (WGS84)', 'DEM CRS'],
                'Value': [f"{watershed_area_km2:.2f}", f"{perimeter_km:.2f}", f"{int(max_acc)}",
                         f"{min_elev:.2f}", f"{max_elev:.2f}", f"{mean_elev:.2f}", f"{mean_slope:.2f}",
                         f"{y_snap_wgs84:.6f}", f"{x_snap_wgs84:.6f}", 
                         crs_info['original_crs'] if crs_info else 'Unknown']
            }
            df_stats = pd.DataFrame(stats_data)
            csv = df_stats.to_csv(index=False)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="📥 Download Statistics (CSV)",
                data=csv,
                file_name=f"watershed_stats_{timestamp}.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.warning(f"⚠️ Could not delineate here. Try clicking closer to a major stream. (Error: {e})")
        import traceback
        st.error(traceback.format_exc())

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <h4>🌊 Professional WebGIS Watershed Delineation System</h4>
    <p>Built with Streamlit, PySheds, Folium, and GeoPandas</p>
    <p>© 2026 - Advanced Hydrological Analysis & GIS Tools</p>
    <p><small>Version 2.0 - Professional Edition</small></p>
</div>
""", unsafe_allow_html=True)