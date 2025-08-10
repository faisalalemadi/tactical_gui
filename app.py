import streamlit as st
import os
import math
import numpy as np
import rasterio
from rasterio.windows import from_bounds
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import pydeck as pdk
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAI
from datetime import datetime
import json

# ---------- UI: wide layout + header ----------
st.set_page_config(page_title="🧠 Tactical Reasoning Assistant", page_icon="🛰️", layout="wide")
st.title("🧠 Tactical Reasoning Assistant (Qatar Armed Forces)")

# Session history for exports
if "runs" not in st.session_state:
    st.session_state["runs"] = []

# ===== Model Config =====
DEFAULT_MODEL = "gpt-4o"  # change to gpt-4o, gpt-5, or gpt-5-reasoning as needed
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Mapbox token for pydeck (must be a public 'pk.' token)
MAPBOX_TOKEN = st.secrets.get("MAPBOX_TOKEN") or os.getenv("MAPBOX_TOKEN")
if MAPBOX_TOKEN:
    pdk.settings.mapbox_api_key = MAPBOX_TOKEN
else:
    st.warning("Mapbox token missing. Add MAPBOX_TOKEN in Streamlit → Settings → Secrets for the basemap to render.")

    
# Allow duplicate OpenMP DLL (Windows quirk)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# === Constants ===
DEM_FOLDER = "./dem_tiles"
LANDCOVER_FOLDER = "./landcover_tiles"

WORLD_COVER_CLASSES = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
    50: "Built-up", 60: "Bare / sparse vegetation", 70: "Snow and ice",
    80: "Permanent water bodies", 90: "Herbaceous wetland",
    95: "Mangroves", 100: "Moss and lichen"
}

ESA_COLORS = {
    10: "#006400", 20: "#FFBB22", 30: "#FFFF4C", 40: "#F096FF",
    50: "#FA0000", 60: "#B4B4B4", 70: "#F0F0F0", 80: "#0032C8",
    90: "#0096A0", 95: "#00CF75", 100: "#FAE6A0"
}
ESA_KEYS = sorted(ESA_COLORS.keys())


# Qatar-specific constraints
QATAR_BOMBING_TYPES = [
    "Precision Bombing",
    "Stand-off Strike",
    "Close Air Support (CAS)",
    "Interdiction",
    "Anti-Armor Strike",
    "SEAD/DEAD",
    "Maritime Strike",
]

QATAR_AIR_PLATFORMS = [
    "F-15QA",
    "Dassault Rafale",
    "Eurofighter Typhoon",
    "AH-64E Apache",
    "Bayraktar TB2",
]

QATAR_GROUND_PLATFORMS = [
    "M142 HIMARS",
    "PzH 2000",
]

QATAR_AIR_MUNITIONS = [
    "AASM Hammer", "SCALP-EG", "Paveway IV", "Brimstone 2",
    "JDAM", "SDB", "AGM-114R Hellfire", "APKWS", "MAM-L"
]

QATAR_GROUND_MUNITIONS = [
    "GMLRS",
    "ATACMS",
    "155mm ERFB-BB",
    "155mm HE"
]

@st.cache_resource
def load_vectorstore():
    return FAISS.load_local(
        "doctrine_vectorstore",
        OpenAIEmbeddings(model="text-embedding-3-small"),
        allow_dangerous_deserialization=True
    )

vectorstore = load_vectorstore()

# === Utility: Get tile path by lat/lon ===
def get_tile_path(folder, lat, lon, extension):
    lat_prefix = 'N' if lat >= 0 else 'S'
    lon_prefix = 'E' if lon >= 0 else 'W'
    if extension == ".hgt":
        lat_deg = int(math.floor(lat))
        lon_deg = int(math.floor(lon))
    else:
        lat_deg = int(math.floor(lat / 3.0) * 3)
        lon_deg = int(math.floor(lon / 3.0) * 3)

    lat_str = f"{lat_prefix}{abs(lat_deg):02d}"
    lon_str = f"{lon_prefix}{abs(lon_deg):03d}"

    if extension == ".hgt":
        filename = f"{lat_str}{lon_str}{extension}"
    else:
        filename = f"ESA_WorldCover_10m_2021_v200_{lat_str}{lon_str}_Map{extension}"

    full_path = os.path.join(folder, filename)
    return full_path if os.path.exists(full_path) else None

def expected_tile_path(folder, lat, lon, extension):
    lat_prefix = 'N' if lat >= 0 else 'S'
    lon_prefix = 'E' if lon >= 0 else 'W'
    if extension == ".hgt":
        lat_deg = int(math.floor(lat))
        lon_deg = int(math.floor(lon))
        lat_str = f"{lat_prefix}{abs(lat_deg):02d}"
        lon_str = f"{lon_prefix}{abs(lon_deg):03d}"
        filename = f"{lat_str}{lon_str}{extension}"
    else:
        lat_deg = int(math.floor(lat / 3.0) * 3)
        lon_deg = int(math.floor(lon / 3.0) * 3)
        lat_str = f"{lat_prefix}{abs(lat_deg):02d}"
        lon_str = f"{lon_prefix}{abs(lon_deg):03d}"
        filename = f"ESA_WorldCover_10m_2021_v200_{lat_str}{lon_str}_Map{extension}"
    return os.path.join(folder, filename)

# === Feature Extraction ===
def extract_features(lat, lon):
    buffer_deg = 1.0 / 111.0
    minx, maxx = lon - buffer_deg, lon + buffer_deg
    miny, maxy = lat - buffer_deg, lat + buffer_deg

    dem_path = get_tile_path(DEM_FOLDER, lat, lon, ".hgt")
    lc_path = get_tile_path(LANDCOVER_FOLDER, lat, lon, ".tif")

    if not dem_path or not lc_path:
        exp_dem = expected_tile_path(DEM_FOLDER, lat, lon, ".hgt")
        exp_lc = expected_tile_path(LANDCOVER_FOLDER, lat, lon, ".tif")
        raise FileNotFoundError(
            "Missing raster tiles for this coordinate.\n"
            f"- DEM expected: {exp_dem} -> {'FOUND' if dem_path else 'NOT FOUND'}\n"
            f"- Land cover expected: {exp_lc} -> {'FOUND' if lc_path else 'NOT FOUND'}"
        )

    with rasterio.open(dem_path) as dem:
        window = from_bounds(minx, miny, maxx, maxy, dem.transform)
        dem_data = dem.read(1, window=window).astype("float64")
        dem_data = np.where(dem_data == dem.nodata, np.nan, dem_data)

        transform = dem.window_transform(window)
        xres_deg = transform.a
        yres_deg = abs(transform.e)
        center_lat = (miny + maxy) / 2.0
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(center_lat))
        m_per_deg_lat = 111_320.0
        xres_m = xres_deg * m_per_deg_lon
        yres_m = yres_deg * m_per_deg_lat

        dzdx = np.gradient(dem_data, axis=1) / xres_m
        dzdy = np.gradient(dem_data, axis=0) / yres_m
        slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
        elevation = np.nanmean(dem_data)
        slope_deg = np.nanmean(np.degrees(slope_rad))

    with rasterio.open(lc_path) as lc:
        window = from_bounds(minx, miny, maxx, maxy, lc.transform)
        lc_data = lc.read(1, window=window).astype("float64")
        lc_data = np.where(lc_data == lc.nodata, np.nan, lc_data)
        terrain_code_raw = np.nanmean(lc_data)
        terrain_code = int(np.round(terrain_code_raw))
        closest_code = min(WORLD_COVER_CLASSES.keys(), key=lambda x: abs(x - terrain_code))
        terrain_class = WORLD_COVER_CLASSES[closest_code]

    return {
        "elevation_m": round(float(elevation), 2),
        "slope_deg": round(float(slope_deg), 2),
        "terrain_code": closest_code,
        "terrain_class": terrain_class
    }

# === Plot land cover ===
def plot_landcover(lc_path, lat, lon):
    buffer_deg = 1.0 / 111.0
    minx, maxx = lon - buffer_deg, lon + buffer_deg
    miny, maxy = lat - buffer_deg, lat + buffer_deg

    with rasterio.open(lc_path) as src:
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        data = src.read(1, window=window)
        data = np.where(data == src.nodata, np.nan, data)
        bounds = rasterio.windows.bounds(window, src.transform)

    cmap = mcolors.ListedColormap([ESA_COLORS[k] for k in ESA_KEYS])
    boundaries = ESA_KEYS + [ESA_KEYS[-1] + 10]
    norm = mcolors.BoundaryNorm(boundaries=boundaries, ncolors=len(ESA_KEYS))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(
        data, cmap=cmap, norm=norm,
        extent=[bounds[0], bounds[2], bounds[1], bounds[3]]
    )
    ax.plot(lon, lat, 'ro', label="Target Point")
    ax.set_title("Land Cover Map (Local Region)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    patches = [
        plt.plot([], [], marker="s", ms=10, ls="", mec=None, color=ESA_COLORS[k],
                 label=WORLD_COVER_CLASSES[k])[0]
        for k in ESA_KEYS if k in WORLD_COVER_CLASSES
    ]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', title="Land Cover Class")
    return fig

# === GPT Qatar Response ===
def generate_qatar_response(features, tactical_description, model_name=DEFAULT_MODEL):
    query = f"{tactical_description}. Terrain is {features['terrain_class']}, elevation {features['elevation_m']}m, slope {features['slope_deg']}°."
    try:
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([d.page_content for d in docs]) if docs else ""
    except Exception:
        context = ""

    system_msg = (
        "You are a tactical reasoning assistant for the Qatar Armed Forces. "
        "Select a bombing type, delivery platform, and munitions ONLY from the allowed lists below. "
        "Respond ONLY with valid JSON in this exact schema:\n"
        '{"type_of_bombing":"","delivery_platform":"","munitions":[],"justification":[],"additional_considerations":[]}\n\n'
        f"Allowed bombing types: {QATAR_BOMBING_TYPES}\n"
        f"Allowed delivery platforms: {QATAR_AIR_PLATFORMS + QATAR_GROUND_PLATFORMS}\n"
        f"Allowed air munitions: {QATAR_AIR_MUNITIONS}\n"
        f"Allowed ground munitions: {QATAR_GROUND_MUNITIONS}\n"
        "- munitions MUST be a list of max 2 items from the correct category for the chosen platform."
    )

    user_msg = f"Doctrine:\n{context}\n\nTerrain:\n{features}\n\nTactical Description:\n{tactical_description}"

    try:
        if model_name in ["gpt-4o", "gpt-4o-mini", "gpt-5", "gpt-5-reasoning"]:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system_msg},
                          {"role": "user", "content": user_msg}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            data = json.loads(resp.choices[0].message.content)
        else:
            system_msg_fallback = system_msg + "\nRespond ONLY with pure JSON."
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system_msg_fallback},
                          {"role": "user", "content": user_msg}],
                temperature=0
            )
            data = json.loads(resp.choices[0].message.content)
    except Exception as e:
        raise RuntimeError(f"Model response error: {e}")

    if data.get("delivery_platform") in QATAR_AIR_PLATFORMS:
        allowed_munitions = QATAR_AIR_MUNITIONS
    elif data.get("delivery_platform") in QATAR_GROUND_PLATFORMS:
        allowed_munitions = QATAR_GROUND_MUNITIONS
    else:
        data["delivery_platform"] = "F-15QA"
        allowed_munitions = QATAR_AIR_MUNITIONS

    data["munitions"] = [m for m in data.get("munitions", []) if m in allowed_munitions][:2]
    if not data["munitions"]:
        data["munitions"] = [allowed_munitions[0]]

    return data

# === Inputs ===
lat = st.number_input("Latitude", format="%f", value=31.040000)
lon = st.number_input("Longitude", format="%f", value=34.850000)
tactical_description = st.text_area("Mission / Tactical Description", placeholder="e.g. Radar near civilian area")

# === Streamlit UI ===
if st.button("Generate Recommendation", type="primary"):
    try:
        features = extract_features(lat, lon)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        st.stop()

        # --- Map ---
    st.markdown("### 🗺️ Location Map (Sentinel-2)")
        
  
    # Target point
    target_df = pd.DataFrame([{"lat": float(lat), "lon": float(lon), "tooltip": "Target"}])
    pdk.settings.mapbox_api_key = None  # disable Mapbox completely
    deck = pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=13, pitch=45),
        layers=[
            pdk.Layer("TileLayer",
                      data="https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                      min_zoom=0, max_zoom=19, tile_size=256),
            pdk.Layer("ScatterplotLayer", data=target_df, get_position='[lon, lat]', get_radius=120,
                      get_fill_color='[255,0,0,200]', pickable=True),
        ],
    )
    st.pydeck_chart(deck, use_container_width=True, height=700)

    # --- Land cover ---
    st.markdown("### 🖼️ Land Cover Region")
    fig = plot_landcover(expected_tile_path(LANDCOVER_FOLDER, lat, lon, ".tif"), lat, lon)
    st.pyplot(fig)

    # --- Feature JSON ---
    st.markdown("### 📍 Geospatial Features")
    st.json({k: v for k, v in features.items()})

    # --- GPT recommendation ---
    st.markdown("### 🧠 GPT Tactical Recommendation")
    try:
        gpt_response = generate_qatar_response(features, tactical_description)
        st.json(gpt_response)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    # --- Export session history ---
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "lat": float(lat),
            "lon": float(lon),
            "tactical_description": tactical_description.strip(),
        },
        "features": features,
        "recommendation": gpt_response,
    }
    st.session_state["runs"].append(record)

    st.markdown("### ⬇️ Export")
    if st.session_state["runs"]:
        hist_json = json.dumps(st.session_state["runs"], indent=2)
        st.download_button(
            "Download Session History (JSON)",
            hist_json,
            file_name="session_runs.json",
            mime="application/json",
        )
















