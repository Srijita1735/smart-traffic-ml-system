import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import networkx as nx

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Smart Traffic System", layout="wide")

st.title("🚦 Smart Traffic Management System")

# -----------------------------
# GEOLOCATOR + FALLBACK SYSTEM
# -----------------------------
geolocator = Nominatim(user_agent="smart_traffic_app")

# fallback locations (VERY IMPORTANT FIX)
fallback_locations = {
    "chennai": (13.0827, 80.2707),
    "bangalore": (12.9716, 77.5946),
    "delhi": (28.7041, 77.1025),
    "mumbai": (19.0760, 72.8777)
}

def get_coords(place):
    try:
        location = geolocator.geocode(place, timeout=5)
        if location:
            return (location.latitude, location.longitude)
    except (GeocoderUnavailable, GeocoderTimedOut):
        pass

    # fallback if API fails
    place_lower = place.lower()
    if place_lower in fallback_locations:
        return fallback_locations[place_lower]

    # default fallback (Chennai)
    return (13.0827, 80.2707)

# -----------------------------
# USER INPUT
# -----------------------------
st.sidebar.header("📍 Enter Locations")

start_location = st.sidebar.text_input("Start Location", "Chennai")
end_location = st.sidebar.text_input("End Location", "Bangalore")

# -----------------------------
# GET COORDS (SAFE)
# -----------------------------
start_coords = get_coords(start_location)
end_coords = get_coords(end_location)

# -----------------------------
# MAP DISPLAY
# -----------------------------
st.subheader("🗺 Route Visualization")

m = folium.Map(location=start_coords, zoom_start=6)

# markers
folium.Marker(start_coords, tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
folium.Marker(end_coords, tooltip="End", icon=folium.Icon(color="red")).add_to(m)

# route line
folium.PolyLine([start_coords, end_coords], color="blue", weight=3).add_to(m)

st_folium(m, width=700, height=500)

# -----------------------------
# TRAFFIC PREDICTION (DEMO)
# -----------------------------
st.subheader("📊 Traffic Prediction")

hours = np.arange(0, 24)
traffic = 300 + 50 * np.sin(hours / 3) + np.random.randint(-20, 20, 24)

df = pd.DataFrame({
    "Hour": hours,
    "Traffic": traffic
})

st.line_chart(df.set_index("Hour"))

# -----------------------------
# RECOMMENDATION
# -----------------------------
st.subheader("🚦 Recommendation")

best_time = df.loc[df["Traffic"].idxmin(), "Hour"]

st.success(f"✅ Best time to travel: {int(best_time)}:00 hrs")

# -----------------------------
# ROUTE OPTIMIZATION
# -----------------------------
st.subheader("🛣 Route Optimization")

G = nx.Graph()
G.add_edge("A", "B", weight=4)
G.add_edge("B", "C", weight=3)
G.add_edge("A", "C", weight=10)

path = nx.shortest_path(G, "A", "C", weight="weight")

st.write(f"Optimal Route: {path}")