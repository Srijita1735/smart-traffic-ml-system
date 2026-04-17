import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
import networkx as nx

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut


st.set_page_config(page_title="Smart Traffic System", layout="wide")

st.title("🚦 Smart Traffic Management System")


geolocator = Nominatim(user_agent="smart_traffic_app")

def get_coords(place):
    try:
        location = geolocator.geocode(place, timeout=5)
        if location:
            return (location.latitude, location.longitude)
        else:
            return (None, None)
    except (GeocoderUnavailable, GeocoderTimedOut):
        return (None, None)


st.sidebar.header("📍 Enter Locations")

start_location = st.sidebar.text_input("Start Location", "Chennai")
end_location = st.sidebar.text_input("End Location", "Bangalore")


start_coords = get_coords(start_location)
end_coords = get_coords(end_location)

if start_coords[0] is None:
    st.error("⚠️ Unable to fetch START location. Try another place.")
    st.stop()

if end_coords[0] is None:
    st.error("⚠️ Unable to fetch END location. Try another place.")
    st.stop()


st.subheader("🗺 Route Visualization")

m = folium.Map(location=start_coords, zoom_start=6)

folium.Marker(start_coords, tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
folium.Marker(end_coords, tooltip="End", icon=folium.Icon(color="red")).add_to(m)


folium.PolyLine([start_coords, end_coords], color="blue", weight=3).add_to(m)

st_folium(m, width=700, height=500)


st.subheader("📊 Traffic Prediction")

hours = np.arange(0, 24)
traffic = 300 + 50 * np.sin(hours / 3) + np.random.randint(-20, 20, 24)

df = pd.DataFrame({
    "Hour": hours,
    "Traffic": traffic
})

st.line_chart(df.set_index("Hour"))


st.subheader("🚦 Recommendation")

best_time = df.loc[df["Traffic"].idxmin(), "Hour"]

st.success(f"✅ Best time to travel: {int(best_time)}:00 hrs")


st.subheader("🛣 Route Optimization")

G = nx.Graph()

G.add_edge("A", "B", weight=4)
G.add_edge("B", "C", weight=3)
G.add_edge("A", "C", weight=10)

path = nx.shortest_path(G, "A", "C", weight="weight")

st.write(f"Optimal Route: {path}")