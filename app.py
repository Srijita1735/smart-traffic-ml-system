import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import networkx as nx
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut

st.set_page_config(page_title="Smart Traffic System", layout="wide")

st.markdown("""
<style>
.main {
    background-color: #0f172a;
}
.card {
    padding: 20px;
    border-radius: 12px;
    background: linear-gradient(135deg, #1e293b, #0f172a);
    color: white;
    text-align: center;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

st.title("🚦 Smart Traffic Management System")

geolocator = Nominatim(user_agent="smart_traffic_app")

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
    return fallback_locations.get(place.lower(), (13.0827, 80.2707))

st.sidebar.header("📍 Enter Details")

start_location = st.sidebar.text_input("Start Location", "Chennai")
end_location = st.sidebar.text_input("End Location", "Bangalore")
time_input = st.sidebar.slider("Select Time (Hour)", 0, 23, 14)
date_input = st.sidebar.date_input("Select Date")
weather = st.sidebar.selectbox("Weather", ["Clear", "Rain", "Heavy Rain"])
event = st.sidebar.selectbox("Event", ["No Event", "Festival", "Concert", "Accident"])

start_coords = get_coords(start_location)
end_coords = get_coords(end_location)

base_traffic = 300
traffic = base_traffic + (time_input * 5)

if weather == "Rain":
    traffic += 50
elif weather == "Heavy Rain":
    traffic += 100

if event != "No Event":
    traffic += 80

traffic = traffic + np.random.randint(-20, 20)

G = nx.Graph()
G.add_edge("A", "B", weight=traffic)
G.add_edge("B", "C", weight=traffic - 50)
G.add_edge("A", "C", weight=traffic + 30)

path = nx.shortest_path(G, "A", "C", weight="weight")

st.markdown("## 🚦 Smart Traffic Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="card">
    📍 <b>Route</b><br>
    {start_location} → {end_location}
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
    ⏰ <b>Time</b><br>
    {time_input}:00 hrs
    </div>
    """, unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    st.markdown(f"""
    <div class="card">
    🌦 <b>Weather</b><br>
    {weather}
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="card">
    🎉 <b>Event</b><br>
    {event}
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.subheader("🗺 Route Map")

m = folium.Map(location=start_coords, zoom_start=6)

folium.Marker(start_coords, tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
folium.Marker(end_coords, tooltip="End", icon=folium.Icon(color="red")).add_to(m)
folium.PolyLine([start_coords, end_coords], color="blue").add_to(m)

st_folium(m, width=1000, height=500)

st.markdown("---")

st.subheader("📊 Traffic Prediction")

st.markdown(f"""
<div class="card">
🚗 <b>Traffic Volume</b>
<h2>{int(traffic)}</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.subheader("🚦 Recommendation")

if traffic < 350:
    color = "#064e3b"
    msg = "Low Traffic - Good to travel"
elif traffic < 450:
    color = "#78350f"
    msg = "Moderate Traffic - Plan accordingly"
else:
    color = "#7f1d1d"
    msg = "High Traffic - Avoid"

st.markdown(f"""
<div style="background:{color};padding:15px;border-radius:10px;color:white;text-align:center;">
{msg}
</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.subheader("🛣 Optimal Route")

st.markdown(f"""
<div class="card">
{path}
</div>
""", unsafe_allow_html=True)