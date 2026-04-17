import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import datetime
import networkx as nx

st.set_page_config(page_title="Smart Traffic System", layout="wide")

st.title("🚦 Smart Traffic Intelligence System")

# -----------------------------
# SESSION STATE
# -----------------------------
if "show_map" not in st.session_state:
    st.session_state.show_map = False

# -----------------------------
# USER INPUTS (FINAL UI)
# -----------------------------
st.subheader("📍 Enter Travel Details")

col1, col2 = st.columns(2)

with col1:
    start_location = st.text_input("Start Location", "Chennai")

with col2:
    end_location = st.text_input("Destination", "Bangalore")

col3, col4, col5 = st.columns(3)

with col3:
    travel_time = st.time_input("Time of Travel", datetime.time(12, 0))

with col4:
    weather = st.selectbox("Weather", ["Clear", "Rain", "Heavy Rain"])

with col5:
    travel_date = st.date_input("Date of Travel")

event = st.checkbox("Event / Festival?")

# Convert time to hour
hour = travel_time.hour

# Convert weather
rain = 0
if weather == "Rain":
    rain = 5
elif weather == "Heavy Rain":
    rain = 9

# -----------------------------
# BUTTON
# -----------------------------
if st.button("🚀 Show Route & Prediction"):
    st.session_state.show_map = True

# -----------------------------
# GEOLOCATION
# -----------------------------
geolocator = Nominatim(user_agent="traffic_app")

def get_coords(place):
    location = geolocator.geocode(place)
    if location:
        return (location.latitude, location.longitude)
    return None

# -----------------------------
# MAIN OUTPUT
# -----------------------------
if st.session_state.show_map:

    start_coords = get_coords(start_location)
    end_coords = get_coords(end_location)

    if start_coords and end_coords:

        # -----------------------------
        # MAP
        # -----------------------------
        mid_lat = (start_coords[0] + end_coords[0]) / 2
        mid_lon = (start_coords[1] + end_coords[1]) / 2

        m = folium.Map(location=[mid_lat, mid_lon], zoom_start=6)

        folium.Marker(start_coords, tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
        folium.Marker(end_coords, tooltip="End", icon=folium.Icon(color="red")).add_to(m)

        folium.PolyLine([start_coords, end_coords], color="blue", weight=5).add_to(m)

        st.subheader("🗺 Route Map")
        st_folium(m, width=1000, height=500)

        # -----------------------------
        # TRAFFIC MODEL
        # -----------------------------
        traffic = 300

        if hour in [8, 9, 18, 19]:
            traffic *= 1.6
        elif hour in [11, 12, 13, 14]:
            traffic *= 0.85

        if rain > 5:
            traffic *= 1.2

        if event:
            traffic *= 1.4

        st.subheader("📊 Traffic Prediction")
        st.write(f"Estimated Traffic: {traffic:.2f}")

        # -----------------------------
        # RECOMMENDATION
        # -----------------------------
        st.subheader("🕒 Travel Recommendation")

        if traffic < 350:
            st.success("✅ Good time to travel")
        else:
            st.error("⚠️ Avoid this time")

        # -----------------------------
        # AI EXPLANATION
        # -----------------------------
        st.subheader("🧠 AI Explanation")

        reasons = []

        if weather != "Clear":
            reasons.append("Weather conditions")

        if hour in [8, 9, 18, 19]:
            reasons.append("Peak hour")

        if event:
            reasons.append("Event")

        if reasons:
            st.write("Traffic influenced by:", ", ".join(reasons))
        else:
            st.write("Traffic conditions normal")

        # -----------------------------
        # GRAPH ROUTE
        # -----------------------------
        st.subheader("🧠 Smart Routing")

        G = nx.Graph()
        G.add_edge("A", "B", weight=5)
        G.add_edge("B", "C", weight=3)
        G.add_edge("A", "C", weight=10)

        path = nx.shortest_path(G, "A", "C", weight="weight")

        st.write("Optimized Route:", path)

        # -----------------------------
        # CARBON EMISSION
        # -----------------------------
        st.subheader("🌱 Carbon Emission")

        emission = traffic * 0.2
        st.write(f"CO2 Emission: {emission:.2f}")

    else:
        st.error("❌ Location not found. Try different names.")