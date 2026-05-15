import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import networkx as nx

from geopy.geocoders import Nominatim

from geopy.exc import (
    GeocoderUnavailable,
    GeocoderTimedOut,
    GeocoderServiceError
)

from geopy.extra.rate_limiter import RateLimiter

from geopy.distance import geodesic

st.set_page_config(
    page_title="Smart Traffic Management System",
    page_icon="🚦",
    layout="wide"
)

st.markdown("""
<style>

.main {
    background-color: #020617;
}

.card {
    padding: 25px;
    border-radius: 16px;
    background: linear-gradient(135deg, #172554, #0f172a);
    color: white;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
}

.big-font {
    font-size: 22px;
    font-weight: bold;
}

.metric {
    font-size: 32px;
    font-weight: bold;
    color: #38bdf8;
}

</style>
""", unsafe_allow_html=True)

st.title("🚦 Smart Traffic Management System")

st.markdown(
    """
    AI-Based Dynamic Traffic Prediction, Route Optimization,
    and ETA Forecasting Across India
    """
)

geolocator = Nominatim(
    user_agent="smart_traffic_india_2026"
)

geocode = RateLimiter(
    geolocator.geocode,
    min_delay_seconds=1
)

fallback_locations = {

    "chennai": (13.0827, 80.2707),

    "bangalore": (12.9716, 77.5946),

    "delhi": (28.7041, 77.1025),

    "mumbai": (19.0760, 72.8777),

    "kolkata": (22.5726, 88.3639),

    "hyderabad": (17.3850, 78.4867),

    "krishnanagar": (23.4058, 88.4907)

}


def get_coords(place):

    try:

        location = geocode(
            place,
            timeout=10
        )

        if location:

            return (
                location.latitude,
                location.longitude
            )

    except (

        GeocoderUnavailable,
        GeocoderTimedOut,
        GeocoderServiceError

    ):

        pass

    return fallback_locations.get(
        place.lower(),
        (20.5937, 78.9629)
    )


st.sidebar.header("📍 Enter Journey Details")

start_location = st.sidebar.text_input(
    "Start Location",
    "Krishnanagar"
)

end_location = st.sidebar.text_input(
    "Destination",
    "Kolkata"
)

time_input = st.sidebar.slider(
    "Departure Time (Hour)",
    0,
    23,
    14
)

date_input = st.sidebar.date_input(
    "Select Date"
)

weather = st.sidebar.selectbox(
    "Weather Condition",
    [
        "Clear",
        "Rain",
        "Heavy Rain"
    ]
)

event = st.sidebar.selectbox(
    "Nearby Event",
    [
        "No Event",
        "Festival",
        "Concert",
        "Accident"
    ]
)

start_coords = get_coords(start_location)

end_coords = get_coords(end_location)

distance_km = round(
    geodesic(
        start_coords,
        end_coords
    ).km,
    2
)

base_traffic = 250

traffic = base_traffic

if 7 <= time_input <= 11:

    traffic += 120

elif 17 <= time_input <= 21:

    traffic += 150

else:

    traffic += 50

if weather == "Rain":

    traffic += 70

elif weather == "Heavy Rain":

    traffic += 130

if event == "Festival":

    traffic += 100

elif event == "Concert":

    traffic += 80

elif event == "Accident":

    traffic += 150

traffic += np.random.randint(-15, 15)

if traffic < 350:

    average_speed = 70

elif traffic < 500:

    average_speed = 45

else:

    average_speed = 25

if weather == "Rain":

    average_speed -= 10

elif weather == "Heavy Rain":

    average_speed -= 20

average_speed = max(15, average_speed)

estimated_travel_time = (
    distance_km / average_speed
) * 60

travel_hours = int(
    estimated_travel_time // 60
)

travel_minutes = int(
    estimated_travel_time % 60
)

departure_minutes = time_input * 60

arrival_total_minutes = (
    departure_minutes +
    estimated_travel_time
)

arrival_hour = int(
    arrival_total_minutes // 60
) % 24

arrival_minutes = int(
    arrival_total_minutes % 60
)

G = nx.Graph()

G.add_edge(
    "Route A",
    "Route B",
    weight=traffic
)

G.add_edge(
    "Route B",
    "Route C",
    weight=traffic - 40
)

G.add_edge(
    "Route A",
    "Route C",
    weight=traffic + 20
)

path = nx.shortest_path(
    G,
    "Route A",
    "Route C",
    weight="weight"
)

st.markdown("## 🚦 Smart Traffic Dashboard")

col1, col2 = st.columns(2)

with col1:

    st.markdown(f"""
    <div class="card">

    <div class="big-font">
    📍 Route
    </div>

    <br>

    {start_location} → {end_location}

    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown(f"""
    <div class="card">

    <div class="big-font">
    🚗 Departure Time
    </div>

    <br>

    {time_input}:00 hrs

    </div>
    """, unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:

    st.markdown(f"""
    <div class="card">

    <div class="big-font">
    🌦 Weather
    </div>

    <br>

    {weather}

    </div>
    """, unsafe_allow_html=True)

with col4:

    st.markdown(f"""
    <div class="card">

    <div class="big-font">
    🎉 Event
    </div>

    <br>

    {event}

    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.subheader("🗺 Live Route Map")

m = folium.Map(
    location=start_coords,
    zoom_start=6
)

folium.Marker(
    start_coords,
    tooltip="Start Location",
    icon=folium.Icon(color="green")
).add_to(m)

folium.Marker(
    end_coords,
    tooltip="Destination",
    icon=folium.Icon(color="red")
).add_to(m)

folium.PolyLine(
    [start_coords, end_coords],
    color="blue",
    weight=6
).add_to(m)

st_folium(
    m,
    width=1200,
    height=500
)

st.markdown("---")

st.subheader("📊 Traffic Analysis")

col5, col6, col7 = st.columns(3)

with col5:

    st.markdown(f"""
    <div class="card">

    🚗 Traffic Volume

    <div class="metric">
    {int(traffic)}
    </div>

    </div>
    """, unsafe_allow_html=True)

with col6:

    st.markdown(f"""
    <div class="card">

    📍 Distance

    <div class="metric">
    {distance_km} km
    </div>

    </div>
    """, unsafe_allow_html=True)

with col7:

    st.markdown(f"""
    <div class="card">

    ⚡ Avg Speed

    <div class="metric">
    {average_speed} km/h
    </div>

    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.subheader("🕒 Estimated Time of Arrival")

st.markdown(f"""
<div class="card">

🚗 <b>Departure Time</b><br><br>

{time_input}:00 hrs

<br><br>

⏳ <b>Estimated Travel Time</b><br><br>

{travel_hours} hrs {travel_minutes} mins

<br><br>

⏰ <b>Estimated Arrival Time (ETA)</b><br><br>

{arrival_hour}:{arrival_minutes:02d} hrs

</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.subheader("🚦 Traffic Recommendation")

if traffic < 350:

    color = "#065f46"

    msg = "Low Traffic — Smooth Journey Expected"

elif traffic < 500:

    color = "#92400e"

    msg = "Moderate Traffic — Expect Delays"

else:

    color = "#991b1b"

    msg = "Heavy Traffic — Consider Alternate Route"

st.markdown(f"""
<div style="
background:{color};
padding:20px;
border-radius:15px;
color:white;
font-size:22px;
text-align:center;
">

{msg}

</div>
""", unsafe_allow_html=True)

st.markdown("---")

st.subheader("🛣 Optimal Route Suggestion")

st.markdown(f"""
<div class="card">

Recommended Smart Route:

<br><br>

{path}

</div>
""", unsafe_allow_html=True)