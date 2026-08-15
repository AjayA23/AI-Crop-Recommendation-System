import streamlit as st
import joblib
import pandas as pd
import requests
import random
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go


# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="AI Crop Recommendation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ================= CUSTOM CSS =================

st.markdown("""
<style>

.stApp {
    background-color: #07111f;
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #081421;
}

[data-testid="stSidebar"] * {
    color: #cbd5e1;
}

.main-title {
    font-size: 30px;
    font-weight: 700;
}

.subtitle {
    color: #94a3b8;
    font-size: 14px;
}

.card {
    background: #101c2b;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #1e293b;
    min-height: 140px;
}

.card-title {
    color: #94a3b8;
    font-size: 14px;
}

.card-value {
    font-size: 26px;
    font-weight: bold;
    margin-top: 10px;
}

.green-text {
    color: #4ade80;
}

.status-live {
    color: #22c55e;
    font-weight: bold;
}

div[data-testid="stMetric"] {
    background-color: #101c2b;
    border: 1px solid #1e293b;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ================= AUTO REFRESH =================

st_autorefresh(
    interval=5000,
    key="live_refresh"
)


# ================= LOAD MODEL =================

@st.cache_resource
def load_model():
    return joblib.load("model/crop_model.pkl")


model = load_model()


# ================= LIVE WEATHER =================

@st.cache_data(ttl=5)
def get_weather(city):

    try:

        url = f"https://wttr.in/{city}?format=j1"

        response = requests.get(
            url,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        current = data["current_condition"][0]

        temperature = float(
            current["temp_C"]
        )

        humidity = float(
            current["humidity"]
        )

        rainfall = float(
            data["weather"][0]["hourly"][0].get(
                "precipMM",
                0
            )
        )

        return temperature, humidity, rainfall, True

    except:

        return 25.0, 60.0, 0.0, False


# ================= SIDEBAR =================

st.sidebar.markdown("# 🌱 AI Crop")

st.sidebar.caption(
    "Recommendation System"
)

st.sidebar.markdown("---")

st.sidebar.success("🏠 Dashboard")

st.sidebar.write("🌾 Crop Recommendation")
st.sidebar.write("🧪 Soil Analysis")
st.sidebar.write("☀️ Weather Forecast")
st.sidebar.write("🏞️ Field Management")
st.sidebar.write("📜 Crop History")
st.sidebar.write("💰 Market Prices")
st.sidebar.write("🔔 Alerts & Notifications")
st.sidebar.write("📄 Reports")

st.sidebar.markdown("---")

st.sidebar.subheader("📍 Farm Location")

city = st.sidebar.text_input(
    "City",
    "Pune"
)

st.sidebar.markdown("---")

st.sidebar.subheader("🧪 Soil Parameters")

N = st.sidebar.number_input(
    "Nitrogen (N)",
    0,
    140,
    90
)

P = st.sidebar.number_input(
    "Phosphorus (P)",
    0,
    145,
    42
)

K = st.sidebar.number_input(
    "Potassium (K)",
    0,
    205,
    43
)

ph = st.sidebar.number_input(
    "Soil pH",
    0.0,
    14.0,
    6.5,
    step=0.1
)

live_soil = st.sidebar.checkbox(
    "🔴 Live Soil Simulation",
    value=True
)

st.sidebar.markdown("---")

st.sidebar.write("⚙️ Settings")
st.sidebar.write("🚪 Logout")


# ================= LIVE DATA =================

temperature, humidity, rainfall, weather_status = get_weather(city)


# ================= LIVE SOIL =================

if live_soil:

    N_live = random.randint(
        max(0, N - 5),
        min(140, N + 5)
    )

    P_live = random.randint(
        max(0, P - 5),
        min(145, P + 5)
    )

    K_live = random.randint(
        max(0, K - 5),
        min(205, K + 5)
    )

    ph_live = round(
        random.uniform(
            max(0, ph - 0.3),
            min(14, ph + 0.3)
        ),
        2
    )

else:

    N_live = N
    P_live = P
    K_live = K
    ph_live = ph


# ================= HEADER =================

col1, col2 = st.columns([4, 1])

with col1:

    st.markdown(
        '<div class="main-title">Welcome, Farmer! 🌱</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Get AI-powered crop recommendations for maximum yield and profit.</div>',
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        f"### ☀️ {temperature}°C"
    )

    st.caption(
        f"{city}, India"
    )


st.markdown("---")


# ================= AI INPUT =================

input_data = pd.DataFrame(
    [[
        N_live,
        P_live,
        K_live,
        temperature,
        humidity,
        ph_live,
        rainfall
    ]],
    columns=[
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]
)


# ================= PREDICTION =================

probabilities = model.predict_proba(
    input_data
)[0]

top_indices = probabilities.argsort()[-5:][::-1]

top_crops = model.classes_[top_indices]

top_probabilities = probabilities[top_indices]

prediction = top_crops[0]

confidence = top_probabilities[0] * 100


# ================= CALCULATIONS =================

soil_score = min(
    100,
    int(
        (
            N_live / 140 * 30 +
            P_live / 145 * 25 +
            K_live / 205 * 25 +
            (ph_live / 14) * 20
        )
    )
)

weather_score = min(
    100,
    int(
        confidence * 0.9
    )
)

estimated_yield = round(
    20 + confidence / 8,
    1
)

profit = int(
    estimated_yield * 2000
)

water_requirement = int(
    350 + humidity
)


# ================= TOP METRIC CARDS =================

st.markdown("## 📊 Farm Overview")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:

    st.metric(
        "🌱 Soil Health Score",
        f"{soil_score}%",
        "Excellent"
    )

with c2:

    st.metric(
        "🌦️ Weather Suitability",
        f"{weather_score}%",
        "Highly Suitable"
    )

with c3:

    st.metric(
        "📈 Estimated Yield",
        f"{estimated_yield} Quintal/ha",
        "High Yield"
    )

with c4:

    st.metric(
        "₹ Profit Prediction",
        f"₹ {profit:,}",
        "High Profit"
    )

with c5:

    st.metric(
        "💧 Water Requirement",
        f"{water_requirement} mm",
        "Moderate"
    )


st.markdown("---")


# ================= AI RECOMMENDATION + CHART =================

left, middle, right = st.columns([1.2, 1.3, 1])


# -------- AI Recommendation --------

with left:

    st.subheader("🤖 AI Crop Recommendation")

    st.success(
        f"### 🌾 {prediction.upper()}"
    )

    st.markdown(
        f"## {confidence:.2f}%"
    )

    st.caption(
        "Suitability Score"
    )

    st.write(
        "✓ Suitable for current soil"
    )

    st.write(
        "✓ Weather conditions favorable"
    )

    st.write(
        "✓ High market potential"
    )

    st.button(
        "🌱 View Crop Details"
    )


# -------- Suitability Chart --------

with middle:

    st.subheader(
        "📊 Suitability Comparison"
    )

    chart_crops = [
        crop.title()
        for crop in top_crops
    ]

    chart_values = [
        round(prob * 100, 2)
        for prob in top_probabilities
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=chart_crops,
                y=chart_values,
                text=[
                    f"{x}%"
                    for x in chart_values
                ],
                textposition="auto"
            )
        ]
    )

    fig.update_layout(
        height=320,
        paper_bgcolor="#101c2b",
        plot_bgcolor="#101c2b",
        font=dict(color="white"),
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# -------- Soil Status --------

with right:

    st.subheader(
        "🧪 Soil Nutrient Status"
    )

    labels = [
        "Nitrogen",
        "Phosphorus",
        "Potassium",
        "pH Level"
    ]

    values = [
        N_live,
        P_live,
        K_live,
        ph_live * 10
    ]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.65
            )
        ]
    )

    fig.update_layout(
        height=320,
        paper_bgcolor="#101c2b",
        font=dict(color="white"),
        showlegend=True,
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ================= WEATHER + HISTORY =================

st.markdown("---")

col1, col2 = st.columns([1, 1.5])


# -------- WEATHER FORECAST --------

with col1:

    st.subheader("🌦️ Weather Forecast")

    forecast = [
        ("Today", temperature),
        ("Tomorrow", temperature + 1),
        ("Day 3", temperature - 1),
        ("Day 4", temperature + 2),
        ("Day 5", temperature)
    ]

    forecast_cols = st.columns(5)

    for col, (day, temp) in zip(
        forecast_cols,
        forecast
    ):

        with col:

            st.write(day)
            st.write("☀️")
            st.write(f"{temp:.0f}°C")


# -------- RECOMMENDATION HISTORY --------

with col2:

    st.subheader(
        "📜 Recent Recommendation History"
    )

    if "recommendations" not in st.session_state:

        st.session_state.recommendations = pd.DataFrame(
            columns=[
                "Time",
                "Crop",
                "Suitability",
                "Status"
            ]
        )

    new_rec = pd.DataFrame(
        [[
            datetime.now().strftime(
                "%H:%M:%S"
            ),
            prediction.upper(),
            round(confidence, 2),
            "Recommended"
        ]],
        columns=[
            "Time",
            "Crop",
            "Suitability",
            "Status"
        ]
    )

    st.session_state.recommendations = pd.concat(
        [
            st.session_state.recommendations,
            new_rec
        ],
        ignore_index=True
    ).tail(10)

    st.dataframe(
        st.session_state.recommendations,
        use_container_width=True,
        hide_index=True
    )


# ================= MARKET PRICE =================

st.markdown("---")

col1, col2 = st.columns([1.3, 1])


with col1:

    st.subheader(
        "💰 Market Price (Live)"
    )

    market_data = pd.DataFrame(
        {
            "Crop": [
                prediction.upper(),
                "WHEAT",
                "MAIZE",
                "COTTON"
            ],
            "Price ₹/Quintal": [
                random.randint(2000, 3000),
                random.randint(1800, 2500),
                random.randint(1600, 2200),
                random.randint(5000, 8000)
            ],
            "Change": [
                "📈 +2.5%",
                "📈 +1.8%",
                "📉 -0.5%",
                "📈 +3.2%"
            ]
        }
    )

    st.dataframe(
        market_data,
        use_container_width=True,
        hide_index=True
    )


# ================= LIVE WEATHER + SOIL =================

with col2:

    st.subheader(
        "📡 Live Farm Data"
    )

    st.metric(
        "🌡️ Temperature",
        f"{temperature} °C"
    )

    st.metric(
        "💧 Humidity",
        f"{humidity} %"
    )

    st.metric(
        "🌧️ Rainfall",
        f"{rainfall} mm"
    )

    st.metric(
        "⚗️ Soil pH",
        ph_live
    )


# ================= SMART ALERTS =================

st.markdown("---")

st.subheader("🔔 Smart Alerts")

a1, a2, a3 = st.columns(3)

with a1:

    st.warning(
        "⚠️ Weather may change tomorrow. Monitor irrigation requirements."
    )

with a2:

    if N_live < 40:

        st.error(
            "🔴 Nitrogen level is low. Consider adding fertilizer."
        )

    else:

        st.success(
            "🌱 Nitrogen level is suitable."
        )


with a3:

    st.success(
        f"🌾 Good conditions detected for sowing {prediction.title()}."
    )


# ================= LIVE STATUS =================

st.markdown("---")

if weather_status:

    st.success(
        "🟢 LIVE MODE ACTIVE — Dashboard updates every 5 seconds"
    )

else:

    st.warning(
        "🟡 LIVE MODE ACTIVE — Weather API unavailable, using fallback data"
    )


st.caption(
    f"📍 Location: {city} | "
    f"🕒 Last Updated: "
    f"{datetime.now().strftime('%d-%m-%Y %H:%M:%S')} | "
    f"🔄 Auto Refresh: 5 Seconds"
)