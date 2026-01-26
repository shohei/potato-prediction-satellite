"""
Potato Disease & Yield Prediction System
Integrated Streamlit Application

Features:
1. FLABS - Potato Infection Suitability Index Calculator
2. LINTUL-POTATO-DSS - Potato Potential Yield Prediction Model
"""

import streamlit as st
import ee
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import requests


# Page configuration
st.set_page_config(
    page_title="Potato Disease & Yield Prediction",
    page_icon="🥔",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
/* Risk date highlight boxes */
.risk-date-box {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    text-align: center;
}
.prediction-date-box {
    background: linear-gradient(135deg, #ffa500 0%, #ff8c00 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    text-align: center;
}
/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    padding-left: 20px;
    padding-right: 20px;
}
</style>
""", unsafe_allow_html=True)

# Google Earth Engine Project ID
GEE_PROJECT_ID = 'ee-shohei-2'

# Default coordinates: Kipipiri, Kenya
DEFAULT_LAT = -0.4167
DEFAULT_LON = 36.5833


# =============================================================================
# FLABS Functions
# =============================================================================

def _has_streamlit_secrets():
    """Check if Streamlit secrets are available (for Cloud deployment)"""
    try:
        return 'gee_service_account' in st.secrets
    except Exception:
        return False


@st.cache_resource
def initialize_earth_engine():
    """
    Initialize Google Earth Engine (cached)

    Authentication priority:
    1. Streamlit Secrets (service account) - for Streamlit Cloud
    2. Existing credentials - for local development (Mac)
    3. Interactive authentication - for local development (Mac)
    """
    # Method 1: Streamlit Secrets service account authentication (for Streamlit Cloud)
    if _has_streamlit_secrets():
        try:
            service_account_info = dict(st.secrets['gee_service_account'])
            service_account_email = service_account_info['client_email']
            credentials = ee.ServiceAccountCredentials(
                service_account_email,
                key_data=json.dumps(service_account_info)
            )
            ee.Initialize(credentials=credentials, project=GEE_PROJECT_ID)
            return True, "GEE initialized with service account (Cloud)"
        except Exception as e:
            return False, f"Service account authentication failed: {str(e)}"

    # Method 2: Existing credentials (for local development on Mac)
    try:
        ee.Initialize(project=GEE_PROJECT_ID)
        return True, "GEE initialized with existing credentials (Local)"
    except Exception:
        pass

    # Method 3: Interactive authentication (for local development on Mac)
    try:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT_ID)
        return True, "GEE initialized after authentication (Local)"
    except Exception as auth_error:
        return False, f"GEE initialization failed: {str(auth_error)}"


def get_chirps_precipitation(start_date, end_date, lat, lon, buffer_radius):
    """Get daily precipitation data from CHIRPS"""
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(buffer_radius)

    chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
        .filterDate(start_date, end_date) \
        .select('precipitation')

    def extract_precip(image):
        date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
        mean_precip = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=5566
        ).get('precipitation')
        return ee.Feature(None, {'date': date, 'precip': mean_precip})

    precip_fc = chirps.map(extract_precip)
    precip_list = precip_fc.reduceColumns(
        ee.Reducer.toList(2), ['date', 'precip']
    ).get('list').getInfo()

    df = pd.DataFrame(precip_list, columns=['date', 'precip'])
    df['date'] = pd.to_datetime(df['date'])
    df['precip'] = df['precip'].astype(float)
    return df.sort_values('date').reset_index(drop=True)


def get_era5_temperature(start_date, end_date, lat, lon, buffer_radius):
    """Get daily temperature data from ERA5-Land"""
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(buffer_radius)

    era5_daily = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR') \
        .filterDate(start_date, end_date)

    def extract_temp(image):
        date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
        stats = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=11132
        )
        mean_temp = stats.get('temperature_2m')
        min_temp = stats.get('temperature_2m_min')
        return ee.Feature(None, {
            'date': date,
            'temp_avg_k': mean_temp,
            'temp_min_k': min_temp
        })

    temp_fc = era5_daily.map(extract_temp)
    temp_list = temp_fc.reduceColumns(
        ee.Reducer.toList(3), ['date', 'temp_avg_k', 'temp_min_k']
    ).get('list').getInfo()

    df = pd.DataFrame(temp_list, columns=['date', 'temp_avg_k', 'temp_min_k'])
    df['date'] = pd.to_datetime(df['date'])
    df['temp_avg'] = df['temp_avg_k'].astype(float) - 273.15
    df['temp_min'] = df['temp_min_k'].astype(float) - 273.15
    return df[['date', 'temp_avg', 'temp_min']].sort_values('date').reset_index(drop=True)


def calculate_infection_index(df):
    """Calculate infection suitability index"""
    data = df.copy()

    data['precip_5days_sum'] = data['precip'].shift(1).rolling(window=5, min_periods=1).sum().fillna(0)
    data['precip_10days_sum'] = data['precip'].shift(1).rolling(window=10, min_periods=1).sum().fillna(0)

    data['daily_index'] = 0
    data['cumulative_index'] = 0

    cumulative_value = 0

    for i in range(len(data)):
        row = data.iloc[i]
        t_avg = row['temp_avg']
        t_min = row['temp_min']
        p_day = row['precip']
        p_5sum = row['precip_5days_sum']
        p_10sum = row['precip_10days_sum']

        daily_idx = 0

        if pd.isna(t_avg) or pd.isna(t_min):
            data.at[data.index[i], 'daily_index'] = np.nan
            data.at[data.index[i], 'cumulative_index'] = cumulative_value
            continue

        if t_avg >= 26.6:
            cumulative_value = 0
            daily_idx = 0
        else:
            if t_min < 7.2 and p_5sum >= 30.0 and t_avg >= 7.2:
                daily_idx = 2
            elif t_min >= 7.2:
                base_idx = 0

                if p_5sum < 5.0:
                    col_idx = 0
                elif p_5sum < 10.5:
                    col_idx = 1
                elif p_5sum < 20.5:
                    col_idx = 2
                elif p_5sum < 25.5:
                    col_idx = 3
                else:
                    col_idx = 4

                if 15.1 <= t_avg <= 26.5:
                    scores = [0, 1, 2, 2, 3]
                    base_idx = scores[col_idx]
                elif 11.7 <= t_avg <= 15.0:
                    scores = [0, 0, 1, 2, 2]
                    base_idx = scores[col_idx]
                elif 7.2 <= t_avg <= 11.6:
                    scores = [0, 0, 0, 2, 2]
                    base_idx = scores[col_idx]

                daily_idx = base_idx

                if daily_idx == 0 and p_day >= 0.5 and t_avg >= 7.2:
                    daily_idx = 1

            cumulative_value += daily_idx

            if cumulative_value <= 5 and p_10sum == 0:
                cumulative_value = 0

        data.at[data.index[i], 'daily_index'] = daily_idx
        data.at[data.index[i], 'cumulative_index'] = cumulative_value

    return data


def find_risk_dates(df, threshold=21):
    """Find risk threshold date and predicted onset date"""
    risk_rows = df[df['cumulative_index'] >= threshold]

    if len(risk_rows) > 0:
        risk_date = risk_rows.iloc[0]['date']
        prediction_date = risk_date + timedelta(days=14)
        return risk_date, prediction_date

    return None, None


def create_flabs_time_series_plot(df, risk_date=None, prediction_date=None):
    """Create time series plot for FLABS"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Cumulative Infection Index', 'Daily Infection Index',
                       'Temperature (°C)', 'Precipitation (mm)')
    )

    fig.add_trace(
        go.Scatter(x=df['date'], y=df['cumulative_index'],
                   mode='lines', name='Cumulative Index',
                   line=dict(color='red', width=2)),
        row=1, col=1
    )

    fig.add_hline(y=21, line_dash="dash", line_color="green",
                  annotation_text="Threshold (21)", row=1, col=1)

    fig.add_trace(
        go.Bar(x=df['date'], y=df['daily_index'],
               name='Daily Index', marker_color='orange'),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=df['date'], y=df['temp_avg'],
                   mode='lines', name='Avg Temp',
                   line=dict(color='blue')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['temp_min'],
                   mode='lines', name='Min Temp',
                   line=dict(color='lightblue')),
        row=3, col=1
    )

    fig.add_trace(
        go.Bar(x=df['date'], y=df['precip'],
               name='Precipitation', marker_color='steelblue'),
        row=4, col=1
    )

    if risk_date is not None:
        risk_date_str = risk_date.strftime('%Y-%m-%d') if hasattr(risk_date, 'strftime') else str(risk_date)
        for row in range(1, 5):
            fig.add_vline(
                x=risk_date_str, line_dash="solid", line_color="red", line_width=2,
                row=row, col=1
            )
        fig.add_annotation(
            x=risk_date_str, y=1, yref="y domain",
            text="Risk Date", showarrow=False,
            font=dict(color="red", size=10),
            xanchor="left", yanchor="top",
            row=1, col=1
        )

    if prediction_date is not None:
        prediction_date_str = prediction_date.strftime('%Y-%m-%d') if hasattr(prediction_date, 'strftime') else str(prediction_date)
        if prediction_date <= df['date'].max():
            for row in range(1, 5):
                fig.add_vline(
                    x=prediction_date_str, line_dash="dash", line_color="orange", line_width=2,
                    row=row, col=1
                )
            fig.add_annotation(
                x=prediction_date_str, y=0.85, yref="y domain",
                text="Predicted Onset", showarrow=False,
                font=dict(color="orange", size=10),
                xanchor="left", yanchor="top",
                row=1, col=1
            )

    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def create_monthly_summary_plot(df):
    """Create monthly summary plot"""
    df_monthly = df.copy()
    df_monthly['month'] = df_monthly['date'].dt.to_period('M').astype(str)

    monthly = df_monthly.groupby('month').agg({
        'temp_avg': 'mean',
        'precip': 'sum',
        'daily_index': 'sum',
        'cumulative_index': 'max'
    }).reset_index()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Monthly Precipitation & Index Sum', 'Monthly Max Cumulative Index')
    )

    fig.add_trace(
        go.Bar(x=monthly['month'], y=monthly['precip'],
               name='Precipitation (mm)', marker_color='steelblue'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=monthly['month'], y=monthly['daily_index'],
                   mode='lines+markers', name='Sum Daily Index',
                   line=dict(color='orange'), yaxis='y2'),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(x=monthly['month'], y=monthly['cumulative_index'],
               name='Max Cumulative', marker_color='red'),
        row=1, col=2
    )

    fig.update_layout(
        height=400,
        showlegend=True
    )

    return fig


# =============================================================================
# LINTUL-POTATO-DSS Functions
# =============================================================================

def fetch_nasa_power_data(latitude: float, longitude: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch weather data from NASA POWER API"""
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"

    params = {
        "parameters": "T2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN,T2M_MAX,T2M_MIN",
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": start_date.replace("-", ""),
        "end": end_date.replace("-", ""),
        "format": "JSON"
    }

    try:
        response = requests.get(base_url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        parameters = data["properties"]["parameter"]
        dates = list(parameters["T2M"].keys())

        evapotranspiration = []
        for d in dates:
            t_mean = parameters["T2M"][d]
            t_max = parameters["T2M_MAX"][d]
            t_min = parameters["T2M_MIN"][d]
            solar_rad = parameters["ALLSKY_SFC_SW_DWN"][d]

            if t_max > t_min and t_mean > -17.8 and solar_rad > 0:
                et0 = 0.0023 * solar_rad * (t_mean + 17.8) * ((t_max - t_min) ** 0.5)
                evapotranspiration.append(max(0, et0))
            else:
                evapotranspiration.append(0)

        df = pd.DataFrame({
            "date": pd.to_datetime(dates, format="%Y%m%d"),
            "temperature": [parameters["T2M"][d] for d in dates],
            "precipitation": [parameters["PRECTOTCORR"][d] for d in dates],
            "solar_radiation": [parameters["ALLSKY_SFC_SW_DWN"][d] for d in dates],
            "evapotranspiration": evapotranspiration
        })

        df = df.replace(-999, np.nan)
        df = df.interpolate(method='linear')
        df = df.bfill().ffill()

        return df

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from NASA POWER API: {e}")
        return None


def calculate_sprout_elongation_rate(temperature: float) -> float:
    """Calculate daily sprout elongation rate (mm/day)"""
    return temperature * 0.7


def estimate_emergence_date(planting_date: datetime, planting_depth: float, temperatures: pd.Series, dates: pd.Series) -> tuple:
    """Estimate emergence date based on planting depth and sprout elongation rate"""
    accumulated_length = 0.0
    days = 0

    for i, (temp, date) in enumerate(zip(temperatures, dates)):
        if date < pd.Timestamp(planting_date):
            continue

        if temp > 0:
            elongation_rate = calculate_sprout_elongation_rate(temp)
            accumulated_length += elongation_rate

        days += 1

        if accumulated_length >= planting_depth:
            return date, days

    return None, days


def calculate_light_interception_rate(accumulated_temp: float) -> float:
    """Calculate light interception rate (%)"""
    rate = (accumulated_temp / 650) * 100
    return min(rate, 100.0)


def calculate_light_use_efficiency(temperature: float) -> float:
    """Calculate light use efficiency (g/MJ) based on temperature"""
    if temperature < 3:
        return 0.0
    elif 3 <= temperature < 15:
        return 0.104167 * temperature - 0.312501
    elif 15 <= temperature < 20:
        return 1.25
    elif 20 <= temperature < 28:
        return -0.15625 * temperature + 4.375
    else:
        return 0.0


def calculate_daily_dry_matter(light_efficiency: float, solar_radiation: float, light_interception_rate: float) -> float:
    """Calculate daily total dry matter production (t/ha)"""
    dry_matter_g_m2 = light_efficiency * solar_radiation * (light_interception_rate / 100)
    dry_matter_t_ha = dry_matter_g_m2 * 0.01
    return dry_matter_t_ha


def calculate_tuber_dry_matter(total_dry_matter: float) -> float:
    """Calculate tuber dry matter yield (t/ha)"""
    return total_dry_matter * 0.75


def calculate_fresh_weight_yield(tuber_dry_matter: float) -> float:
    """Calculate fresh weight yield (t/ha)"""
    return tuber_dry_matter / 0.2


def run_lintul_model(weather_df: pd.DataFrame, planting_date: datetime, harvest_date: datetime, planting_depth: float) -> dict:
    """Run the complete LINTUL-POTATO-DSS model"""
    results = {
        "daily_data": [],
        "emergence_date": None,
        "days_to_emergence": 0,
        "total_dry_matter": 0.0,
        "tuber_dry_matter": 0.0,
        "fresh_weight_yield": 0.0
    }

    emergence_date, days_to_emergence = estimate_emergence_date(
        planting_date,
        planting_depth,
        weather_df["temperature"],
        weather_df["date"]
    )

    if emergence_date is None:
        st.warning("Emergence did not occur within the specified period.")
        return results

    results["emergence_date"] = emergence_date
    results["days_to_emergence"] = days_to_emergence

    accumulated_temp = 0.0
    total_dry_matter = 0.0

    for _, row in weather_df.iterrows():
        date = row["date"]
        temp = row["temperature"]
        solar_rad = row["solar_radiation"]

        daily_record = {
            "date": date,
            "temperature": temp,
            "solar_radiation": solar_rad,
            "accumulated_temp": 0.0,
            "light_interception_rate": 0.0,
            "light_use_efficiency": 0.0,
            "daily_dry_matter": 0.0,
            "cumulative_dry_matter": 0.0
        }

        if date >= pd.Timestamp(emergence_date) and date <= pd.Timestamp(harvest_date):
            if temp > 0:
                accumulated_temp += temp

            light_interception = calculate_light_interception_rate(accumulated_temp)
            light_efficiency = calculate_light_use_efficiency(temp)
            daily_dm = calculate_daily_dry_matter(light_efficiency, solar_rad, light_interception)
            total_dry_matter += daily_dm

            daily_record["accumulated_temp"] = accumulated_temp
            daily_record["light_interception_rate"] = light_interception
            daily_record["light_use_efficiency"] = light_efficiency
            daily_record["daily_dry_matter"] = daily_dm
            daily_record["cumulative_dry_matter"] = total_dry_matter

        results["daily_data"].append(daily_record)

    results["total_dry_matter"] = total_dry_matter
    results["tuber_dry_matter"] = calculate_tuber_dry_matter(total_dry_matter)
    results["fresh_weight_yield"] = calculate_fresh_weight_yield(results["tuber_dry_matter"])

    return results


def create_lintul_results_charts(results: dict) -> go.Figure:
    """Create visualization charts for LINTUL model results"""
    daily_df = pd.DataFrame(results["daily_data"])

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Daily Temperature (°C)",
            "Solar Radiation (MJ/m²/day)",
            "Accumulated Temperature (°C)",
            "Light Interception Rate (%)",
            "Light Use Efficiency (g/MJ)",
            "Cumulative Dry Matter (t/ha)"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    fig.add_trace(
        go.Scatter(x=daily_df["date"], y=daily_df["temperature"],
                   mode="lines", name="Temperature", line=dict(color="red")),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=daily_df["date"], y=daily_df["solar_radiation"],
                   mode="lines", name="Solar Radiation", line=dict(color="orange")),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=daily_df["date"], y=daily_df["accumulated_temp"],
                   mode="lines", name="Accumulated Temp", line=dict(color="darkred")),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=daily_df["date"], y=daily_df["light_interception_rate"],
                   mode="lines", name="Light Interception", line=dict(color="green")),
        row=2, col=2
    )

    fig.add_trace(
        go.Scatter(x=daily_df["date"], y=daily_df["light_use_efficiency"],
                   mode="lines", name="Light Use Efficiency", line=dict(color="blue")),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(x=daily_df["date"], y=daily_df["cumulative_dry_matter"],
                   mode="lines", name="Cumulative DM", line=dict(color="brown")),
        row=3, col=2
    )

    fig.update_layout(
        height=800,
        showlegend=False
    )

    return fig


# =============================================================================
# Main Application
# =============================================================================

def run_flabs_tab():
    """FLABS Tab Content"""
    st.markdown("""
    This tool calculates the potato infection suitability index
    using satellite-derived weather data (CHIRPS precipitation + ERA5-Land temperature).
    """)

    # Sidebar parameters for FLABS
    with st.sidebar:
        st.header("⚙️ FLABS Parameters")

        st.subheader("📍 Location")
        presets = {
            "Kipipiri, Kenya": (-0.4167, 36.5833),
            "Custom": None
        }
        selected_preset = st.selectbox("Select location", list(presets.keys()), key="flabs_preset")

        if selected_preset == "Custom":
            lat = st.number_input("Latitude", value=-0.4167, format="%.4f", key="flabs_lat")
            lon = st.number_input("Longitude", value=36.5833, format="%.4f", key="flabs_lon")
        else:
            lat, lon = presets[selected_preset]
            st.info(f"Lat: {lat}, Lon: {lon}")

        buffer_radius = st.slider("Buffer radius (m)", 1000, 20000, 5000, 1000, key="flabs_buffer")

        st.subheader("📅 Analysis Period")
        default_end = datetime.now() - timedelta(days=7)
        default_start = default_end - timedelta(days=365)

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start date", value=default_start, key="flabs_start")
        with col2:
            end_date = st.date_input("End date", value=default_end, key="flabs_end")

        st.subheader("🛰️ Data Source")
        st.info("""
        - **Precipitation**: CHIRPS Daily
        - **Temperature**: ERA5-Land Daily
        """)

        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True, key="flabs_run")

    # GEE initialization
    gee_status, gee_message = initialize_earth_engine()

    if not gee_status:
        st.error(f"❌ Google Earth Engine initialization failed: {gee_message}")
        with st.expander("🔧 Setup Instructions"):
            st.markdown("""
            ### For Local Development (Mac):
            Run the following command in your terminal:
            ```bash
            earthengine authenticate
            ```
            Then restart this application.

            ### For Streamlit Cloud:
            1. Create a service account in [Google Cloud Console](https://console.cloud.google.com/)
            2. Enable the Earth Engine API
            3. Register the service account at [Earth Engine](https://code.earthengine.google.com/)
            4. Download the JSON key and add it to Streamlit Secrets:

            In your Streamlit Cloud app settings, add the following to **Secrets**:
            ```toml
            [gee_service_account]
            type = "service_account"
            project_id = "your-project-id"
            private_key_id = "your-private-key-id"
            private_key = "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"
            client_email = "your-service-account@your-project.iam.gserviceaccount.com"
            client_id = "your-client-id"
            auth_uri = "https://accounts.google.com/o/oauth2/auth"
            token_uri = "https://oauth2.googleapis.com/token"
            auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
            client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account"
            ```
            """)
        return

    # Run analysis
    if run_analysis:
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        st.header("📊 Analysis Results")
        st.markdown(f"**Location**: ({lat}, {lon}) | **Period**: {start_str} to {end_str}")

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("Fetching CHIRPS precipitation data...")
            progress_bar.progress(10)
            df_precip = get_chirps_precipitation(start_str, end_str, lat, lon, buffer_radius)
            progress_bar.progress(40)

            status_text.text("Fetching ERA5-Land temperature data...")
            df_temp = get_era5_temperature(start_str, end_str, lat, lon, buffer_radius)
            progress_bar.progress(70)

            status_text.text("Merging data...")
            df = pd.merge(df_precip, df_temp, on='date', how='inner')
            progress_bar.progress(80)

            status_text.text("Calculating infection suitability index...")
            result = calculate_infection_index(df)
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")

            st.session_state['flabs_result'] = result

        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            return

    # Display results
    if 'flabs_result' in st.session_state:
        result = st.session_state['flabs_result']
        risk_date, prediction_date = find_risk_dates(result, threshold=21)

        if risk_date is not None:
            st.subheader("🚨 Risk Alert")
            col_risk1, col_risk2 = st.columns(2)
            with col_risk1:
                st.markdown(f"""
                <div class="risk-date-box">
                    <h3 style="margin:0;">Risk Threshold Date</h3>
                    <h2 style="margin:0.5rem 0;">{risk_date.strftime('%Y-%m-%d')}</h2>
                    <p style="margin:0;font-size:0.9em;">Cumulative index reached 21</p>
                </div>
                """, unsafe_allow_html=True)
            with col_risk2:
                st.markdown(f"""
                <div class="prediction-date-box">
                    <h3 style="margin:0;">Predicted Onset Date</h3>
                    <h2 style="margin:0.5rem 0;">{prediction_date.strftime('%Y-%m-%d')}</h2>
                    <p style="margin:0;font-size:0.9em;">2 weeks after risk threshold</p>
                </div>
                """, unsafe_allow_html=True)

        st.subheader("📈 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Days", len(result))
        with col2:
            st.metric("Days with Index > 0", int((result['daily_index'] > 0).sum()))
        with col3:
            st.metric("Max Cumulative Index", int(result['cumulative_index'].max()))
        with col4:
            st.metric("Total Precipitation", f"{result['precip'].sum():.1f} mm")

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("Avg Temperature", f"{result['temp_avg'].mean():.1f} °C")
        with col6:
            st.metric("Min Temperature", f"{result['temp_min'].min():.1f} °C")
        with col7:
            st.metric("Max Temperature", f"{result['temp_avg'].max():.1f} °C")
        with col8:
            risk_level = "High" if result['cumulative_index'].max() > 200 else "Medium" if result['cumulative_index'].max() > 100 else "Low"
            st.metric("Risk Level", risk_level)

        st.subheader("📉 Time Series")
        fig_ts = create_flabs_time_series_plot(result, risk_date, prediction_date)
        st.plotly_chart(fig_ts, use_container_width=True)

        st.subheader("📅 Monthly Summary")
        fig_monthly = create_monthly_summary_plot(result)
        st.plotly_chart(fig_monthly, use_container_width=True)

        st.subheader("📋 Data Table")
        display_cols = ['date', 'temp_avg', 'temp_min', 'precip',
                       'precip_5days_sum', 'daily_index', 'cumulative_index']
        st.dataframe(
            result[display_cols].tail(30).style.format({
                'temp_avg': '{:.1f}',
                'temp_min': '{:.1f}',
                'precip': '{:.1f}',
                'precip_5days_sum': '{:.1f}',
                'daily_index': '{:.0f}',
                'cumulative_index': '{:.0f}'
            }),
            use_container_width=True
        )

        st.subheader("💾 Download Results")
        csv = result.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="flabs_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    else:
        st.info("👈 Configure parameters in the sidebar and click **Run Analysis** to start.")

        with st.expander("📱 Parameters (for Mobile)", expanded=False):
            st.markdown("*Use this section on mobile devices.*")
            # Mobile parameters would go here (simplified for brevity)

        with st.expander("📖 About FLABS"):
            st.markdown("""
            ### About the Infection Index
            The potato infection suitability index is calculated based on:
            - **Temperature**: Average and minimum daily temperature
            - **Precipitation**: Daily and cumulative (5-day, 10-day) rainfall

            Higher cumulative index values indicate more favorable conditions for infection transmission.
            """)


def run_lintul_tab():
    """LINTUL-POTATO-DSS Tab Content"""
    st.markdown("""
    This tool predicts **potato potential yield** using the LINTUL-POTATO-DSS model.
    Weather data is automatically fetched from NASA POWER satellite data.

    **Note:** NASA POWER provides historical weather data only. Please select dates from the past.
    """)

    # Sidebar parameters for LINTUL
    with st.sidebar:
        st.header("⚙️ LINTUL Parameters")

        st.subheader("📍 Location")
        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input("Latitude", value=DEFAULT_LAT, format="%.4f",
                                       min_value=-90.0, max_value=90.0, key="lintul_lat")
        with col2:
            longitude = st.number_input("Longitude", value=DEFAULT_LON, format="%.4f",
                                        min_value=-180.0, max_value=180.0, key="lintul_lon")
        st.caption("Default: Kipipiri, Kenya")

        st.subheader("🌱 Crop Management")
        planting_depth = st.number_input(
            "Planting Depth (mm)", value=100, min_value=10, max_value=300, step=10,
            key="lintul_depth"
        )

        default_planting = datetime(2024, 3, 1)
        default_harvest = datetime(2024, 6, 29)

        planting_date = st.date_input("Planting Date", value=default_planting, key="lintul_plant")
        harvest_date = st.date_input("Harvest Date", value=default_harvest, key="lintul_harvest")

        if harvest_date <= planting_date:
            st.error("Harvest date must be after planting date!")
            return

        st.subheader("🌡️ Weather Adjustments")
        temp_offset = st.number_input(
            "Temperature Offset (°C)", value=0.0, min_value=-20.0, max_value=20.0,
            step=0.5, format="%.1f", key="lintul_temp_offset"
        )
        solar_offset = st.number_input(
            "Solar Radiation Offset (MJ/m²)", value=0.0, min_value=-20.0, max_value=20.0,
            step=0.5, format="%.1f", key="lintul_solar_offset"
        )

        run_model = st.button("🚀 Run Model", type="primary", use_container_width=True, key="lintul_run")

    # Run model
    if run_model:
        start_date = datetime.combine(planting_date, datetime.min.time()).strftime("%Y-%m-%d")
        end_date = datetime.combine(harvest_date, datetime.min.time()).strftime("%Y-%m-%d")

        with st.spinner("Fetching weather data from NASA POWER API..."):
            weather_df = fetch_nasa_power_data(latitude, longitude, start_date, end_date)

        if weather_df is None or weather_df.empty:
            st.error("Failed to fetch weather data. Please check your inputs and try again.")
            return

        if temp_offset != 0.0:
            weather_df["temperature"] = weather_df["temperature"] + temp_offset
        if solar_offset != 0.0:
            weather_df["solar_radiation"] = (weather_df["solar_radiation"] + solar_offset).clip(lower=0)

        st.success(f"Successfully fetched {len(weather_df)} days of weather data!")

        with st.spinner("Running LINTUL-POTATO-DSS model..."):
            results = run_lintul_model(
                weather_df,
                datetime.combine(planting_date, datetime.min.time()),
                datetime.combine(harvest_date, datetime.min.time()),
                planting_depth
            )

        st.session_state['lintul_results'] = results
        st.session_state['lintul_weather'] = weather_df

    # Display results
    if 'lintul_results' in st.session_state:
        results = st.session_state['lintul_results']
        weather_df = st.session_state['lintul_weather']

        st.header("📊 Model Results")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="Emergence Date",
                value=results["emergence_date"].strftime("%Y-%m-%d") if results["emergence_date"] else "N/A"
            )
        with col2:
            st.metric(label="Days to Emergence", value=f"{results['days_to_emergence']} days")
        with col3:
            st.metric(label="Total Dry Matter", value=f"{results['total_dry_matter']:.2f} t/ha")
        with col4:
            st.metric(label="Fresh Weight Yield", value=f"{results['fresh_weight_yield']:.2f} t/ha")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Tuber Dry Matter", value=f"{results['tuber_dry_matter']:.2f} t/ha")
        with col2:
            growth_period = len(weather_df)
            st.metric(label="Growth Period", value=f"{growth_period} days")

        st.header("📈 Time Series Analysis")
        fig = create_lintul_results_charts(results)
        st.plotly_chart(fig, use_container_width=True)

        st.header("🌤️ Weather Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Temperature", f"{weather_df['temperature'].mean():.1f} °C")
        with col2:
            st.metric("Total Precipitation", f"{weather_df['precipitation'].sum():.1f} mm")
        with col3:
            st.metric("Avg Solar Radiation", f"{weather_df['solar_radiation'].mean():.2f} MJ/m²/day")
        with col4:
            st.metric("Total Evapotranspiration", f"{weather_df['evapotranspiration'].sum():.1f} mm")

        with st.expander("View Raw Weather Data"):
            st.dataframe(weather_df, use_container_width=True)

        with st.expander("View Daily Model Calculations"):
            daily_df = pd.DataFrame(results["daily_data"])
            st.dataframe(daily_df, use_container_width=True)

    else:
        st.info("👈 Configure parameters in the sidebar and click **Run Model** to start.")

        with st.expander("ℹ️ About LINTUL-POTATO-DSS Model"):
            st.markdown("""
            ### Model Algorithm

            **1. Sprout Elongation Rate**
            - Daily sprout elongation (mm/day) = Daily average temperature × 0.7 mm

            **2. Emergence Date Estimation**
            - Calculated from planting depth and cumulative sprout elongation

            **3. Light Interception Rate**
            - Rate (%) = Accumulated temperature from emergence / 650 × 100 (max 100%)

            **4. Light Use Efficiency (η)**
            - Varies with temperature (0-1.25 g/MJ)

            **5. Yield Calculations**
            - Total DM = η × Solar radiation × Light interception
            - Tuber DM = Total DM × 0.75
            - Fresh Weight = Tuber DM / 0.2

            ### Data Source
            Weather data from [NASA POWER](https://power.larc.nasa.gov/)
            """)


def main():
    st.title("🥔 Potato Disease & Yield Prediction from Satellite Data")

    # Create tabs
    tab1, tab2 = st.tabs(["🦠 FLABS - Infection Index", "🌱 LINTUL - Yield Prediction"])

    with tab1:
        run_flabs_tab()

    with tab2:
        run_lintul_tab()


def show_footer():
    """Display footer"""
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #888; font-size: 0.8em;">
        © 2026 Shohei Aoki. All rights reserved.<br>
        <span style="font-size: 0.9em;">Jomo Kenyatta University of Agriculture and Technology</span><br>
        <span style="font-size: 0.85em;">Contact: aoki [at] jkuat.ac.ke</span><br>
        <a href="https://github.com/shohei/flabs-satellite" target="_blank" style="color: #888;">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
    show_footer()
