"""
FLABS - Fasciola Infection Suitability Index Calculator
Streamlit GUI Application

Kenya Nyandarua County - Kipipiri地域の感染好適指数計算アプリ

Streamlit Cloud デプロイ用設定:
1. Google Cloud Consoleでサービスアカウントを作成
2. Earth Engine APIを有効化
3. サービスアカウントをEarth Engineに登録
4. JSONキーをStreamlit Secretsに設定
"""

import streamlit as st
import ee
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json


# ページ設定
st.set_page_config(
    page_title="FLABS - Infection Index Calculator",
    page_icon="🦠",
    layout="wide"
)

# Google Earth Engine プロジェクトID
GEE_PROJECT_ID = 'ee-shohei-2'


@st.cache_resource
def initialize_earth_engine():
    """
    Google Earth Engineを初期化（キャッシュ）

    認証方法の優先順位:
    1. Streamlit Secrets（サービスアカウント）- Streamlit Cloud用
    2. 既存の認証情報（ローカル開発用）
    3. インタラクティブ認証（ローカル開発用）
    """
    # 方法1: Streamlit Secretsからサービスアカウント認証（Streamlit Cloud用）
    if 'gee_service_account' in st.secrets:
        try:
            # AttrDictを通常のdictに変換してからJSONに変換
            service_account_info = dict(st.secrets['gee_service_account'])
            service_account_email = service_account_info['client_email']

            # サービスアカウントの認証情報を作成
            credentials = ee.ServiceAccountCredentials(
                service_account_email,
                key_data=json.dumps(service_account_info)
            )
            ee.Initialize(credentials=credentials, project=GEE_PROJECT_ID)
            return True, "GEE initialized with service account"
        except Exception as e:
            return False, f"Service account authentication failed: {str(e)}"

    # 方法2: 既存の認証情報を使用（ローカル開発用）
    try:
        ee.Initialize(project=GEE_PROJECT_ID)
        return True, "GEE initialized with existing credentials"
    except Exception as e:
        pass

    # 方法3: インタラクティブ認証（ローカル開発用）
    try:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT_ID)
        return True, "GEE initialized after authentication"
    except Exception as auth_error:
        return False, f"GEE initialization failed: {str(auth_error)}"


def get_chirps_precipitation(start_date, end_date, lat, lon, buffer_radius):
    """CHIRPSから日次降水量データを取得"""
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
    """ERA5-Landから日次気温データを取得"""
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
    """感染好適指数を計算"""
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
            daily_idx = 0
            cumulative_value = 0
        else:
            if t_min < 7.2 and p_5sum >= 30.0 and t_avg >= 7.2:
                daily_idx = 2
            elif t_avg < 26.6 and t_min >= 7.2:
                base_idx = 0
                col_idx = 0
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
                else:
                    base_idx = 0

                daily_idx = base_idx

                if daily_idx == 0 and p_day >= 0.5 and t_avg >= 7.2:
                    daily_idx = 1

            cumulative_value += daily_idx

            if cumulative_value <= 5 and p_10sum == 0:
                cumulative_value = 0

        data.at[data.index[i], 'daily_index'] = daily_idx
        data.at[data.index[i], 'cumulative_index'] = cumulative_value

    return data


def create_time_series_plot(df):
    """時系列グラフを作成"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Cumulative Infection Index', 'Daily Infection Index',
                       'Temperature (°C)', 'Precipitation (mm)')
    )

    # 累積指数
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['cumulative_index'],
                   mode='lines', name='Cumulative Index',
                   line=dict(color='red', width=2)),
        row=1, col=1
    )

    # 日次指数
    fig.add_trace(
        go.Bar(x=df['date'], y=df['daily_index'],
               name='Daily Index', marker_color='orange'),
        row=2, col=1
    )

    # 気温
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

    # 降水量
    fig.add_trace(
        go.Bar(x=df['date'], y=df['precip'],
               name='Precipitation', marker_color='steelblue'),
        row=4, col=1
    )

    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def create_monthly_summary_plot(df):
    """月別サマリーグラフを作成"""
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


# メインアプリ
def main():
    st.title("🦠 FLABS - Fasciola Infection Suitability Index Calculator")
    st.markdown("""
    This application calculates the **Fasciola (Liver Fluke) infection suitability index**
    using satellite-derived weather data (CHIRPS precipitation + ERA5-Land temperature).
    """)

    # サイドバー - パラメータ設定
    with st.sidebar:
        st.header("⚙️ Parameters")

        st.subheader("📍 Location")

        # プリセット地点
        presets = {
            "Kipipiri, Kenya": (-0.4167, 36.5833),
            "Custom": None
        }
        selected_preset = st.selectbox("Select location", list(presets.keys()))

        if selected_preset == "Custom":
            lat = st.number_input("Latitude", value=-0.4167, format="%.4f")
            lon = st.number_input("Longitude", value=36.5833, format="%.4f")
        else:
            lat, lon = presets[selected_preset]
            st.info(f"Lat: {lat}, Lon: {lon}")

        buffer_radius = st.slider("Buffer radius (m)", 1000, 20000, 5000, 1000)

        st.subheader("📅 Analysis Period")

        # デフォルト期間（過去1年）
        default_end = datetime.now() - timedelta(days=7)
        default_start = default_end - timedelta(days=365)

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start date", value=default_start)
        with col2:
            end_date = st.date_input("End date", value=default_end)

        st.subheader("🛰️ Data Source")
        st.info("""
        - **Precipitation**: CHIRPS Daily
        - **Temperature**: ERA5-Land Daily
        """)

        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    # GEE初期化状態の確認
    gee_status, gee_message = initialize_earth_engine()

    if not gee_status:
        st.error(f"❌ Google Earth Engine initialization failed: {gee_message}")
        with st.expander("🔧 Setup Instructions"):
            st.markdown("""
            ### For Local Development:
            Run the following command in your terminal:
            ```bash
            earthengine authenticate
            ```

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

    # メインコンテンツ
    if run_analysis:
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        st.header("📊 Analysis Results")
        st.markdown(f"**Location**: ({lat}, {lon}) | **Period**: {start_str} to {end_str}")

        # プログレスバー
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # CHIRPSデータ取得
            status_text.text("Fetching CHIRPS precipitation data...")
            progress_bar.progress(10)
            df_precip = get_chirps_precipitation(start_str, end_str, lat, lon, buffer_radius)
            progress_bar.progress(40)

            # ERA5データ取得
            status_text.text("Fetching ERA5-Land temperature data...")
            df_temp = get_era5_temperature(start_str, end_str, lat, lon, buffer_radius)
            progress_bar.progress(70)

            # データ結合
            status_text.text("Merging data...")
            df = pd.merge(df_precip, df_temp, on='date', how='inner')
            progress_bar.progress(80)

            # 感染好適指数計算
            status_text.text("Calculating infection suitability index...")
            result = calculate_infection_index(df)
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")

            # 結果をセッションに保存
            st.session_state['result'] = result

        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            return

    # 結果の表示
    if 'result' in st.session_state:
        result = st.session_state['result']

        # サマリー統計
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

        # 時系列グラフ
        st.subheader("📉 Time Series")
        fig_ts = create_time_series_plot(result)
        st.plotly_chart(fig_ts, use_container_width=True)

        # 月別サマリー
        st.subheader("📅 Monthly Summary")
        fig_monthly = create_monthly_summary_plot(result)
        st.plotly_chart(fig_monthly, use_container_width=True)

        # データテーブル
        st.subheader("📋 Data Table")

        # 表示カラムの選択
        display_cols = ['date', 'temp_avg', 'temp_min', 'precip',
                       'precip_5days_sum', 'daily_index', 'cumulative_index']

        # 最新30日間のデータを表示
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

        # CSVダウンロード
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
        # 初期画面
        st.info("👈 Configure parameters in the sidebar and click **Run Analysis** to start.")

        # 使い方の説明
        with st.expander("📖 How to use this application"):
            st.markdown("""
            ### Steps:
            1. **Select Location**: Choose a preset location or enter custom coordinates
            2. **Set Analysis Period**: Select start and end dates
            3. **Run Analysis**: Click the "Run Analysis" button
            4. **View Results**: Explore the charts and data tables
            5. **Download**: Export results as CSV

            ### About the Infection Index:
            The Fasciola (Liver Fluke) infection suitability index is calculated based on:
            - **Temperature**: Average and minimum daily temperature
            - **Precipitation**: Daily and cumulative (5-day, 10-day) rainfall

            Higher cumulative index values indicate more favorable conditions for Fasciola transmission.

            ### Data Sources:
            - **CHIRPS**: Climate Hazards Group InfraRed Precipitation with Station data
            - **ERA5-Land**: ECMWF Reanalysis v5 Land component
            """)


if __name__ == "__main__":
    main()
