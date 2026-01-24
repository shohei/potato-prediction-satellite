"""
Kenya Nyandarua County - Kipipiri地域の感染好適指数計算プログラム

衛星データソース:
- 降水量: CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data)
- 気温: ERA5-Land (ECMWF Reanalysis v5)

使用方法:
1. Google Earth Engine認証が必要です
   pip install earthengine-api
   earthengine authenticate

2. スクリプトを実行
   python kipipiri_flabs.py
"""

import ee
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Kipipiri, Nyandarua County, Kenya の座標
# Kipipiriは山岳地域で、およその中心座標
KIPIPIRI_LAT = -0.4167
KIPIPIRI_LON = 36.5833

# 解析対象の半径（メートル）- 地域の代表値を取得するため
BUFFER_RADIUS = 5000  # 5km

# Google Earth Engine プロジェクトID
GEE_PROJECT_ID = 'ee-shohei-2'


def initialize_earth_engine():
    """Google Earth Engineを初期化"""
    try:
        ee.Initialize(project=GEE_PROJECT_ID)
        print(f"Google Earth Engine initialized successfully (project: {GEE_PROJECT_ID})")
    except Exception as e:
        print(f"Earth Engine initialization failed: {e}")
        print("Attempting authentication...")
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT_ID)
        print(f"Google Earth Engine initialized after authentication (project: {GEE_PROJECT_ID})")


def get_chirps_precipitation(start_date, end_date, lat, lon, buffer_radius):
    """
    CHIRPSから日次降水量データを取得

    Parameters:
    -----------
    start_date : str
        開始日 (YYYY-MM-DD形式)
    end_date : str
        終了日 (YYYY-MM-DD形式)
    lat : float
        緯度
    lon : float
        経度
    buffer_radius : int
        バッファ半径（メートル）

    Returns:
    --------
    pandas.DataFrame
        日付と降水量のデータフレーム
    """
    # 対象地点のジオメトリ作成
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(buffer_radius)

    # CHIRPS Daily: UCSB-CHG/CHIRPS/DAILY
    chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
        .filterDate(start_date, end_date) \
        .select('precipitation')

    def extract_precip(image):
        """各画像から地域平均降水量を抽出"""
        date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
        mean_precip = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=5566  # CHIRPSの解像度 (~0.05度)
        ).get('precipitation')

        return ee.Feature(None, {
            'date': date,
            'precip': mean_precip
        })

    # データ抽出
    precip_fc = chirps.map(extract_precip)
    precip_list = precip_fc.reduceColumns(
        ee.Reducer.toList(2), ['date', 'precip']
    ).get('list').getInfo()

    # DataFrameに変換
    df = pd.DataFrame(precip_list, columns=['date', 'precip'])
    df['date'] = pd.to_datetime(df['date'])
    df['precip'] = df['precip'].astype(float)

    return df.sort_values('date').reset_index(drop=True)


def get_era5_temperature(start_date, end_date, lat, lon, buffer_radius):
    """
    ERA5-Landから日次気温データ（平均・最低）を取得

    Parameters:
    -----------
    start_date : str
        開始日 (YYYY-MM-DD形式)
    end_date : str
        終了日 (YYYY-MM-DD形式)
    lat : float
        緯度
    lon : float
        経度
    buffer_radius : int
        バッファ半径（メートル）

    Returns:
    --------
    pandas.DataFrame
        日付、平均気温、最低気温のデータフレーム
    """
    # 対象地点のジオメトリ作成
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(buffer_radius)

    # ERA5-Land Daily Aggregated: 日次集計済みデータを使用
    era5_daily = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR') \
        .filterDate(start_date, end_date)

    def extract_temp(image):
        """各画像から気温データを抽出"""
        date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')

        stats = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=11132  # ERA5-Landの解像度 (~0.1度)
        )

        # temperature_2mが日平均、temperature_2m_minが日最低
        mean_temp = stats.get('temperature_2m')
        min_temp = stats.get('temperature_2m_min')

        return ee.Feature(None, {
            'date': date,
            'temp_avg_k': mean_temp,
            'temp_min_k': min_temp
        })

    # 各日のデータを取得
    temp_fc = era5_daily.map(extract_temp)
    temp_list = temp_fc.reduceColumns(
        ee.Reducer.toList(3), ['date', 'temp_avg_k', 'temp_min_k']
    ).get('list').getInfo()

    # DataFrameに変換
    df = pd.DataFrame(temp_list, columns=['date', 'temp_avg_k', 'temp_min_k'])
    df['date'] = pd.to_datetime(df['date'])

    # ケルビンから摂氏に変換
    df['temp_avg'] = df['temp_avg_k'].astype(float) - 273.15
    df['temp_min'] = df['temp_min_k'].astype(float) - 273.15

    return df[['date', 'temp_avg', 'temp_min']].sort_values('date').reset_index(drop=True)


def calculate_infection_index(df):
    """
    感染好適指数を計算
    (flabs_prediction.pyからの移植)

    必要なカラム:
    - 'temp_avg': 平均気温
    - 'temp_min': 最低気温
    - 'precip':   当日の降水量
    """
    data = df.copy()

    # 過去の降水量の合計を計算
    data['precip_5days_sum'] = data['precip'].shift(1).rolling(window=5, min_periods=1).sum().fillna(0)
    data['precip_10days_sum'] = data['precip'].shift(1).rolling(window=10, min_periods=1).sum().fillna(0)

    # 結果を格納するカラムを作成
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

        # ⑤ 平均気温が26.6℃以上の日はリセット
        if t_avg >= 26.6:
            daily_idx = 0
            cumulative_value = 0
        else:
            # ③ 例外ルール: 最低気温が7.2未満でも、前5日雨量>=30かつ平均>=7.2なら指数2
            if t_min < 7.2 and p_5sum >= 30.0 and t_avg >= 7.2:
                daily_idx = 2

            # ① 基本ルール: 平均気温26.6未満 かつ 最低気温7.2以上
            elif t_avg < 26.6 and t_min >= 7.2:
                base_idx = 0

                # 降水量区分の判定
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

                # 気温区分の判定
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

                # ② 微雨補正
                if daily_idx == 0 and p_day >= 0.5 and t_avg >= 7.2:
                    daily_idx = 1

            # 累積値の加算
            cumulative_value += daily_idx

            # ④ 累積値リセットルール
            if cumulative_value <= 5 and p_10sum == 0:
                cumulative_value = 0

        data.at[data.index[i], 'daily_index'] = daily_idx
        data.at[data.index[i], 'cumulative_index'] = cumulative_value

    return data


def fetch_weather_data(start_date, end_date, lat=KIPIPIRI_LAT, lon=KIPIPIRI_LON,
                       buffer_radius=BUFFER_RADIUS):
    """
    CHIRPSとERA5から気象データを取得して結合

    Parameters:
    -----------
    start_date : str
        開始日 (YYYY-MM-DD形式)
    end_date : str
        終了日 (YYYY-MM-DD形式)
    lat : float
        緯度 (デフォルト: Kipipiri)
    lon : float
        経度 (デフォルト: Kipipiri)
    buffer_radius : int
        バッファ半径（メートル）

    Returns:
    --------
    pandas.DataFrame
        結合された気象データ
    """
    print(f"Fetching data for location: ({lat}, {lon})")
    print(f"Period: {start_date} to {end_date}")

    # 降水量データ取得
    print("\nFetching CHIRPS precipitation data...")
    df_precip = get_chirps_precipitation(start_date, end_date, lat, lon, buffer_radius)
    print(f"  Retrieved {len(df_precip)} days of precipitation data")

    # 気温データ取得
    print("\nFetching ERA5-Land temperature data...")
    df_temp = get_era5_temperature(start_date, end_date, lat, lon, buffer_radius)
    print(f"  Retrieved {len(df_temp)} days of temperature data")

    # データを結合
    df = pd.merge(df_precip, df_temp, on='date', how='inner')
    print(f"\nMerged data: {len(df)} days")

    return df


def main():
    """メイン実行関数"""
    # Google Earth Engine初期化
    initialize_earth_engine()

    # 解析期間の設定（例: 過去1年間）
    end_date = datetime.now() - timedelta(days=2)  # データの遅延を考慮
    start_date = end_date - timedelta(days=365)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print("=" * 60)
    print("Kipipiri, Nyandarua County, Kenya - FLABS Analysis")
    print("=" * 60)
    print(f"Location: Lat {KIPIPIRI_LAT}, Lon {KIPIPIRI_LON}")
    print(f"Period: {start_str} to {end_str}")
    print("=" * 60)

    # 気象データ取得
    df = fetch_weather_data(start_str, end_str)

    # 感染好適指数の計算
    print("\nCalculating infection suitability index...")
    result = calculate_infection_index(df)

    # 結果の表示
    columns_to_show = ['date', 'temp_avg', 'temp_min', 'precip',
                       'precip_5days_sum', 'daily_index', 'cumulative_index']

    print("\n" + "=" * 60)
    print("Results (last 30 days):")
    print("=" * 60)
    print(result[columns_to_show].tail(30).to_string(index=False))

    # 統計サマリー
    print("\n" + "=" * 60)
    print("Summary Statistics:")
    print("=" * 60)
    print(f"Total days analyzed: {len(result)}")
    print(f"Days with infection index > 0: {(result['daily_index'] > 0).sum()}")
    print(f"Maximum cumulative index: {result['cumulative_index'].max()}")
    print(f"Average temperature: {result['temp_avg'].mean():.1f}°C")
    print(f"Total precipitation: {result['precip'].sum():.1f}mm")

    # CSVに保存
    output_file = 'kipipiri_flabs_results.csv'
    result.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    return result


if __name__ == '__main__':
    result = main()
