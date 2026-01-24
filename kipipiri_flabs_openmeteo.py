"""
Kenya Nyandarua County - Kipipiri地域の感染好適指数計算プログラム
（Open-Meteo API版 - 認証不要）

データソース:
- Open-Meteo API (ERA5データを含む)
  https://open-meteo.com/

使用方法:
pip install openmeteo-requests requests-cache pandas numpy
python kipipiri_flabs_openmeteo.py
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Kipipiri, Nyandarua County, Kenya の座標
KIPIPIRI_LAT = -0.4167
KIPIPIRI_LON = 36.5833


def get_weather_data_openmeteo(start_date, end_date, lat, lon):
    """
    Open-Meteo APIから気象データを取得

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

    Returns:
    --------
    pandas.DataFrame
        日付、降水量、平均気温、最低気温のデータフレーム
    """
    # Open-Meteo Archive API（ERA5データ）
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_mean", "temperature_2m_min", "precipitation_sum"],
        "timezone": "Africa/Nairobi"
    }

    print(f"Fetching data from Open-Meteo API...")
    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(f"API request failed: {response.status_code} - {response.text}")

    data = response.json()

    # DataFrameに変換
    df = pd.DataFrame({
        'date': pd.to_datetime(data['daily']['time']),
        'temp_avg': data['daily']['temperature_2m_mean'],
        'temp_min': data['daily']['temperature_2m_min'],
        'precip': data['daily']['precipitation_sum']
    })

    # 欠損値を0で埋める（降水量のみ）
    df['precip'] = df['precip'].fillna(0)

    return df


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

        # 欠損値チェック
        if pd.isna(t_avg) or pd.isna(t_min):
            data.at[data.index[i], 'daily_index'] = np.nan
            data.at[data.index[i], 'cumulative_index'] = cumulative_value
            continue

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


def main():
    """メイン実行関数"""
    # 解析期間の設定
    # Open-Meteo Archive APIは過去のデータのみ提供（約5日前まで）
    end_date = datetime.now() - timedelta(days=7)
    start_date = end_date - timedelta(days=365)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print("=" * 60)
    print("Kipipiri, Nyandarua County, Kenya - FLABS Analysis")
    print("(Using Open-Meteo API - ERA5 Reanalysis Data)")
    print("=" * 60)
    print(f"Location: Lat {KIPIPIRI_LAT}, Lon {KIPIPIRI_LON}")
    print(f"Period: {start_str} to {end_str}")
    print("=" * 60)

    # 気象データ取得
    df = get_weather_data_openmeteo(start_str, end_str, KIPIPIRI_LAT, KIPIPIRI_LON)
    print(f"Retrieved {len(df)} days of weather data")

    # データプレビュー
    print("\nWeather data preview (first 10 days):")
    print(df.head(10).to_string(index=False))

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

    # 月別サマリー
    print("\n" + "=" * 60)
    print("Monthly Summary:")
    print("=" * 60)
    result['month'] = result['date'].dt.to_period('M')
    monthly = result.groupby('month').agg({
        'temp_avg': 'mean',
        'precip': 'sum',
        'daily_index': 'sum',
        'cumulative_index': 'max'
    }).round(1)
    monthly.columns = ['Avg Temp (°C)', 'Total Precip (mm)', 'Sum Daily Index', 'Max Cumulative']
    print(monthly.to_string())

    # CSVに保存
    output_file = 'kipipiri_flabs_results.csv'
    result.drop(columns=['month'], inplace=True, errors='ignore')
    result.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    return result


if __name__ == '__main__':
    result = main()
