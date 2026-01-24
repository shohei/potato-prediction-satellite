import pandas as pd
import numpy as np

def calculate_infection_index(df):
    """
    データフレームを受け取り、以下の画像ルールに基づいて感染好適指数を計算する。
    
    必要なカラム:
    - 'temp_avg': 平均気温
    - 'temp_min': 最低気温
    - 'precip':   当日の降水量
    """
    
    # データのコピー（元のデータを変更しないため）
    data = df.copy()
    
    # ---------------------------------------------------------
    # 前準備: 過去の降水量の合計を計算
    # ---------------------------------------------------------
    # shift(1)を入れることで「当日を含まない」過去n日間とする
    # min_periods=1 は、データ欠損があっても計算可能な範囲で合計する設定
    data['precip_5days_sum'] = data['precip'].shift(1).rolling(window=5, min_periods=1).sum().fillna(0)
    data['precip_10days_sum'] = data['precip'].shift(1).rolling(window=10, min_periods=1).sum().fillna(0)

    # 結果を格納するカラムを作成
    data['daily_index'] = 0  # その日の指数
    data['cumulative_index'] = 0 # 累積値

    # ---------------------------------------------------------
    # 日ごとの計算ループ (累積値が前日の結果に依存するためループ処理)
    # ---------------------------------------------------------
    cumulative_value = 0
    
    for i in range(len(data)):
        row = data.iloc[i]
        
        t_avg = row['temp_avg']
        t_min = row['temp_min']
        p_day = row['precip']
        p_5sum = row['precip_5days_sum']
        p_10sum = row['precip_10days_sum']
        
        daily_idx = 0
        
        # === ルール判定ロジック ===
        
        # ⑤ 平均気温が26.6℃以上の日はリセット (指数計算の対象外とし、累積も0に戻す)
        if t_avg >= 26.6:
            daily_idx = 0
            cumulative_value = 0 # ルール⑤によるリセット
        else:
            # ③ 例外ルール: 最低気温が7.2未満でも、前5日雨量>=30かつ平均>=7.2なら指数2
            if t_min < 7.2 and p_5sum >= 30.0 and t_avg >= 7.2:
                daily_idx = 2
            
            # ① 基本ルール: 平均気温26.6未満 かつ 最低気温7.2以上
            elif t_avg < 26.6 and t_min >= 7.2:
                # 表に基づく判定
                base_idx = 0
                
                # 降水量区分の判定 (画像の区切り値を使用: <5, 5~10, 10.5~20, 20.5~25, 25.5~)
                # ※間の数値(10.1など)は上位の区分に含まれるよう境界を設定しています
                col_idx = 0
                if p_5sum < 5.0:
                    col_idx = 0
                elif p_5sum < 10.5: # 5.0 ~ 10.0
                    col_idx = 1
                elif p_5sum < 20.5: # 10.5 ~ 20.0
                    col_idx = 2
                elif p_5sum < 25.5: # 20.5 ~ 25.0
                    col_idx = 3
                else:               # 25.5 ~
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
                    # 平均気温が7.2未満の場合は定義なし（通常は0）
                    base_idx = 0
                
                daily_idx = base_idx

                # ② 微雨補正: 指数が0でも、当日0.5mm以上の雨があり、平均気温>=7.2なら指数1
                if daily_idx == 0 and p_day >= 0.5 and t_avg >= 7.2:
                    daily_idx = 1
            
            # 条件に当てはまらない場合（平均気温7.2未満で③も満たさないなど）は0のまま

            # 累積値の加算
            cumulative_value += daily_idx
            
            # ④ 累積値リセットルール
            # 累積値が5以下 でかつ 前10日間の雨量が0なら、累積値を0にする
            if cumulative_value <= 5 and p_10sum == 0:
                cumulative_value = 0

        # 結果をデータフレームに書き込み
        data.at[data.index[i], 'daily_index'] = daily_idx
        data.at[data.index[i], 'cumulative_index'] = cumulative_value

    return data

# ==========================================
# 実行テスト用データの作成
# ==========================================
# ランダムな気象データを作成してテストします
np.random.seed(42)
days = 30
df_sample = pd.DataFrame({
    'date': pd.date_range(start='2024-06-01', periods=days),
    'temp_avg': np.random.uniform(10, 28, days).round(1), # 10℃〜28℃
    'temp_min': np.random.uniform(5, 20, days).round(1),  # 5℃〜20℃
    'precip':   np.random.choice([0, 0, 0, 5, 10, 30], days) # 雨または晴れ
})

# 計算実行
result_df = calculate_infection_index(df_sample)

# 結果の表示（確認用）
# 見やすいように必要な列だけ表示
columns_to_show = ['date', 'temp_avg', 'temp_min', 'precip', 'precip_5days_sum', 'daily_index', 'cumulative_index']
print(result_df[columns_to_show].to_string())
