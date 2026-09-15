# 進捗ログ

このファイルには、準備したデータ、実装したモデル、実行した実験、残している結果を記録します。

Predictive Policing が「うまく機能しているか」を単純な指標で判断するのは難しいため、現時点では評価メトリックの設計は保留しています。現在は、逐次予測の挙動を確認できるGIFと、その再現に必要な予測表・設定表を残す方針です。

## 2026-09-14

### リポジトリ構成

以下の構成を作成しました。

- `datas/`
- `models/`
- `experiments/`
- `results/`
- `project_sources/`

初期段階ではLAPDの小規模データ、Chicago、Philadelphiaも試しましたが、現在はLAPDの全量レガシーデータに焦点を移しています。`datas/` はGit管理外にしました。

## 2026-09-15

### 参照文献

以下の実験仕様書を参照しました。

- `project_sources/0c338802-a3c7-4e4c-a35a-fa94a78d6948_Predictive_Policingモデル再現実験：公開データ・実運用アルゴリズム・実行手順書.pdf`

この文書は参考資料として扱い、ユーザーからの依頼文とは区別しています。

また、将来のSTNPP / Transformer型点過程モデルの検討用に以下の文献を追加しました。

- `project_sources/2409.10882v2.pdf`

### 現在の実装

現在残している主な実装は以下です。

- `models/adaptive_etas.py`
- `models/marked_adaptive_etas.py`
- `models/temporal_attention_transformer.py`
- `experiments/common.py`
- `experiments/animate_adaptive_forecast.py`
- `experiments/animate_etas_v2_forecast.py`
- `experiments/run_lapd_full_etas.py`
- `experiments/run_lapd_transformer_v1.py`
- `datas/lapd_full/prepare_lapd_legacy_full.py`

`datas/lapd_full/prepare_lapd_legacy_full.py` はローカル専用スクリプトで、`datas/` ごとGit管理外です。

削除または退役させたものは以下です。

- PhiladelphiaとChicagoのローカルデータ・結果
- 以前のLAPD小規模区域データ・結果
- 以前のETAS_v1 / ETAS_v2結果フォルダ
- 静的診断プロット
- 評価メトリックCSVやバックテスト出力

### LAPD全量レガシーデータ

現在のローカルデータセットは以下です。

- `lapd_legacy_2010_2024_all_crimes_grid300m_h168h`

元データは以下です。

- LA City Open Data `Crime Data from 2010 to 2019`
- LA City Open Data `Crime Data from 2020 to 2024`
- LA City GIS LAPD division boundaries

処理内容は以下です。

- Socrataのraw行数: 3,138,031
- 処理後イベント数: 3,061,145
- LAPD区域数: 21
- 詳細犯罪コード数: 143
- 粗い犯罪グループ数: 10
- グリッド: 300m、14,659セル
- 予測ホライズン: 168時間
- `crime_code`, `crime_type`, `crime_group`, `area_id`, `area_name` を保持

重要な修正点:

- LAPD境界APIはページングが必要でした。ページングなしでは最初の1000境界断片しか取れず、`Olympic`、`Topanga`、`Mission` の多くが落ちていました。
- 現在の準備スクリプトではArcGIS endpointをページング取得し、区域ごとに境界断片をdissolveしてからセル割当をしています。

## ETAS実験

現在のETAS結果は以下です。

- `results/ETAS_full/lapd_legacy_2010_2024_pooled_etas_weekly/`

モデル表示名:

- `M6_pooled_lapd_weekly_online_etas_fixed_theta`

設定:

- 予測期間: 2010-01-01 から 2025-01-10 表示終了まで
- フレーム数: 784
- フレーム間隔: 7日
- 予測ホライズン: 168時間
- 背景率の履歴窓: 365日
- theta: 0.35
- omega: 1 / 14日
- 事件表示のフェード窓: 4週間
- ヒートマップ: 21区域を一括学習した共通赤スケール
- 観測事件: `crime_group` ごとの色付きリング

2026-09-15の更新:

- ヒートマップが薄く見えたため、赤系カラーマップを強めました。
- `vmax_quantile` を 0.985、`heatmap_gamma` を 0.4 に変更しました。
- GIFを再生成しました。

保持している出力:

- `animations/M6_pooled_lapd_weekly_online_etas_fixed_theta_weekly_forecast_heatmap.gif`
- `tables/animation_config.yml`
- `tables/forecast_frames.csv`
- `tables/forecast_cell_risk.parquet`
- `tables/forecast_top_cells.csv`
- `tables/forecast_group_risk.csv`
- `tables/observed_events.parquet`

検証:

- GIFフレーム数: 784
- GIFサイズ: 98,348,895 bytes
- `forecast_frames.csv`: 784行
- `forecast_cell_risk.parquet`: 11,492,656行
- `forecast_top_cells.csv`: 78,400行
- `forecast_group_risk.csv`: 7,840行
- `observed_events.parquet`: 3,061,145行

## Transformer v1実験

最初のPyTorch Transformer実験を実装・実行しました。

- `models/temporal_attention_transformer.py`
- `experiments/run_lapd_transformer_v1.py`
- `results/Transformer_v1/lapd_legacy_2020_2024_transformer_v1_weekly/`

モデル表示名:

- `T1_temporal_attention_transformer_v1`

設計:

- 2010-01-01から2020-01-03までの週次セル履歴で学習
- 2020-01-03から2025-01-10表示終了までを週次予測
- 各セルの過去52週の `log1p(count)` を時系列トークンとして入力
- learned positional embeddingを使用
- LAPD区域embeddingを使用
- セル重心の正規化座標を使用
- 予測週の季節特徴量を使用
- Poisson lossで次週のセル別期待件数を出力

学習設定:

- PyTorch: 2.14.0+cpu
- 学習サンプル数: 250,000
- positiveサンプル数: 125,000
- context window: 52週
- model width: 32
- attention heads: 4
- transformer layers: 1
- epochs: 3
- weighted Poisson loss: 0.43958, 0.38270, 0.37940

2026-09-15の更新:

- ETASと同じく、ヒートマップが薄く見えたため赤系カラーマップを強めました。
- `vmax_quantile` を 0.985、`heatmap_gamma` を 0.4 に変更しました。
- GIFを再生成しました。

保持している出力:

- `animations/T1_temporal_attention_transformer_v1_weekly_forecast_heatmap.gif`
- `model/model_state.pt`
- `tables/animation_config.yml`
- `tables/training_summary.yml`
- `tables/forecast_frames.csv`
- `tables/forecast_cell_risk.parquet`
- `tables/forecast_top_cells.csv`
- `tables/forecast_group_risk.csv`
- `tables/observed_events.parquet`

検証:

- GIFフレーム数: 262
- GIFサイズ: 31,786,068 bytes
- `forecast_frames.csv`: 262行
- `forecast_cell_risk.parquet`: 3,840,658行
- `forecast_top_cells.csv`: 26,200行
- `forecast_group_risk.csv`: 2,620行
- `observed_events.parquet`: 996,535行

注意:

- 最後の表示週は2024年末以降まで伸びるため、観測事件数が不完全になります。これはGIFの連続表示のために残しており、評価メトリックとしては扱っていません。

## 今後の課題

評価メトリックの設計は保留中です。今後検討する候補は以下です。

- 時間的な適応性
- 実運用上の巡回面積や重点区域数
- 予測の持続性と反応性
- 注意の過度な集中
- 事件が少ない週やゼロ件週の扱い
- 地域ごとの負担の偏り
- 不確実性とキャリブレーション
- 現実的な分析官・巡回ベースラインとの比較
