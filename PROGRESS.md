# 進捗ログ

このファイルは、現在のPredictive Policing再現実験の研究ノートです。以前はサブフォルダREADMEに分散していた方針もここへ統合しています。

添付・保存済み文献は参考資料として扱い、ユーザーの依頼文とは区別します。現時点では、予測精度を単一指標で評価することは保留し、逐次予測の挙動を確認できるGIFと、その再現に必要な予測表・設定表を残す方針です。

## 0. リポジトリ方針

### ディレクトリ構成

```text
datas/
  ローカルに準備したデータセット。Git管理外。
metrics/
  評価指標スクリプトを置く場所。出力は各モデルの results/metrics/ に保存する。
project_sources/
  参照文献・仕様書。
our_model/
  root直下に置く、STNPP-GATをベースにした本プロジェクト独自モデル。改良ごとにvolを分ける。
our_experiment/
  root直下に置く、our_modelの実験コードと結果。モデルvolに対応して管理する。
ref_models/
  既存研究の再現実装。
  models/
    参照モデルごとの実装。
  experiments/
    参照モデルごとの実験コードと結果。
README.md
  GitHub掲載用の概要。
PROGRESS.md
  データ、モデル、実験、限界の詳細ログ。
```

### データ管理方針

`datas/` は丸ごとGit管理外です。公開データであっても、raw chunk、処理済みParquet、空間グリッド、境界ファイルはサイズが大きく、再取得・再生成可能なローカル成果物として扱います。

データセットvariantは、都市、対象区域、犯罪種、期間、グリッドサイズ、予測ホライズンが変わるたびに別IDとして分けます。意図しない上書きを避けるため、フィルタや空間範囲が変わる場合は新しいdataset IDを作る方針です。

理想的なデータセット構成は以下です。

```text
datas/
  <city>/
    <dataset_id>/
      metadata.yml
      raw/
      interim/
      processed/
```

`metadata.yml` には、source URL、download timestamp、hash、フィルタ条件、座標参照系、グリッドサイズ、除外件数、処理後件数を記録します。

### 実験管理方針

`metrics/` は評価指標専用フォルダです。`our_model/` と `our_experiment/` は `metrics/` 配下ではなくroot直下に置き、今後STNPP-GATを改良していく本プロジェクト側の履歴として扱います。`our_model/vol1` を現在のSTNPP-GAT baselineとし、改良版は `vol2`, `vol3` のように増やします。対応する実験と結果は `our_experiment/vol*` に置きます。

`ref_models/` は既存研究の再現用です。参照モデル実装は `ref_models/models/<model_name>/`、その実験と結果は `ref_models/experiments/<model_name>/` に置きます。ETASはこの枠に移動しました。

各実験は、dataset ID、model ID、学習または履歴期間、予測ホライズン、予測期間、top-kや重点区域数、乱数seed、出力先を明示する必要があります。

`metrics/evaluate_current_results.py` は、現在のETAS、our_model vol1、our_model vol2の実験結果に対して、文献 `Predictive Policingモデルの数値実験で採用可能な評価指標` に挙げられた指標を可能な範囲で計算します。ただし、GIFや評価値を「モデルが社会的に有効である」証拠としては扱いません。介入効果、住民影響、feedback loopは別途検討が必要です。

## 1. LAPDデータセット

### 1.1 元データ

現在の主データは、LA City Open Dataで公開されているLAPDのlegacy crime dataです。

- `Crime Data from 2010 to 2019`
- `Crime Data from 2020 to 2024`
- LA City GIS LAPD division boundaries

ローカルデータセットIDは以下です。

- `lapd_legacy_2010_2024_all_crimes_grid300m_h168h`

このデータセットは、2つのSocrata APIから全件をページング取得し、LAPD境界と結合して作成しました。LAPD境界APIもページングが必要で、ページングなしでは最初の1000境界断片しか取得できず、`Olympic`、`Topanga`、`Mission` の境界の一部が欠落していました。現在の準備スクリプトではArcGIS endpointをページング取得し、区域ごとに境界断片をdissolveしてからセル割当しています。

### 1.2 全量データの処理結果

処理結果は以下です。

- raw行数: 3,138,031
- 処理後イベント数: 3,061,145
- 対象期間: 2010-01-01 00:01:00 から 2024-12-30 23:00:00
- LAPD区域数: 21
- 詳細犯罪コード数: 143
- 粗い犯罪グループ数: 10
- 座標系: source `EPSG:4326`、投影 `EPSG:3310`
- グリッド: 300m、14,659セル
- 境界面積: 約1,238,634,946 m2

除外・整形は以下を行いました。

- `DR_NO` 重複行の除去
- occurrence date と occurrence time から `occurred_at` を作成
- invalid datetime の除外
- invalid coordinate、`(0, 0)`、範囲外座標の除外
- LAPD division boundary外の点の除外
- 300mセルへの空間割当
- 事件の詳細犯罪名、粗い犯罪グループ、区域ID、区域名を保持

sourceごとの処理概要:

- `63jg-8b9z`
  - raw rows: 2,133,137
  - duplicate `DR_NO`: 57,808
  - invalid coordinate: 943
  - outside boundary: 11,461
  - processed rows: 2,062,925
- `2nrs-mtv8`
  - raw rows: 1,004,894
  - duplicate `DR_NO`: 0
  - invalid coordinate: 2,240
  - outside boundary: 4,433
  - processed rows: 998,221

### 1.3 全量イベントテーブルのフォーマット

処理済みイベントは以下に保存されています。

```text
datas/lapd_full/lapd_legacy_2010_2024_all_crimes_grid300m_h168h/processed/events.parquet
```

主な列は以下です。

- `event_id`: LAPDの事件ID。元データの `DR_NO` に相当。
- `occurred_at`: 発生日時。`DATE OCC` と `TIME OCC` から作成。
- `date_rptd`: 報告日時。将来、reporting delay分析に使えるよう保持。
- `crime_code`: LAPD crime code。
- `crime_type`: LAPD crime description。
- `crime_group`: 本プロジェクトで付与した粗い犯罪カテゴリ。
- `area_id`: LAPD area number。2桁文字列。
- `area_name`: LAPD area name。
- `rpt_dist_no`: reporting district。
- `part_1_2`: Part I / Part II 区分。
- `premis_cd`, `premis_desc`: 発生場所種別。
- `weapon_used_cd`, `weapon_desc`: 凶器情報。欠損あり。
- `status`, `status_desc`: 捜査状態。
- `lat`, `lon`: WGS84緯度経度。
- `x`, `y`: `EPSG:3310` に投影した座標。
- `cell_id`: 300mグリッドのセルID。
- `source_legacy_dataset`: どちらのSocrata datasetから来たか。

### 1.4 空間データのフォーマット

300mグリッドは以下です。

```text
datas/lapd_full/lapd_legacy_2010_2024_all_crimes_grid300m_h168h/processed/grid.geojson
```

主な列:

- `cell_id`: `r0138_c0122` のようなrow/col由来ID。
- `row`, `col`: グリッド上の行列番号。
- `area_m2`: セルの有効面積。境界でclipされたセルは小さくなる。
- `geometry`: セルpolygon。GeoJSONでは `EPSG:4326`。

LAPD区域境界は以下です。

```text
datas/lapd_full/lapd_legacy_2010_2024_all_crimes_grid300m_h168h/processed/division_boundaries.geojson
```

主な列:

- `area_id`
- `area_name`
- `geometry`

### 1.5 実験用150m対象犯罪データ

Mohler et al. のLA実験に寄せるため、全量データから対象犯罪のみを抽出し、150mセルへ再割当した派生データを作成しました。

- `lapd_legacy_2010_2024_mohler_target_crimes_grid150m_h24h`

保存先:

```text
datas/lapd_full/lapd_legacy_2010_2024_all_crimes_grid300m_h168h/derived/lapd_legacy_2010_2024_mohler_target_crimes_grid150m_h24h/
```

対象犯罪:

- `BURGLARY`
- `CAR_THEFT`
- `THEFT_FROM_VEHICLE`

処理結果:

- 対象犯罪イベント数: 919,284
- burglary: 217,369
- car theft: 279,996
- theft from vehicle: 421,919
- LAPD区域数: 21
- グリッド: 150m、56,927セル
- 期間: 2010-01-01 00:01:00 から 2024-12-30 18:30:00
- 投影座標系: `EPSG:3310`

対象犯罪へのマッピング:

- `BURGLARY` と `BURGLARY, ATTEMPTED` を `BURGLARY` とした。
- `VEHICLE - STOLEN`、`VEHICLE, STOLEN - OTHER...`、`VEHICLE - ATTEMPT STOLEN` などを `CAR_THEFT` とした。
- `BURGLARY FROM VEHICLE` と `THEFT FROM MOTOR VEHICLE` を `THEFT_FROM_VEHICLE` とした。

150m対象犯罪イベントテーブルの主な列:

- `event_id`
- `occurred_at`
- `crime_code`
- `crime_type`
- `crime_group`
- `source_area_id`, `source_area_name`: 元データ上の区域。
- `area_id`, `area_name`: 150mセル所属に基づく区域。
- `lat`, `lon`, `x`, `y`
- `target_crime`
- `row`, `col`, `cell_id`
- `source_legacy_dataset`

境界付近では、元データのareaとセル中心に基づくareaがずれることがあります。区域別モデルでは、セルごとの予測と整合させるため、150mセル所属に基づく `area_id` を使います。元データの区域は `source_area_id` として残しています。

### 1.6 退役したデータ

初期段階では以下も試しました。

- LAPDのFoothill、N Hollywood、Southwest小規模variant
- burglary単独variant
- vehicle stolen単独variant
- burglary + vehicle stolen のmulti-crime variant
- Chicago、Philadelphia

現在の実験結果では使っていません。今後、複数都市比較や区域別の小規模検証を再開する場合は、dataset IDを分けて再導入します。

## 2. 予測モデル

### 2.1 実装方針

モデル実装は、先行研究の再現と本プロジェクト独自モデルを分けて管理します。どちらもデータ取得やクリーニングには依存せず、標準化されたイベント表とグリッド表を受け取り、セルごとの予測リスクまたは期待件数を返すことを目指します。

モデル固有パラメータは、できるだけ実験スクリプト側で指定します。解釈と再現のため、実行時設定、学習設定、モデル状態、予測表を保存します。

現在の主なモデル実装:

- `ref_models/models/etas/adaptive_etas.py`
- `ref_models/models/etas/marked_adaptive_etas.py`
- `our_model/vol1/stnpp_gat.py`
- `our_model/vol2/etas_enhanced_stnpp_gat.py`

以前の週次Transformer v1スキャフォールドは現行構成から外しました。今後Transformer系を再開する場合は、STNPP-GATとの違いを明確にしたうえで、新しい `ref_models` または `our_model` のvolとして作り直します。

将来候補:

- adaptive SEPP variants
- spatial-kernel ETAS
- report-delay-aware adaptive ETAS
- observation-feedback simulation models
- HunchLab風のfeature-based gradient boosting model

### 2.2 ETAS / SEPP系モデルの概要

参照したMohler et al. のPredictive Policing実験では、犯罪を自己励起点過程として扱います。直感的には、犯罪リスクを以下の2成分で表します。

- 背景率 `mu`: その場所に長期的に存在する犯罪発生傾向。
- トリガー成分: 直近の事件が近い時空間で次の事件を誘発するnear-repeat効果。

本プロジェクトのETAS実装では、セル `i` の予測リスクを概ね以下として扱います。

```text
risk_i = background_i + trigger_i
background_i = mu_i * horizon_days
trigger_i = theta * z_i * (1 - exp(-omega * horizon_days))
```

`z_i` は過去イベントから作る自己励起状態で、時間が経つと `omega` に従って指数減衰します。イベントが起きると該当セルの `z_i` が増えます。

Mohler論文における重要設定:

- LAでは burglary, car theft, burglary-theft from vehicle が対象。
- 150m x 150m のprediction boxを使う。
- ETASは過去365日の犯罪データを入力にする。
- 実運用では1日1回、午前4時にETASパラメータを再推定する。
- 各division / shiftごとに限られた数のprediction boxを提示する想定。

本プロジェクトの現在のETAS再現実装:

- `ref_models/models/etas/adaptive_etas.py`
  - same-cell triggering
  - event-by-event state update
  - rolling 365-day background
  - theta / omega の制約付き尤度推定
  - branching-ratio的な背景率縮小を使う近似実装
- `ref_models/models/etas/marked_adaptive_etas.py`
  - `crime_type` をmarkとして保持できる拡張
  - crime-type別背景率
  - source-to-target crime-type transition matrix
  - 現行のMohler-style実験では総リスクGIFを優先するため、主に `adaptive_etas.py` を使う

現行実験のETASパラメータ:

- model ID: `M7_mohler_style_lapd_etas`
- cell size: 150m
- target crimes: `BURGLARY`, `CAR_THEFT`, `THEFT_FROM_VEHICLE`
- horizon: 24時間
- history window: 365日
- fitting unit: LAPD 21区域ごと
- initial theta: 0.35
- initial omega: 1 / 14日
- theta minimum: 0.01
- background alpha: 0.001
- parameter refit interval: 28日
- display frame interval: 14日

利用状況の想定:

- 分析官または実務担当者が、次の24時間の重点観察セルを確認する。
- 全域を一枚で俯瞰できるが、モデル推定は区域単位で行う。
- GIFは「モデルが動的に反応しているか」を観察するための可視化であり、巡回指示や介入効果の評価ではない。

### 2.3 STNPP-GAT系モデルの概要

参照した `2409.10882v2` は、spatio-temporal-network point processです。事件には時刻、場所、種類があり、さらに都市空間のnetwork構造とmark間の相互作用を考慮します。論文の中心は、mark-to-markの影響係数をGraph Attention Networkで学習する点です。

論文で確認した主要設定:

- GATでmark間のinteractionを学ぶ。
- attention heads `R` はcross-validationで選び、実験では `R=8`。
- MLE原理で点過程パラメータを推定する。
- 学習はSGDで、batch size `M=3`、learning rate `eta=1.0`、epochs `E=1500`。
- street-network distanceを使い、単純なEuclidean distanceより都市構造を反映する。

本プロジェクトの現在のSTNPP-GAT vol1実装:

- `our_model/vol1/stnpp_gat.py`
  - markを `target_crime x LAPD area` として定義
  - 3犯罪種 x 21区域 = 63 marks
  - multi-head GATでsource markからtarget markへの励起確率を学習
  - 学習された遷移行列は、source列ごとに1へ正規化
- `our_experiment/vol1/run_lapd_stnpp_gat.py`
  - 2010-2019の対象犯罪から同一150mセル内の近接遷移を集計
  - 経験的なsource-to-target mark transition matrixを作成
  - GATでその遷移行列を近似
  - 2020-2024の各フレームで、背景率 + marked self-excitation による24時間リスクを計算

現行実験のSTNPP-GATパラメータ:

- model ID: `T2_stnpp_gat_marked_point_process`
- marks: 63
- attention heads: 8
- hidden dimension: 64
- dropout: 0.05
- epochs: 1500
- learning rate: 1.0
- gradient clip: 5.0
- transition smoothing: 0.25
- empirical transition lookback: 30日
- theta: 0.35
- omega: 1 / 14日
- background alpha: 0.001
- training period: 2010-01-01 から 2020-01-01直前
- forecast period: 2020-01-01 から 2025-01-01直前
- display frame interval: 14日

利用状況の想定:

- ETASよりも、犯罪種と区域の相互作用を明示的に持たせたい場合に使う。
- どの犯罪種・区域が他の犯罪種・区域を誘発しやすいかを、mark transitionとして点検できる。
- 現状ではstreet networkがないため、完全なSTNPP再現というより、STNPPのGAT mark interaction部分を公開LAPDデータで動かす再現スキャフォールドである。

### 2.4 our_model vol2: ETAS-enhanced STNPP-GAT

`our_model/vol2/etas_enhanced_stnpp_gat.py` は、vol1のGAT mark interactionを維持しつつ、ETASから使いやすい部分を取り込んだ改良版です。

採用したETAS由来の要素:

- 区域ごとのrolling historyに基づく `theta` / `omega` の再推定。
- 短期的なnear-repeat効果を指数減衰状態として扱う。
- 背景率と自己励起成分を分け、自己励起の強さを区域ごとに変えられるようにする。

vol2では、GATは「どの犯罪種・区域markがどのmarkへ影響しやすいか」を表し、ETAS由来の区域別 `theta` / `omega` は「その時点で自己励起がどれくらい強いか」を表します。つまり、mark間の向きはSTNPP-GATから、時間的な強弱の再調整はETASから借りています。

現行実験のvol2パラメータ:

- model ID: `O2_etas_enhanced_stnpp_gat`
- marks: 63
- attention heads: 8
- hidden dimension: 64
- dropout: 0.05
- epochs: 1500
- learning rate: 1.0
- gradient clip: 5.0
- empirical transition lookback: 30日
- transition smoothing: 0.25
- ETAS refit interval: 28日
- ETAS initial theta: 0.35
- ETAS initial omega: 1 / 14日
- theta minimum: 0.01
- background alpha: 0.001
- forecast horizon: 24時間
- display frame interval: 14日

これは論文そのもののSTNPP完全再現ではありません。むしろ、今後の独自モデル開発で「GATによるmark interaction」と「ETASの逐次的な自己励起校正」を同時に扱うためのvol2です。

### 2.5 退役したTransformer v1

以前は、週次セルカウント列を入力する小型Transformer v1も試しました。

特徴:

- 過去52週の `log1p(count)` を時系列tokenとして入力
- learned positional embedding
- LAPD area embedding
- セル重心座標
- 季節特徴量
- weighted Poisson loss

ただし、このモデルはイベントレベルの連続時間点過程ではなく、犯罪種markも十分に扱わないため、現行構成からは外しました。今後使う場合は、STNPPとの違いを明示したbaselineとして扱います。

## 3. 実験

### 3.1 実験共通設定

現行実験は、同じ150m対象犯罪データに対して、参照モデルのETAS、our_model vol1のSTNPP-GAT、our_model vol2のETAS-enhanced STNPP-GATを比較できる形で実行しました。

共通データ:

- dataset ID: `lapd_legacy_2010_2024_mohler_target_crimes_grid150m_h24h`
- source dataset ID: `lapd_legacy_2010_2024_all_crimes_grid300m_h168h`
- target crimes: `BURGLARY`, `CAR_THEFT`, `THEFT_FROM_VEHICLE`
- cell size: 150m
- areas: LAPD 21区域
- forecast horizon: 24時間
- forecast display period: 2020-01-01 から 2024-12-26表示まで
- frame interval: 14日
- frames: 131

GIF表現:

- ヒートマップ: 24時間先の予測リスクまたは期待件数。
- 円: 実際に起こった事件。
- 円の色: `target_crime` ごと。
- 事件円は28日間のtrailとして表示し、時間が経つほど薄くなる。
- 地域ごとに学習・推定している場合でも、表示は全LAPD区域を1枚に統合する。

共通出力:

- `animation_config.yml`
- `forecast_frames.csv`
- `forecast_cell_risk.parquet`
- `forecast_top_cells.csv`
- `observed_events.parquet`
- GIF animation

### 3.2 Mohler-style ETAS実験

実行スクリプト:

```text
ref_models/experiments/etas/run_lapd_mohler_etas.py
```

出力先:

```text
ref_models/experiments/etas/results/
```

設定:

- model ID: `M7_mohler_style_lapd_etas`
- training/history window: 各forecast cutoff直前の365日
- forecast horizon: 24時間
- refit interval: 28日
- frame interval: 14日
- fitting unit: LAPD区域ごと
- display unit: LAPD全域統合
- initial theta: 0.35
- initial omega: 0.0714285714
- theta minimum: 0.01
- background alpha: 0.001
- top-k table: 20 cells per frame
- heatmap smoothing sigma: 2.1
- heatmap vmax quantile: 0.985
- heatmap gamma: 0.42
- GIF duration: 105ms per frame
- GIF dpi: 82

予測内容:

- 各フレームのforecast date時点で、区域ごとに直近365日の対象犯罪履歴を使う。
- ETASパラメータを区域単位で推定する。
- 各150mセルの24時間先の対象犯罪期待件数を計算する。
- 区域別に推定したセルリスクを、全LAPDの1枚の地図へ統合する。

保持している出力:

- `animations/M7_mohler_style_lapd_etas_24h_forecast_heatmap.gif`
- `tables/animation_config.yml`
- `tables/dataset_metadata.yml`
- `tables/forecast_frames.csv`
- `tables/forecast_cell_risk.parquet`
- `tables/forecast_top_cells.csv`
- `tables/etas_area_parameters.csv`
- `tables/observed_events.parquet`

確認結果:

- GIFフレーム数: 131
- GIFサイズ: 13,752,135 bytes
- `forecast_frames.csv`: 131行
- `forecast_top_cells.csv`: 2,620行
- `etas_area_parameters.csv`: 2,751行
- 予測期待件数の範囲: 138.67 から 198.92
- 表示フレーム上の観測事件合計: 23,166

### 3.3 STNPP-GAT実験

実行スクリプト:

```text
our_experiment/vol1/run_lapd_stnpp_gat.py
```

出力先:

```text
our_experiment/vol1/results/
```

設定:

- model ID: `T2_stnpp_gat_marked_point_process`
- train start: 2010-01-01
- train end: 2020-01-01
- forecast start: 2020-01-01
- forecast end: 2025-01-01
- forecast horizon: 24時間
- history window: 365日
- frame interval: 14日
- marks: `target_crime x LAPD area`
- number of marks: 63
- GAT attention heads: 8
- hidden dimension: 64
- dropout: 0.05
- epochs: 1500
- learning rate: 1.0
- gradient clip: 5.0
- empirical transition lookback: 30日
- transition smoothing: 0.25
- theta: 0.35
- omega: 0.0714285714
- background alpha: 0.001
- top-k table: 20 cells per frame

学習結果:

- train events: 593,806
- weighted pair count: 334,510
- transition KL loss: 3.119379 から 2.577063
- device: CPU
- PyTorch: 2.14.0+cpu

予測内容:

- 2010-2019の対象犯罪から、同一150mセル内で30日以内に続いたsource-target mark関係を重み付きで集計する。
- その経験的遷移行列をGATで近似する。
- 2020-2024のforecast cutoffごとに、直近365日の背景率と30日lookbackの自己励起状態を使う。
- 各セルの24時間先リスクを計算し、全LAPD区域を1枚のGIFへ統合する。

保持している出力:

- `animations/T2_stnpp_gat_marked_point_process_24h_forecast_heatmap.gif`
- `model/model_state.pt`
- `tables/animation_config.yml`
- `tables/training_summary.yml`
- `tables/forecast_frames.csv`
- `tables/forecast_cell_risk.parquet`
- `tables/forecast_top_cells.csv`
- `tables/learned_mark_transition.csv`
- `tables/observed_events.parquet`

確認結果:

- GIFフレーム数: 131
- GIFサイズ: 13,835,249 bytes
- `forecast_frames.csv`: 131行
- `forecast_top_cells.csv`: 2,620行
- `learned_mark_transition.csv`: 63 x 63 の遷移係数
- 予測期待件数の範囲: 143.25 から 205.99
- 表示フレーム上の観測事件合計: 23,166

### 3.4 our_model vol2実験

実行スクリプト:

```text
our_experiment/vol2/run_lapd_etas_enhanced_stnpp_gat.py
```

出力先:

```text
our_experiment/vol2/results/
```

設定:

- model ID: `O2_etas_enhanced_stnpp_gat`
- train start: 2010-01-01
- train end: 2020-01-01
- forecast start: 2020-01-01
- forecast end: 2025-01-01
- forecast horizon: 24時間
- history window: 365日
- frame interval: 14日
- marks: `target_crime x LAPD area`
- number of marks: 63
- GAT attention heads: 8
- hidden dimension: 64
- dropout: 0.05
- epochs: 1500
- learning rate: 1.0
- gradient clip: 5.0
- empirical transition lookback: 30日
- transition smoothing: 0.25
- ETAS refit interval: 28日
- initial theta: 0.35
- initial omega: 0.0714285714
- theta minimum: 0.01
- background alpha: 0.001
- top-k table: 20 cells per frame

予測内容:

- 2010-2019の対象犯罪から、vol1と同じ方法でGAT mark transitionを学習する。
- 2020-2024のforecast cutoffごとに、区域単位で直近365日のETAS `theta` / `omega` を再推定する。
- 各セルでは、背景率、GATによるmark間励起、区域ごとのETAS自己励起強度を組み合わせて24時間先リスクを計算する。
- 区域ごとの再推定を行っていても、表示は全LAPD区域を1枚のGIFへ統合する。

保持している出力:

- `animations/O2_etas_enhanced_stnpp_gat_24h_forecast_heatmap.gif`
- `model/model_state.pt`
- `tables/animation_config.yml`
- `tables/training_summary.yml`
- `tables/area_etas_parameters.csv`
- `tables/forecast_frames.csv`
- `tables/forecast_cell_risk.parquet`
- `tables/forecast_top_cells.csv`
- `tables/learned_mark_transition.csv`
- `tables/observed_events.parquet`

確認結果:

- GIFフレーム数: 131
- GIFサイズ: 13,764,150 bytes
- `forecast_frames.csv`: 131行
- `forecast_top_cells.csv`: 2,620行
- `area_etas_parameters.csv`: 2,751行
- `learned_mark_transition.csv`: 63 x 63 の遷移係数
- GAT transition KL loss: 3.119379 から 2.577063
- 予測期待件数の範囲: 141.70 から 201.57
- 表示フレーム上の観測事件合計: 23,166

### 3.5 退役した実験

以下は退役または参考用です。

- 固定パラメータ・週次・全犯罪の `ETAS_full`
- 週次セル系列の `Transformer_v1`
- 3区域だけを使った `ETAS_v1` / `ETAS_v2`
- 静的診断プロット
- metric CSVやbacktest summary

今回のファイル整理では、古い結果を保持するよりも、参照モデル再現と本プロジェクト独自モデルの改良履歴を分けることを優先しました。

### 3.6 評価指標実験

追加文献 `Predictive Policingモデルの数値実験で採用可能な評価指標` に基づき、現在の3実験に対して評価指標をまとめて計算しました。

実行スクリプト:

```text
metrics/evaluate_current_results.py
```

対象モデル:

- `Reference ETAS`
- `Our vol1 STNPP-GAT`
- `Our vol2 ETAS-enhanced STNPP-GAT`

主な出力:

各モデルの `results/metrics/` に、モデル単位の評価表と単体プロットを保存します。モデル横断の比較プロットは生成せず、比較が必要な場合は各モデルの `summary_metrics.csv` を後から読み込んで行います。

- `ref_models/experiments/etas/results/metrics/`
- `our_experiment/vol1/results/metrics/`
- `our_experiment/vol2/results/metrics/`

各ディレクトリに含める主な出力:

- `summary_metrics.csv`
- `metric_applicability.csv`
- `hotspot_frame_metrics.csv`
- `classification_frame_metrics.csv`
- `count_probability_frame_metrics.csv`
- `calibration_bins.csv`
- `cumulative_intensity.csv`
- `area_proxy_frame_metrics.csv`
- `plots/*.png`

計算した指標:

- Hotspot / ranking: Hit Count, Hit Rate, Miss Rate, Area Coverage, Hotspot Density, PAI, RRI, PEI*, Gain@q, Lift@q, Precision@k, Hit-rate curve, PAI curve, Area Under Hitrate Curve, IoU/Jaccard, Dice, Centroid Distance
- Binary classification: Accuracy, Recall/TPR, Specificity/TNR, Precision/PPV, FPR, FNR, F1, Balanced Accuracy, MCC, ROC-AUC, PR-AUC
- Count / rate: MAE, RMSE, Poisson Deviance, RMSLE, MASE, R2, Pearson, Spearman, Residual Moran's I
- Probability: Log Loss, Brier Score, Calibration Curve, ECE, O/E ratio
- Point-process proxy: binned Poisson test log-likelihood, N(t) and Lambda(t), cumulative intensity error
- Area proxy fairness/burden: Allocation Share Gap, Allocation-to-Crime Ratio Gap, Subgroup FPR/FNR Gap, Subgroup Calibration, Allocation Gini, Bias Amplification Slope

現状で計算しなかった指標:

- AIC / BIC: 全モデルで比較可能なtrain log-likelihoodとパラメータ数を保存していない。
- Time-rescaling KS: 現在の保存結果は14日間隔の24時間予測スナップショットであり、イベント間の連続時間積分強度を持っていない。
- Next-event Time MAE / Location MAE: 次イベントの時刻・位置を直接出力していない。
- Type Macro-F1: 保存済み予測は対象犯罪全体の総リスクであり、犯罪種別の予測ラベルではない。

主な確認結果:

- Hit Rate@5% は、ETAS 0.399、vol1 0.402、vol2 0.401。
- PAI@5% は、ETAS 7.98、vol1 8.05、vol2 8.01。
- PR-AUC は、ETAS 0.0316、vol1 0.0316、vol2 0.0313。
- Brier Score は3モデルとも約0.00297。
- binned Poisson log-likelihood/event は、ETAS -6.239、vol1 -6.122、vol2 -6.121。
- 累積強度RMSEは、ETAS 278.7、vol1 370.9、vol2 219.7。

PEI* は今回の24時間窓ではHit Rateとほぼ同じ値になります。これは、1%面積でも実現犯罪セル数に対して十分大きく、事後的な最良領域がほぼ全犯罪を含められるためです。より厳しいPEI*比較をしたい場合は、より小さい面積率、短いセルサイズ、または犯罪種別・時間帯別の評価が必要です。

## 4. 限界と未確認事項

### 4.1 データの限界

LAPD legacy dataは公開履歴データであり、リアルタイム通報データではありません。実運用の予測では、事件の発生時刻、報告時刻、データベース登録時刻の遅れが重要ですが、現在はreport delayを明示的にモデル化していません。

住所情報は公開元でhundred block単位に匿名化されています。したがって、緯度経度は実際の正確な発生地点ではなく、公開用に丸められた位置である可能性があります。150mセルを使っていても、空間精度には限界があります。

2010-2024のlegacy dataを対象にしており、NIBRS-eraの新しいフォーマットはまだ統合していません。将来的に近年データへ拡張する場合、crime codeやcrime descriptionの対応表を再確認する必要があります。

境界外点、無効座標、重複行は除外しています。これはモデル入力を安定させるために必要ですが、除外された事件の地理的・時間的偏りはまだ評価していません。

### 4.2 ETAS再現の限界

Mohler論文では、ETASパラメータを1日1回午前4時に再推定し、現場では限られた数の150m boxを提示する運用でした。現在のGIF実験では、可視化サイズと計算量を抑えるため、24時間予測を14日間隔で表示し、再推定は28日間隔にしています。

現在のETAS推定は、論文のEM推定を完全には再実装していません。制約付き尤度とbranching-ratio的な背景率縮小で近似しています。そのため、論文と同じパラメータ推定過程とは言えません。

空間トリガーは同一セル中心です。隣接セルや距離減衰の空間核を明示的に入れていないため、近隣ブロックへの波及は十分に表現できません。

パラメータ推定は区域単位ですが、区域境界をまたぐ自己励起は扱っていません。実際の犯罪連鎖が区域境界を越える場合、境界近くの予測は過小または不自然になる可能性があります。

### 4.3 STNPP-GAT再現の限界

2409.10882v2のSTNPPはstreet-network distanceを重要な構成要素とします。現在のデータ準備には道路ネットワークが含まれていないため、同一150mセル内の近接遷移で代替しています。これは都市ネットワーク上の距離を使うSTNPPとは異なります。

論文では連続時間点過程のlog-likelihoodをMLE / SGDで最適化します。現在の実装は、経験的なmark transition matrixをGATで近似する形で、完全なevent-sequence NLLではありません。

論文のbatch size `M=3` はイベント系列単位のSGDを前提にしています。現在はmark graphが63ノードと小さいため、full transition matrixを直接最適化しています。設定値としてepochs=1500、learning rate=1.0、attention heads=8は寄せていますが、学習対象は論文と同一ではありません。

markは `crime type x LAPD area` にしています。論文のmark設計と同一とは限らず、より細かいlocation markやstreet-network node markを導入すると結果は変わる可能性があります。

vol2はETASの逐次的な自己励起校正を取り入れていますが、STNPP本体の完全なlikelihood推定やstreet-network distance問題を解決したわけではありません。現時点では、ETAS由来の適応性をour_modelへ取り込むための最初の開発版です。

### 4.4 実験・評価の限界

現在はGIFとオフライン評価指標を併用しています。GIFはモデル挙動を観察するには有用ですが、予測性能や社会的妥当性を単独で判断するものではありません。評価指標も、観測犯罪データへの当てはまりを測るものであり、犯罪抑止効果や住民影響を直接測るものではありません。

まだ弱い評価:

- AIC / BICやtime-rescaling KSのような、連続時間点過程としての厳密な適合診断。
- 次イベントの時刻・位置・犯罪種を直接予測する指標。
- 犯罪種別に分けた予測性能。現行保存結果は総リスク中心。
- demographic属性に基づく公平性評価。現状はLAPD areaを代理群として使った負担評価に留まる。
- analyst baselineやpatrol allocationを仮定した実運用比較。
- 不確実性区間やbootstrap 95%信頼区間。

また、Predictive Policingの有効性は、犯罪予測の当たり外れだけでは決まりません。警察活動の配置、住民への影響、既存の通報・取締りバイアス、地域負担の偏り、feedback loopを含めて検討する必要があります。

### 4.5 計算・公開上の限界

`datas/` はGit管理外であり、GitHubには公開していません。したがって、第三者が完全再現するには、LA City Open Dataから同じデータを再取得し、ローカル準備スクリプトを実行する必要があります。

予測表やモデル状態も大きいため、多くはGit管理外です。現在GitHubに公開する想定の成果物は、主にREADME類、実験コード、参照文献、選定したGIFです。

CPU実行を前提にしているため、論文通りの毎日再推定や完全な連続時間NLLを全期間・全セルで行うには計算量が大きくなります。今後、GPU利用、空間インデックス最適化、期間分割、キャッシュ設計が必要です。

## 5. 今後の課題

優先度が高いもの:

- ETASのEM推定をより論文に忠実に実装する。
- ETASに隣接セルまたは距離減衰の空間核を入れる。
- STNPP-GATに道路ネットワーク距離を導入する。
- STNPP-GATを経験的遷移行列近似ではなく、連続時間event-sequence NLLで学習する。
- vol2を検証し、空間核、report delay、network distanceのいずれかを取り込んだvol3を作る。
- report delayを考慮した実運用風のデータ利用時点を再現する。
- 評価メトリックにbootstrap 95%信頼区間を追加する。
- AIC/BIC、time-rescaling KS、next-event評価に必要なモデル出力を保存する。
- 予測の地域負担や集中度を、人口・属性データと接続して評価する。
- ChicagoやPhiladelphiaなど他都市データを再導入し、同じモデルを比較する。

保留中の評価観点:

- 時間的な適応性
- 実運用上の巡回面積や重点区域数
- 予測の持続性と反応性
- 注意の過度な集中
- 事件が少ない週やゼロ件日の扱い
- 地域ごとの負担の偏り
- 不確実性とキャリブレーションの信頼区間
- 現実的な分析官・巡回ベースラインとの比較
