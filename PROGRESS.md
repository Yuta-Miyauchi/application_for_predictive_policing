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

`metrics/evaluate_current_results.py` は、現在のETASとour_model vol1からvol5の実験結果に対して、文献 `Predictive Policingモデルの数値実験で採用可能な評価指標` に挙げられた指標を可能な範囲で計算します。ただし、GIFや評価値を「モデルが社会的に有効である」証拠としては扱いません。介入効果、住民影響、feedback loopは別途検討が必要です。

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

### 1.6 STMGNN-ZINB / HCL / ST-MoGE用3km日次データ

文献 `2408.04193v1` の実験条件に合わせ、150m対象犯罪データから次の派生データを作成しました。HCLとST-MoGEも日次・多犯罪種tensorを使うため、ST-MoGEではこの3km格子をLAPDのregion proxyとして再利用します。dataset IDには初出時の `stmgnn` が残っていますが、データ自体はモデル非依存です。

- dataset ID: `lapd_legacy_2010_2024_stmgnn_target_crimes_grid3000m_daily`
- 元イベント: `BURGLARY`, `CAR_THEFT`, `THEFT_FROM_VEHICLE`
- 空間単位: LAPD境界と交差する3km x 3km格子、205セル
- 時間単位: 日次
- モデル入力形式: `X[day, cell, crime_type]`
- 使用期間: 2010-01-01から2024-11-30
- 使用イベント数: 917,227
- 日数: 5,448
- 全セル・全犯罪種・全日のゼロ率: 81.01%
- 平均カウント: 0.2738件 / cell / crime type / day
- 最大カウント: 94件

3km格子はLAPD 21区域を分割単位にはせず、都市全体で1つの8近傍グラフを作ります。各セルには自己ループを加え、入次数でrow-normalizeした隣接重みをDGCNへ渡します。区域境界は可視化のため保持しますが、区域ごとに別モデルを学習する方式ではありません。

公開データの2024年12月は、日次対象犯罪数が12月14日の78件から15日58件、16日42件と連続的に落ち、最終日の12月30日は3件しかありません。これは発生件数の通常変動よりもreporting delayまたは公開データ末尾の右打ち切りと考えるのが妥当なため、本実験では最後の完全な月である2024年11月30日までを使います。元イベント自体は削除せず、派生データと実験引数にcutoffを記録しています。

保存先:

```text
datas/lapd_full/lapd_legacy_2010_2024_all_crimes_grid300m_h168h/derived/
  lapd_legacy_2010_2024_stmgnn_target_crimes_grid3000m_daily/
```

この派生データも `datas/` 配下なのでGit管理外です。実験スクリプトが必要に応じて再生成します。

### 1.7 退役したデータ

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
- `ref_models/models/stmgnn_zinb/stmgnn_zinb.py`
- `ref_models/models/hcl/hcl.py`
- `ref_models/models/st_moge/st_moge.py`
- `our_model/vol1/stnpp_gat.py`
- `our_model/vol2/etas_enhanced_stnpp_gat.py`
- `our_model/vol3/stnpp_gat_zinb.py`
- `our_model/vol4/stnpp_gat_hcl_zinb.py`
- `our_model/vol5/stnpp_gat_moge_hcl_zinb.py`

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

### 2.5 STMGNN-ZINB

追加文献 `2408.04193v1` は、犯罪カウントの疎性、ゼロ過剰、過分散を同時に扱う `Spatial Temporal Multivariate Zero-Inflated Negative Binomial Graph Neural Network (STMGNN-ZINB)` を提案しています。点過程ではなく、都市を格子グラフにした離散時間の多変量カウント予測モデルです。

論文で明示された構成:

- 入力は `X in R^(N x T x C)`。`N` は空間格子、`T` は履歴時点、`C` は犯罪種。
- DGCN branchで隣接格子間の空間相関を抽出する。
- MTCN branchで時間と犯罪種を結合し、multivariate-temporal correlationを抽出する。
- 両branchが出すZINBパラメータ埋め込みをHadamard積で融合する。
- ZINBの3パラメータは、extra-zero probability `pi`、negative-binomial probability `p`、dispersion `r`。
- ゼロを含む全観測に対して、変分下限ではなく直接negative log-likelihoodを最小化する。
- 3km x 3km格子、日次予測、時系列方向の7:1 train/test split、train末尾30日のvalidationを用いる。
- 論文実験はNYCの4犯罪種とChicagoの4犯罪種を別々に使用する。

本実装:

- `ref_models/models/stmgnn_zinb/stmgnn_zinb.py`
- DGCNは論文式 `D^-1 A H W + B H` に対応するneighbor transformとself-state transformを持つ。
- MTCNは全履歴日・全犯罪種を結合したshared gated projectionを使い、`pi`, `p`, `r` の各headを持つ。
- 空間branchと時間branchをHadamard積で融合して有効なZINBパラメータへ制約する。
- 予測平均は `(1 - pi) * r * p / (1 - p)`、分散もzero-inflated mixtureとして計算する。
- ZINB log probabilityは論文Equation (3)と同じ場合分けを実装し、SciPyのnegative-binomial PMFとの数値一致を確認した。

論文にないため本実験で定めた設定:

- input history: 28日
- output horizon: 1日
- hidden dimension: 24
- DGCN layers: 1
- MTCN intermediate width: 14日 x 3犯罪種
- dropout: 0.10
- optimizer: Adam
- learning rate: 0.001
- weight decay: 0.00001
- batch size: 256
- maximum epochs: 100
- gradient clipping: 5.0
- early-stopping patience: 12 epochs
- random seed: 42
- graph: 8-neighbor 3km grid with self-loops

逐次予測は可能です。テスト期間の各日について、その直前28日間の実観測を入力し、次の1日を予測します。モデルパラメータを毎日再学習する方式ではなく、固定した学習済みモデルへrolling windowを順次入力するone-step-ahead評価です。

### 2.6 HCL: Hawkes-enhanced Spatial-Temporal Hypergraph Contrastive Learning

追加文献 `01605-AAAI24.LiangK.pdf` は、犯罪予測用ハイパーグラフbackboneへ3種類の犯罪相関を追加するHCLを提案しています。ここでいうHawkesは、事件レベルの条件付き強度を最尤推定するETASとは異なり、ハイパーグラフが生成した時点別表現を直近履歴で加算強調するrepresentation-levelの処理です。

論文が定義する相関:

- type spatial correlation: 同じセル・同じ日に発生した犯罪種集合と、発生しなかった犯罪種集合を対応させる。
- neighbor spatial correlation: ある犯罪が発生したセルと、同じ犯罪が発生していない上下左右1-hopセルを対応させる。
- Hawkes temporal correlation: 現時点に近い過去時点ほど大きい指数重みで、現在の表現へ加える。

Hawkes強調は、primitive表現 `H_p` に対して次の形です。

```text
H[t_j] = H_p[t_j] + delta * sum(i=1..s) w[j-i,j] * H_p[t_j-i]
w[j-i,j] = exp(-(t_j - t_j-i + 1) / (t_j - t_j-s + 1))
```

type相関では発生犯罪種と非発生犯罪種の表現をそれぞれ平均poolし、neighbor相関ではanchorセルと非発生近隣セルの表現を平均poolします。どちらもL2正規化後の二乗距離で近づけ、neighbor lossは犯罪種係数 `beta_c` で重み付けします。`beta_c` は、学習期間中にanchorで犯罪が起きたとき、同じ犯罪が1-hop近隣にも起きた割合です。全損失は次です。

```text
L = L_task + lambda_type * L_type + lambda_neighbor * L_neighbor
```

論文実験で明示された条件:

- 空間単位: 3km x 3km。
- 時間単位: 1日。
- train/test: 時系列7:1。
- validation: train末尾1か月。
- 数量予測backbone: ST-HSL、task lossはregression loss。
- 発生有無予測backbone: ST-SHN、task lossはclassification loss。
- NYC最良値: `lambda_type=0.15`, `lambda_neighbor=0.15`, `delta=0.01`, `s=3`。
- Chicago最良値: `lambda_type=0.20`, `lambda_neighbor=0.20`, `delta=0.01`, `s=5`。
- 数量予測指標: MAE、正の観測に対するMAPE。
- 発生有無指標: Micro-F1、Macro-F1。
- 論文報告値は5 runsの平均。

本実装:

- `ref_models/models/hcl/hcl.py`
- 数量予測側のみを実装し、ST-HSL公式実装と同じMSE task lossを使う。
- 表現tensorは `[batch, cell, day, crime, hidden]`。
- 各セル内の犯罪種を結ぶtype hyperedgeと、犯罪種ごとにセルと上下左右セルを結ぶneighborhood hyperedgeを持つcompact structural hypergraph encoderを使う。
- HCL論文Equation (3), (4)のHawkes強調を表現上で適用する。
- Equation (5)-(11)に対応するtype/neighbor pair poolingとnormalized squared-distance lossを適用する。
- 30日分の強調表現をtemporal attentionでpoolし、直近日表現と結合した後、Softplus headで非負の次日countを出す。この予測headは論文で未指定のため本実験側の補完である。
- 全21 LAPD区域を205個の3kmセルからなる単一ハイパーグラフとして学習し、表示も全域を1枚にする。

HCL著者の実装は確認できず、論文はST-HSL内部の設定を再掲していません。そこでST-HSL公式実装からhistory 30日、hidden 16、batch 16、25 epochs、Adam `lr=0.001`, `weight_decay=0.0001`、MSEを参照しました。ただし、ST-HSLそのもののlearnable 128 hyperedges、local/global dual encoder、Infomax/InfoNCE auxiliary lossを移植した完全実装ではありません。本実装はHCL固有の3相関を検証するcompact backbone版です。

### 2.7 ST-MoGE: Spatial-Temporal Mixture-of-Graph-Experts

追加文献 `s11390-025-5404-1.pdf` は、犯罪種ごとに異なる時空間パターンと、少数地域に事件が集中するlong-tailな空間分布を同時に扱う `Spatial-Temporal Mixture-of-Graph-Experts (ST-MoGE)` を提案しています。入力は地域 x 日 x 犯罪種のcount tensorで、複数犯罪種の翌日件数を一つのモデルで予測します。論文実験はNYC 225地域とChicago 140地域を対象とし、日次、過去7日入力、時系列8:1:1分割です。

- 論文公式ページ: `https://jcst.ict.ac.cn/cn/article/doi/10.1007/s11390-025-5404-1`
- 公式Appendix: `https://jcst.ict.ac.cn/en/supplement/0ec730a7-5099-41ec-9951-9b2225a50bcc`

モデルの主要構成:

- Mixture-of-Graph-Experts (MGE): 犯罪種ごとに1つのcategory-specific ST-expertを置き、全犯罪種を同時に受け取るuniversal ST-expertを1つ置く。前者が固有パターン、後者が犯罪種間で共有されるパターンを学習する。
- ST-expert: 地理的近接graphとnode embeddingから学習するadaptive graphを併用し、graph convolutionとdilated causal temporal convolutionを残差・skip connection付きで積み重ねる。
- attentive spatial gate: category-specific表現をquery、universal表現をkey/valueとするmulti-head cross-attentionの後、セル・犯罪種ごとのsigmoid gateで固有表現と共有表現を混合する。
- regional-aware predictor: category-specific expertの学習済みnode embeddingを犯罪種ごとにK-meansし、地域cluster別の予測headを使う。
- Cross-Expert Contrastive Learning (CECL): category-specific expertとuniversal expertが同じ表現へ混ざることを抑え、各expertの専門性を保つ補助目的。
- Hierarchical Adaptive Loss Re-Weighting (HALR): 犯罪種レベルと地域clusterレベルのloss低下率を追跡し、学習が遅い群へ動的に重みを置く。

論文本文・公式Appendixで明示された主要設定:

- hidden dimension: 32。
- node embedding dimension: 16。
- ST blocks: 3。
- block内のspatial layers: NYC 2、Chicago 3。本実験はLAPDにNYC設定の2層を移植。
- block内のtemporal layers: 3、temporal kernel size: 3。
- regional clusters: 4。
- HALR temperature: 1.0、CECL temperature: 0.05。
- optimizer: Adam、initial learning rate: 0.01を段階的に減衰。
- batch size: 64、epochs: 50。
- 評価指標: MAE、観測countが正の要素に対するMAPE。

本実装:

- `ref_models/models/st_moge/st_moge.py`
- 3犯罪種それぞれのspecific expertと、3犯罪種を同時入力するuniversal expertを実装。
- 各expertはprior 8-neighbor graphと `softmax(ReLU(E1 E2^T))` のadaptive adjacencyを併用する。
- 各ST-blockはAppendix A1の図に合わせて `TCN -> GCN -> TCN -> GCN -> TCN` と交互に処理し、3つのTCNそれぞれの後からskip表現を集約する。causal temporal convolutionのdilationは1、2、4。
- regional predictorは犯罪種別に4 cluster x 4 headを持つ。
- CECLは論文式に従い、specific側は全category-specific expert、universal側は全犯罪種のcorrupted表現をnegative集合にする。印刷式では分母にpositive pairが明示されないため、数値的に安定したpositive-inclusive InfoNCEとして実装した。
- HALRは犯罪種別・cluster別MSEの直近2 epoch間の比率からsoftmax weightを求める。論文のiteration単位更新ではなく、本実験ではepoch単位で更新する。

論文・Appendixにないため本実験で定めた設定:

- attention heads: 4、dropout: 0.10、weight decay: 0.00001、gradient clipping: 5.0。
- total loss: `0.8 * HALR prediction loss + 0.2 * (CECL specific loss + CECL universal loss)`。
- learning-rate schedule: 0.01から0.0001へのcosine annealing。
- CECL positive view: 犯罪種以外をmaskしてuniversal expertへ通した表現に、独立したrepresentation-level dropoutを2回適用。
- K-meansは初期embeddingと1 epoch warm-up後のembeddingで更新し、その後は固定。
- count入力は学習期間の統計で標準化した `log1p(count)`、出力はSoftplusによる非負count。

逐次予測は可能です。学習済みモデルを固定し、各予測日の直前7日間の実観測を入力して翌日を予測します。これは日々online fine-tuningする方式ではなく、rolling one-step-ahead評価です。

### 2.8 our_model vol3: STNPP-GAT-ZINB

STMGNN-ZINBからSTNPP-GATへ採用できる要素を検討した結果、vol3では「3km単位の分布付き犯罪総量予測」と「150m単位の点過程risk配分」を分担させる多尺度構成を採用しました。

検討した案:

- ZINBを150mセルへ直接適用する案は採用しませんでした。150m x 日次 x 犯罪種別ではほぼ全観測が0となり、約56,927ノードのDGCNも必要になるため、論文の3km設定から大きく外れ、計算量とzero inflationの双方が過度になります。
- 150m riskへDGCN平滑化を直接かける案も採用しませんでした。イベント近傍の鋭いhotspotを表すSTNPP-GAT/ETASの役割と、隣接格子を拡散させるDGCNの役割が衝突するためです。
- GATによるmark interactionをSTMGNNへ置き換える案も採用しませんでした。犯罪種 x LAPD区域の励起関係と、連続時間の減衰を失うためです。
- 採用したのは、STMGNN-ZINBを3km・日次・犯罪種別の総量予測器、vol2を150mセルへの配分器として組み合わせる方式です。

実装:

- `our_model/vol3/stnpp_gat_zinb.py`
- coarse branch: 205個の3kmセル、3犯罪種、過去28日を入力するDGCN + MTCN + ZINB。
- fine branch: vol2と同じ63 mark GAT、区域別adaptive ETAS、150mセル上の犯罪種別point-process risk。
- ZINB branchは `mean`, `variance`, `pi`, `p`, `r` を出力する。
- fine branchの各3kmセル内riskを集計し、Poisson近似による分散を `variance_point = mean_point` とする。
- ZINB側の重みは、逆分散融合から `w_zinb = variance_point / (variance_point + variance_zinb)` とする。
- 融合平均は `mean_fused = (1 - w_zinb) * mean_point + w_zinb * mean_zinb`。
- 各3kmセル・犯罪種の `mean_fused` を、その中の150mセルへfine branchのrisk比率で再配分する。
- これにより、ZINB分散が大きい予測ではvol2側を多く残し、分布予測が相対的に確かな場所でZINB補正を強くします。

150m配分後の合計は3kmの融合平均と一致します。ただし、ZINBの予測区間を150mセルへ分解したわけではありません。`pi`, `p`, `r`, prediction intervalの解釈は3km x 犯罪種 x 日の解像度に限定します。

### 2.9 our_model vol4: STNPP-GAT-HCL-ZINB

HCLからvol3へ採用できる要素を、時間モデル、空間解像度、不確実性の3点から検討しました。結論として、HCLをfine branchへ足すのではなく、vol3の3km ZINB branchをHCL-regularized ZINBへ置き換える方式を採用します。

検討した案:

- HCLのHawkes強調を150m fine branchへ追加する案は採用しません。fine branchには既に区域別adaptive ETASがあり、実事件による連続時間の自己励起を扱っています。HCLのrepresentation-level加算を重ねると、同じ直近事件を異なる定義で二重に強調し、条件付き強度としての解釈も弱くなります。
- fine branchのGAT/ETASをHCLへ置き換える案も採用しません。HCLは3km・日次tensor用であり、150mの鋭いhotspot、イベント時刻、365日背景率、区域別ETAS再推定を失います。
- HCL point forecastをvol3の第3の平均として後段融合する案は採用しません。HCL point headには校正されたpredictive varianceがなく、vol3の逆分散融合へ対等に入れる根拠がありません。
- type/neighbor相関をfine riskへ事後平滑化としてかける案も採用しません。実事件近傍のrankingを後処理で変え、coarse総量モデルとfine配分モデルの責務が曖昧になるためです。
- 採用したのは、3km coarse branch内部でHCL表現を作り、その表現からZINB分布を直接出す方式です。これならHCLの3相関を犯罪種別総量と不確実性推定へ反映しつつ、vol3の多尺度設計を保てます。

実装:

- `our_model/vol4/stnpp_gat_hcl_zinb.py`
- primitive表現: 標準化した3km日次count、cell embedding、crime embedding、履歴位置embedding。
- hyperedges: 同一セル内の犯罪種hyperedgeと、同一犯罪種の中心セル＋上下左右セルhyperedge。
- Hawkes enhancement: HCL Equation (3)-(4)に従い、直近 `s=3` 日のprimitive表現を `delta=0.01` で加算。
- temporal head: 強調後28日表現のattention poolと直近日表現を結合。
- distribution head: `pi`, `p`, `r` を出し、ZINBのmeanとvarianceを計算。
- task objective: direct ZINB negative log-likelihood。
- auxiliary objective: `lambda_type * L_type + lambda_neighbor * L_neighbor`。
- 全損失: `L = ZINB_NLL + 0.15 * L_type + 0.15 * L_neighbor`。

vol3から維持する部分:

- 150m fine branchの63 mark GAT。
- LAPD 21区域別adaptive ETASと28日間隔の再推定。
- coarse point-process meanとZINB mean/varianceの逆分散融合。
- 融合した3km犯罪種別meanを、fine branchのrisk比率で150mセルへ配分。
- 24時間予測、14日表示間隔、全21区域を統合したGIF。

この構成では、HCLは「coarseな犯罪種・近隣・短期時間相関」、GAT/ETASは「fineなmark interaction・イベント自己励起」、ZINBは「ゼロ過剰・過分散と予測不確実性」を担当します。HCLとETASを同じHawkesモデルとみなさず、異なる解像度と意味を持つ部品として境界を固定しています。

### 2.10 our_model vol5: STNPP-GAT-MoGE-HCL-ZINB

ST-MoGEからvol4へ移植できる要素を、既存のHCL相関学習との整合性、coarse/fineの責務、比較可能性の3点から検討しました。結論として、vol4の3km HCL-ZINB branchを、HCLの共有表現と犯罪種別graph expertを混合するregional ZINBモデルへ置き換えました。150m fine branchと逆分散融合は変更していません。

採用した要素:

- category-specific ST-expert: 犯罪種ごとにprior graphとadaptive graphを使うgraph-temporal expertを1つ置き、HCLだけでは共有されやすい犯罪種固有の時空間パターンを補う。
- attentive spatial gate: 犯罪種別expertをquery、HCL表現をkey/valueとし、セル・犯罪種ごとに固有表現と共有表現を混合する。
- regional-aware ZINB predictor: 犯罪種別node embeddingをK-meansで4群に分け、犯罪種 x 地域clusterごとに別の `pi`, `p`, `r` headを使う。ST-MoGEのpoint headをそのまま使わず、vol4の分布予測と逆分散融合を維持できるZINB headへ拡張した。
- HALR: 犯罪種・地域cluster別ZINB NLLの低下率から、学習が遅い群へepoch単位で動的重みを付ける。checkpoint選択には重み付きlossではなく全要素のunweighted validation ZINB NLLを使う。

維持した要素:

- HCL encoderをuniversal expertとし、type/neighbor contrastive lossとrepresentation-level Hawkes強調を維持する。
- fine branchの63 mark GAT、21区域別adaptive ETAS、365日背景率を維持する。
- coarse point-process meanとZINB mean/varianceの逆分散融合、および150mへのrisk比率配分を維持する。
- vol4のGAT checkpointを再利用し、vol4との差をcoarse count branchの変更へ限定する。

採用しなかった要素:

- CECLは採用しませんでした。ST-MoGEのCECLはspecific expertとuniversal expertの表現を分離して専門性を保ちますが、HCLのtype lossは同一セル・同一日の犯罪種表現を近づけます。両者を同時に強く最適化すると「分離」と「整列」が競合し、どちらの効果か解釈しにくくなります。vol5ではgateとregional predictorの効果を先に検証し、CECLは独立したablation候補とします。
- ST-MoGEの3 blocks x 2 spatial layers x 3 temporal layers、hidden 32をそのまま移植しませんでした。15年分のrolling windowをCPUで学習する計算量と、HCL universal branchとの容量差を抑えるため、specific expertは1 block、1 spatial layer、2 temporal layers、hidden 16としました。
- specific/universal表現をfine branchへ直接注入しませんでした。3km日次表現を150mイベント強度へ混ぜると解像度と確率的意味が曖昧になるためです。

実装:

- `our_model/vol5/stnpp_gat_moge_hcl_zinb.py`
- HCL universal encoderの出力headは使わず、28日履歴のattention contextと直近日表現をprojectして共有expert表現にする。
- 3つのspecific expert、4-head cross-attention gate、犯罪種ごとに4個のregional ZINB headを持つ。
- lossは `HALR-weighted regional ZINB NLL + 0.15 * HCL type loss + 0.15 * HCL neighbor loss`。
- category/cluster weightは直近2 epochのloss比をtemperature 1.0のsoftmaxへ通して更新する。

vol5は、HCLによる犯罪種間の共有相関と、ST-MoGEによる犯罪種固有・地域固有パターンをcoarse branch内で分担させる版です。ただし、各部品の寄与を切り分けるablation前の開発版であり、ST-MoGE論文またはHCL論文の再現モデルではありません。

### 2.11 退役したTransformer v1

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

ETAS、our_model vol1、our_model vol2、our_model vol3、our_model vol4、our_model vol5は同じ150m対象犯罪データ上に最終riskを出し、共通評価できる形にしています。vol3/vol4/vol5内部のZINB branch、reference STMGNN-ZINB、reference HCL、reference ST-MoGEは3km日次格子を使います。3kmのreferenceモデルは150m予測面へ変換せず、共通比較には含めません。

共通データ:

- dataset ID: `lapd_legacy_2010_2024_mohler_target_crimes_grid150m_h24h`
- source dataset ID: `lapd_legacy_2010_2024_all_crimes_grid300m_h168h`
- target crimes: `BURGLARY`, `CAR_THEFT`, `THEFT_FROM_VEHICLE`
- cell size: 150m
- areas: LAPD 21区域
- forecast horizon: 24時間
- forecast display period: 2020-01-01 から 2024-12-26表示まで
- frame interval: 14日

vol3/vol4/vol5は公開データ末尾の右打ち切りを避けるため2024-11-27予測表示までです。モデル間で数値比較するときは、本文3.8以降のように共通期間へ切り揃えます。ST-MoGEは論文NYC実験と暦期間を揃え、2020-07-31から2022-11-30を使います。
- frames: ETAS/vol1/vol2は131、vol3/vol4/vol5は129。

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

### 3.5 STMGNN-ZINB実験

実行スクリプト:

```text
ref_models/experiments/stmgnn_zinb/run_lapd_stmgnn_zinb.py
```

出力先:

```text
ref_models/experiments/stmgnn_zinb/results/
```

論文準拠の設定:

- model ID: `U1_stmgnn_zinb_lapd`
- cell size: 3km x 3km
- nodes: 205
- crime channels: 3
- temporal resolution: daily
- chronological split: 7:1
- validation: train partition末尾30日
- distribution: ZINB
- loss: direct ZINB negative log-likelihood
- spatial branch: DGCN
- multivariate-temporal branch: MTCN
- branch fusion: Hadamard product

期間とsample数:

- full period: 2010-01-01から2024-11-30
- train target period: 2010-01-29から2022-12-20
- validation: 2022-12-21から2023-01-19、30 samples
- test: 2023-01-20から2024-11-30、681 samples
- train samples: 4,709
- test prediction points: 681日 x 205セル x 3犯罪種 = 418,815

実際の学習結果:

- device: CPU
- PyTorch: 2.14.0+cpu
- trainable parameters: 11,355
- completed epochs: 93 / 100
- best epoch: 81
- best validation NLL: 0.550761 / observation
- stop reason: 12 epochs validation改善なしによるearly stopping

テスト結果:

- MAE: 0.307679 count / cell / crime / day
- PICP: 0.965956（10th-90th percentile interval）
- MPIW: 0.777589 count
- occurrence F1: 0.550405
- true-zero rate: 0.910229
- actual zero rate: 0.817609
- predicted zero rate: 0.829481
- test ZINB NLL: 0.486589 / observation
- empirical-vs-predictive count histogram KL: 0.00011703
- observed mean: 0.279617
- predicted mean: 0.280974

ここでoccurrence F1は、予測平均を最寄り整数へ丸めた後に `count > 0` を犯罪発生としたbinary F1です。true-zero rateは実際に0だった観測のうち0と予測した割合です。KLは各観測のZINB PMFを平均したpredictive count histogramと、実測count histogramの離散KLであり、論文が実装詳細を示していないため完全に同一定義とは保証できません。

予測区間は論文表現どおり10%-90% quantileを使います。この区間のnominal coverageは数学的には80%ですが、論文本文には「90%に近いことを目指す」とも書かれており記述が矛盾しています。結果表にはquantile境界とnominal 0.80の両方を記録しました。今回のPICP 0.9660は、離散カウントの区間が保守的であることを示します。

保持している出力:

- `animations/U1_stmgnn_zinb_lapd_24h_forecast_heatmap.gif`
- `model/model_state.pt`
- `tables/animation_config.yml`
- `tables/training_summary.yml`
- `tables/zinb_daily_predictions.parquet`
- `tables/forecast_frames.csv`
- `tables/forecast_cell_risk.parquet`
- `tables/forecast_top_cells.csv`
- `tables/observed_events.parquet`
- `metrics/summary_metrics.csv`
- `metrics/daily_metrics.csv`
- `metrics/training_history.csv`
- `metrics/plots/*.png`

GIFはテスト期間の日次逐次予測から14日ごとに49 frameを抽出しています。ヒートマップは3犯罪種の24時間予測平均の合計、円は実際の事件です。円はセル中心ではなく公開データの投影座標 `x`, `y` に置き、28日間で薄くなります。モデルはLAPD全域を1つのグラフとして学習し、表示も全21区域を1枚に統合しています。

### 3.6 HCL数量予測実験

実行スクリプト:

```text
ref_models/experiments/hcl/run_lapd_hcl.py
```

出力先:

```text
ref_models/experiments/hcl/results/
```

データと時系列分割:

- model ID: `H1_hcl_lapd_quantity`
- dataset ID: `lapd_legacy_2010_2024_stmgnn_target_crimes_grid3000m_daily`
- cell size: 3km x 3km、205 cells。
- crimes: `BURGLARY`, `CAR_THEFT`, `THEFT_FROM_VEHICLE`。
- full period: 2010-01-01から2024-11-30。
- input history: 30日、forecast horizon: 1日。
- train/test: 時系列7:1、train末尾30日をvalidation。
- train target period: 2010-01-31から2022-12-20。
- validation: 2022-12-21から2023-01-19、30 daily samples。
- test: 2023-01-20から2024-11-30、681 daily samples。
- 利用可能なtrain target 4,707日のうち、7日間隔の673 targetを学習に使用。
- validationとtestは間引かず、毎日のrolling one-day-ahead予測を行う。

論文から採用したHCL設定:

- `lambda_type=0.15`
- `lambda_neighbor=0.15`
- `hawkes_delta=0.01`
- `hawkes_scope=3 days`
- neighbor: 上下左右のcardinal 1-hop。
- task loss: MSE regression。
- contrastive loss: L2正規化したpair表現間のsquared distance。

ST-HSL公式実装を参考にした学習設定:

- hidden dimension: 16。
- history: 30日。
- batch size: 16。
- maximum epochs: 25。
- optimizer: Adam。
- learning rate: 0.001。
- weight decay: 0.0001。
- dropout: 0.20。

本実験で定めた設定:

- hypergraph layers: 1。
- structural hyperedges: 同一セルの3犯罪種、同一犯罪種の中心セル＋cardinal neighbors。
- output head: 全30日のtemporal attention contextと直近日表現を結合し、Softplusで非負countへ変換。
- graph edges: self-loop込み919、contrastive neighbor pairs 714。
- contrastive pairを作る時点: 各30日windowの直近日1日。
- train target stride: 7日。
- early-stopping patience: 6 epochs。
- random seed: 42、1 run。

学習期間から求めたneighbor category coefficient:

- `BURGLARY`: 0.674753。
- `CAR_THEFT`: 0.728095。
- `THEFT_FROM_VEHICLE`: 0.820915。

実際の学習結果:

- device: CPU、PyTorch `2.14.0+cpu`。
- trainable parameters: 5,250。
- completed epochs: 25 / 25。
- best epoch: 25。
- best validation total loss: 0.445806。
- epoch 1から25でvalidation task MSE: 0.745588から0.444172。
- epoch 1から25でvalidation type alignment: 0.170899から0.005559。
- epoch 1から25でvalidation neighbor alignment: 0.157683から0.005335。

テスト結果:

| scope | MAE | MAPE（観測count > 0） | RMSE | observed mean | predicted mean |
|---|---:|---:|---:|---:|---:|
| ALL | 0.299248 | 0.601419 | 0.573701 | 0.279617 | 0.262906 |
| BURGLARY | 0.187013 | 0.728975 | 0.405405 | 0.137431 | 0.122353 |
| CAR_THEFT | 0.344345 | 0.588203 | 0.621465 | 0.341714 | 0.320347 |
| THEFT_FROM_VEHICLE | 0.366386 | 0.554587 | 0.660929 | 0.359708 | 0.346019 |

全体のpredicted/observed mean比は約0.940で、平均件数をやや過小予測しています。同じ3km tensorとtest期間を使ったreference STMGNN-ZINBのMAE 0.307679より小さい値ですが、HCLはpoint-regression、STMGNN-ZINBは確率分布NLLを目的にしており、学習sample数や出力も異なります。この1 runだけを根拠にモデルの優劣とは解釈しません。

テストでは、各予測日の直前30日間の実観測を入力し、3犯罪種それぞれの次日countを205セル全域で予測します。GIFは毎日の予測から14日ごとにframeを選び、3犯罪種の予測count合計をヒートマップ、実際の事件を犯罪種別の円で表示します。学習も可視化もLAPD全21区域を一つに統合しています。

生成GIFは49 frames、770 x 787 px、3,461,867 bytesです。表示予測日は2023-01-20から2024-11-22で、各frameの円はその予測日から次日までの実事件に加え、過去28日間の事件を時間経過に応じて薄く表示します。

保持する出力:

- `animations/H1_hcl_lapd_quantity_24h_forecast_heatmap.gif`
- `model/model_state.pt`
- `tables/animation_config.yml`
- `tables/training_summary.yml`
- `tables/daily_predictions.parquet`
- `tables/forecast_frames.csv`
- `tables/forecast_cell_risk.parquet`
- `tables/forecast_top_cells.csv`
- `tables/observed_events.parquet`
- `metrics/summary_metrics.csv`
- `metrics/daily_metrics.csv`
- `metrics/training_history.csv`
- `metrics/plots/*.png`

### 3.7 ST-MoGE多犯罪種数量予測実験

実行スクリプト:

```text
ref_models/experiments/st_moge/run_lapd_st_moge.py
```

出力先:

```text
ref_models/experiments/st_moge/results/
```

データと時系列分割:

- model ID: `G1_st_moge_lapd_quantity`。
- dataset ID: `lapd_legacy_2010_2024_stmgnn_target_crimes_grid3000m_daily`。
- cell size: 3km x 3km、205 cells。
- graph: 自己loop込みrow-normalized 8-neighbor、directed edges 1,593。
- crimes: `BURGLARY`, `CAR_THEFT`, `THEFT_FROM_VEHICLE`。
- full period: 2020-07-31から2022-11-30。論文AppendixのNYCと同じ暦期間。
- input history: 7日、forecast horizon: 1日。
- train/validation/test: 時系列8:1:1。
- train targets: 2020-08-07から2022-06-12、675日。
- validation: 2022-06-13から2022-09-05、85日。
- test: 2022-09-06から2022-11-30、86日。
- validation/testは間引かず、毎日のrolling one-day-ahead予測を行う。

論文・Appendixに合わせたモデル設定:

- hidden dimension: 32、node embedding dimension: 16。
- ST blocks: 3、spatial layers: 2、temporal layers: 3、kernel size: 3。
- category-specific experts: 3、universal experts: 1。
- regional clusters: 犯罪種ごとに4。
- CECL temperature: 0.05、HALR temperature: 1.0。
- batch size: 64、epochs: 50。
- optimizer: Adam、initial learning rate: 0.01。

本実験で定めた設定:

- attention heads: 4、dropout: 0.10、weight decay: 0.00001。
- loss weights: prediction 0.80、CECL 0.20。
- cosine learning-rate schedule、minimum learning rate 0.0001。
- cluster refresh: 初期状態と1 epoch warm-up後の計2回。その後固定。
- HALR weight update: epoch単位。
- train stride: 7。ただしepochごとにoffsetを循環し、50 epochs中に675の全train targetを訪問。
- `torch.compile`: 無効、eager CPU 8 threads。交互ST-blockのCPU compileが初回epoch前に長時間停止したため、再現可能なeager実行を採用。
- random seed: 42、1 run。

学習結果:

- trainable parameters: 260,972。
- completed epochs: 50 / 50。
- best epoch: 10。
- best validation MAE: 0.309102 count / cell / crime / day。
- epoch 1のvalidation MAE: 0.602825。
- epoch 50のvalidation MAE: 0.326017。

テスト結果:

| scope | MAE | MAPE（観測count > 0） | RMSE | observed mean | predicted mean |
|---|---:|---:|---:|---:|---:|
| ALL | 0.316238 | 0.727069 | 0.667805 | 0.325279 | 0.204618 |
| BURGLARY | 0.224491 | 0.850862 | 0.507755 | 0.196313 | 0.082725 |
| CAR_THEFT | 0.312791 | 0.737717 | 0.684169 | 0.336188 | 0.175925 |
| THEFT_FROM_VEHICLE | 0.411433 | 0.642745 | 0.782297 | 0.443335 | 0.355205 |

全体のpredicted/observed mean比は約0.629で、平均件数を約37.1%過小予測しています。頻度四分位別では、各犯罪種とも低頻度Q1のabsolute MAEは小さい一方、正例MAPEは0.940から0.971と高く、少数地域の相対誤差問題はHALRを入れても残りました。高頻度Q4のMAEはBURGLARY 0.546、CAR_THEFT 0.807、THEFT_FROM_VEHICLE 0.962です。

テスト期間の平均specific-expert gateは、BURGLARYでcluster別0.653から0.677、CAR_THEFTで0.524から0.536、THEFT_FROM_VEHICLEで0.556から0.569でした。3犯罪種ともspecific expertをuniversal側より強く使い、特にBURGLARYでその傾向が大きいことを示しますが、gateの因果的解釈や安定性はまだ検証していません。

GIFは86日の日次予測から7日ごとに13 frameを抽出しています。3犯罪種の翌日予測count合計をヒートマップ、実事件を犯罪種別色の円で表示し、過去28日間の円を時間経過で薄くします。全205セルとLAPD 21区域境界を一枚に統合しています。

保持する出力:

- `animations/G1_st_moge_lapd_quantity_24h_forecast_heatmap.gif`
- `model/model_state.pt`
- `model/best_training_checkpoint.pt`
- `tables/training_summary.yml`
- `tables/animation_config.yml`
- `tables/daily_predictions.parquet`
- `tables/region_clusters_and_gates.csv`
- `tables/forecast_frames.csv`
- `tables/forecast_cell_risk.parquet`
- `tables/forecast_top_cells.csv`
- `tables/observed_events.parquet`
- `metrics/summary_metrics.csv`
- `metrics/daily_metrics.csv`
- `metrics/frequency_quantile_metrics.csv`
- `metrics/gate_summary.csv`
- `metrics/training_history.csv`
- `metrics/plots/*.png`

### 3.8 our_model vol3実験

実行スクリプト:

```text
our_experiment/vol3/run_lapd_stnpp_gat_zinb.py
```

予測期間と分割:

- coarse grid: 3km、205セル。
- fine grid: 150m、56,927セル。
- crime channels: `BURGLARY`, `CAR_THEFT`, `THEFT_FROM_VEHICLE`。
- ZINB train targets: 2010-01-29から2019-12-01、3,594 samples。
- validation: 2019-12-02から2019-12-31、30 samples。
- test: 2020-01-01から2024-11-30、1,796 daily samples。
- GIF: test予測を14日ごとに抽出した129 frames、最終予測日は2024-11-27。
- 2024年12月は公開データ末尾の右打ち切りがあるため含めない。

学習設定:

- ZINB input history: 28日。
- ZINB hidden dimension: 24。
- DGCN layers: 1。
- MTCN intermediate width: 14日 x 3犯罪種。
- optimizer: Adam、learning rate `0.001`、weight decay `0.00001`。
- batch size: 256、maximum epochs: 100、early-stopping patience: 12。
- GAT: hidden dimension 64、8 heads、SGD learning rate 1.0、1,500 epochs。
- ETAS: 365日history、28日refit、30日trigger lookback、21区域別 `theta/omega`。

学習結果:

- ZINB trainable parameters: 11,355。
- ZINB completed epochs: 100 / 100。
- best epoch: 100。
- best validation NLL: 0.517349 / observation。
- GAT transition KL: 3.119379 から 2.577063。
- GAT transition学習イベント: 593,806。
- weighted transition pairs: 334,510。

3km ZINB branchの全テスト日評価:

- MAE: 0.321306 count / cell / crime / day。
- PICP: 0.962517（10th-90th percentile interval）。
- MPIW: 0.790904 count。
- occurrence F1: 0.544209。
- true-zero rate: 0.904428。
- ZINB NLL: 0.514162 / observation。
- empirical-vs-predictive histogram KL: 0.0008103。
- observed mean: 0.292811、predicted mean: 0.279458。

不確実性融合:

- 個別3kmセル・犯罪種のZINB weight範囲: 0.002624から0.984178。
- 全frame・cell・crimeの平均ZINB weight: 0.431394。
- frame平均weightの範囲: 0.385437から0.535385。
- 150m配分時のfallback group: 0。
- 3km融合平均と150m配分後合計の最大誤差: `2.93e-14`。

vol2と共通する2020-01-01から2024-11-27の129 frameで比較すると、次のようになりました。

| 指標 | vol2 | vol3 |
|---|---:|---:|
| cell MAE | 0.006239 | 0.006042 |
| cell RMSE | 0.057009 | 0.056994 |
| Brier score | 0.003001 | 0.003000 |
| log loss | 0.018724 | 0.018702 |
| probability O/E | 0.954794 | 1.016837 |
| count O/E | 0.970003 | 1.034203 |
| PR-AUC | 0.031387 | 0.031588 |
| Spearman | 0.078521 | 0.065191 |
| Hit Rate @ 5% | 0.399665 | 0.399278 |
| PAI @ 5% | 7.991471 | 7.983732 |

MAE、RMSE、Brier score、log loss、PR-AUC、probability O/Eは小幅に改善し、Hit RateとPAIはほぼ維持しました。一方でSpearmanは低下し、count O/Eも1からの絶対差ではわずかに悪化しました。したがって、vol3は全面的な優越ではなく、「総量と確率校正を改善しつつhotspot性能を概ね維持した最初の不確実性融合版」と位置付けます。

保持している主な出力:

- `our_experiment/vol3/results/animations/O3_stnpp_gat_zinb_multiscale_24h_forecast_heatmap.gif`
- `results/model/model_state.pt`: ZINBとGATのstate、標準化統計、split。
- `results/tables/coarse_zinb_daily_predictions.parquet`: 全テスト日のZINB予測表。
- `results/tables/coarse_zinb_frame_predictions.parquet`: point-process mean、ZINB mean/variance、融合weight、融合mean。
- `results/tables/area_etas_parameters.csv`
- `results/tables/learned_mark_transition.csv`
- `results/metrics/zinb_*`: 3km分布予測の評価。
- `results/metrics/summary_metrics.csv`など: 融合後150m予測の共通評価。

`--reuse-model-state` を指定すると、保存済みZINB/GAT重みを使い、ETAS逐次予測、融合、GIF、結果表だけを再生成できます。

### 3.9 our_model vol4実験

実行スクリプト:

```text
our_experiment/vol4/run_lapd_stnpp_gat_hcl_zinb.py
```

出力先:

```text
our_experiment/vol4/results/
```

データと分割:

- model ID: `O4_stnpp_gat_hcl_zinb_multiscale`。
- coarse grid: 3km、205セル、3犯罪種、日次。
- fine grid: 150m、56,927セル、LAPD全21区域。
- input history: 28日、forecast horizon: 24時間。
- HCL-ZINB train targets: 2010-01-29から2019-12-01。
- 利用可能な3,594 train targetsのうち7日間隔の514 samplesを学習に使用。
- validation: 2019-12-02から2019-12-31、30 daily samples。
- test: 2020-01-01から2024-11-30、1,796 daily samples。
- GIF: test predictionから14日ごとに129 framesを表示。

HCL-ZINB設定:

- hidden dimension: 16。
- structural hypergraph layers: 1。
- dropout: 0.20。
- `hawkes_scope=3`, `hawkes_delta=0.01`。
- `lambda_type=0.15`, `lambda_neighbor=0.15`。
- contrastive lossを各28日windowの直近日へ適用。
- task loss: direct ZINB NLL。
- optimizer: Adam、learning rate 0.001、weight decay 0.0001。
- batch size: 16、maximum epochs: 25、early-stopping patience: 6。
- random seed: 42、1 run。

fine branchと融合設定はvol3と揃えます。GATはhidden 64、8 heads、1500 epochsで、区域別ETASは365日履歴、28日ごとに再推定します。HCL-ZINBのmean/varianceとcoarse point-process meanを逆分散融合し、融合後の犯罪種別meanを150mへ再配分します。

実行結果:

- HCL-ZINBの学習可能パラメータ数は5,252。25 epochsを完走し、best epochは23、validation total lossは0.523689、内訳はZINB NLL 0.520151、type loss 0.009212、neighbor loss 0.014374。
- 犯罪種係数はBURGLARY 0.681291、CAR_THEFT 0.705999、THEFT_FROM_VEHICLE 0.820408。
- GATのtransition KL lossは3.119379から2.577063へ低下。
- 129 frame上のHCL-ZINB融合weightは平均0.420050、中央値0.431948。coarse point-process meanの平均0.299088、HCL-ZINB meanの平均0.298917に対し、融合meanは平均0.290907。
- GIFは2020-01-01から2024-11-27までの129 frames、14日間隔。全21区域を同一地図にまとめ、ヒートマップを予測平均、色付き円を観測事件として表示。

3km HCL-ZINB単体のテスト結果:

| 指標 | vol3 STMGNN-ZINB | vol4 HCL-ZINB |
|---|---:|---:|
| MAE | 0.321306 | 0.324480 |
| ZINB NLL / observation | 0.514162 | 0.513416 |
| PICP 10%-90% | 0.962517 | 0.967402 |
| MPIW 10%-90% | 0.790904 | 0.856528 |
| Occurrence F1 | 0.544209 | 0.557056 |
| Mean observed count | 0.292811 | 0.292811 |
| Mean predicted count | 0.279458 | 0.300369 |

HCL-ZINBはNLL、Occurrence F1、平均総量を改善しましたが、MAEは悪化し、80%区間のPICPはさらに高く、MPIWも広がりました。したがって、HCL表現によって発生有無と総量校正は改善した一方、点予測と区間の鋭さを同時には改善できていません。また、平均extra-zero probability `pi` はvol3の0.057084から0.006185へ低下しており、vol4は実質的にNB成分へ寄っています。ゼロ過剰成分が不要なのか、HCLのtype alignmentが構造的ゼロ識別を弱めたのかはablationが必要です。

同じ129 frameに対する融合後150m予測のvol3比較:

| 指標 | vol3 | vol4 |
|---|---:|---:|
| Cell MAE | 0.006042 | 0.006156 |
| Cell RMSE | 0.057231 | 0.057233 |
| Poisson deviance | 0.032203 | 0.032189 |
| Brier score | 0.00299993 | 0.00300031 |
| Log loss | 0.0187019 | 0.0186968 |
| Probability O/E | 1.020686 | 0.982413 |
| Count O/E | 1.038458 | 0.998913 |
| PR-AUC | 0.031588 | 0.031898 |
| Spearman | 0.065191 | 0.065722 |
| Hit Rate @ 5% | 0.399278 | 0.399373 |
| PAI @ 5% | 7.983732 | 7.985632 |
| Area under PAI curve @ 1%-10% | 0.757746 | 0.760723 |

vol4はcount O/Eをほぼ1へ補正し、Poisson deviance、log loss、PR-AUC、Spearman、hotspot rankingを小幅に改善しました。一方、Cell MAE、RMSE、Brier scoreは小幅に悪化しています。fine branchを固定したためranking差は小さく、現段階の採用効果は主にcoarse総量校正に現れたと解釈します。1 seedのみで差も小さいため、vol4がvol3より一般に優れるとはまだ結論しません。

保持する出力:

- `animations/O4_stnpp_gat_hcl_zinb_multiscale_24h_forecast_heatmap.gif`
- `model/model_state.pt`
- `tables/animation_config.yml`
- `tables/training_summary.yml`
- `tables/coarse_hcl_zinb_daily_predictions.parquet`
- `tables/coarse_hcl_zinb_frame_predictions.parquet`
- `tables/area_etas_parameters.csv`
- `tables/learned_mark_transition.csv`
- `metrics/hcl_zinb_summary_metrics.csv`
- `metrics/hcl_zinb_daily_metrics.csv`
- `metrics/hcl_zinb_training_history.csv`
- `metrics/summary_metrics.csv`などの共通150m評価。
- `metrics/plots/*.png`

### 3.10 our_model vol5実験

実行スクリプト:

```text
our_experiment/vol5/run_lapd_stnpp_gat_moge_hcl_zinb.py
```

出力先:

```text
our_experiment/vol5/results/
```

データと分割:

- model ID: `O5_stnpp_gat_moge_hcl_zinb_multiscale`。
- coarse grid: 3km、205セル、3犯罪種、日次。
- fine grid: 150m、56,927セル、LAPD全21区域。
- input history: 28日、forecast horizon: 24時間。
- train targets: 2010-01-29から2019-12-01、利用可能な3,594日。
- validation: 2019-12-02から2019-12-31、30日。
- test: 2020-01-01から2024-11-30、1,796日。
- GIF: 2020-01-01から2024-11-27を14日間隔で表示した129 frames。

coarse model設定:

- HCL universal expert: hidden 16、1 hypergraph layer、dropout 0.20、`hawkes_scope=3`, `hawkes_delta=0.01`。
- HCL losses: `lambda_type=0.15`, `lambda_neighbor=0.15`。
- category-specific expert: 犯罪種ごとに1個、1 ST-block、1 spatial layer、2 temporal layers、kernel size 3、node embedding 16、dropout 0.10。
- attentive spatial gate: 4 heads。
- regional ZINB predictor: 犯罪種ごとに4 K-means clusters x 4 distribution heads。
- HALR temperature: 1.0。clusterは最初の2 epochsで更新し、その後固定。
- optimizer: Adam、learning rate 0.001、weight decay 0.0001、batch size 16、最大25 epochs、early-stopping patience 8。
- train stride: 7日。ただしepochごとにoffsetを循環し、25 epochs中に3,594の全train targetを少なくとも1回使用。
- random seed: 42、CPU 8 threads、1 run。
- fine GATは `our_experiment/vol4/results/model/model_state.pt` を再利用。

学習結果:

- trainable parameters: 42,037。
- completed epochs: 25 / 25、best epoch: 23。
- best validation unweighted ZINB NLL: 0.519835 / observation。
- 学習に訪問した異なるtarget日: 3,594 / 3,594。
- GAT transition KL: 3.119379から2.577063。学習イベント593,806、weighted pairs 334,510。

3km coarse branchのテスト結果:

| 指標 | vol4 HCL-ZINB | vol5 MoGE-HCL-ZINB |
|---|---:|---:|
| MAE | 0.324480 | 0.321448 |
| ZINB NLL / observation | 0.513416 | 0.514242 |
| PICP 10%-90% | 0.967402 | 0.963528 |
| MPIW 10%-90% | 0.856528 | 0.804460 |
| Occurrence F1 | 0.557056 | 0.541710 |
| Mean observed count | 0.292811 | 0.292811 |
| Mean predicted count | 0.300369 | 0.279379 |

vol5はMAEを約0.93%、区間幅MPIWを約6.08%改善し、PICPも名目80%へわずかに近づきました。一方、NLLとOccurrence F1は悪化し、平均countは約4.59%過小予測です。したがって、地域・犯罪種別expertは点予測と区間の鋭さには寄与した可能性がありますが、発生有無と分布全体の適合を同時には改善していません。

gateとlong-tail診断:

- mean specific-expert gateは、BURGLARYのcluster別0.364-0.371、CAR_THEFT 0.425-0.429、THEFT_FROM_VEHICLE 0.472-0.474。全犯罪種でuniversal HCL側の比重がより大きく、頻度の高いTHEFT_FROM_VEHICLEほどspecific側を多く使いました。
- training-frequency Q1のMAEは3犯罪種で0.016-0.023と小さい一方、正例MAPEは0.969-0.980。高頻度Q4はMAE 0.507-0.862、正例MAPE 0.420-0.599でした。HALRを入れても低頻度地域の正例相対誤差は解消していません。
- clusterごとのgate差は各犯罪種内で最大約0.008と小さく、現時点では地域clusterが明確に異なるexpert選択を生んだとは言えません。

同じ129 frameに対する融合後150m予測のvol4比較:

| 指標 | vol4 | vol5 |
|---|---:|---:|
| Cell MAE | 0.006156 | 0.006048 |
| Cell RMSE | 0.057233 | 0.057228 |
| Brier score | 0.00300031 | 0.00299968 |
| Expected calibration error | 0.0008029 | 0.0007144 |
| Count O/E | 0.998913 | 1.036160 |
| PR-AUC | 0.031898 | 0.031728 |
| Spearman | 0.065722 | 0.064756 |
| Hit Rate @ 1% | 0.157130 | 0.155478 |
| Hit Rate @ 5% | 0.399373 | 0.399361 |
| Hit Rate @ 10% | 0.570461 | 0.571393 |

150mではMAE、RMSE、Brier score、ECEが小幅に改善し、5% hotspot rankingは実質同じでした。一方、count O/Eはvol4のほぼ1から1.036へ悪化し、PR-AUCとSpearmanも小幅に低下しました。fine branchを固定しているためranking差は小さく、vol5の主な変化はcoarse総量・不確実性校正に現れています。1 seedかつablationなしなので、vol5がvol4より一般に優れるとは結論しません。

保持する出力:

- `animations/O5_stnpp_gat_moge_hcl_zinb_multiscale_24h_forecast_heatmap.gif`
- `model/model_state.pt`
- `tables/training_summary.yml`
- `tables/animation_config.yml`
- `tables/coarse_moge_hcl_zinb_daily_predictions.parquet`
- `tables/coarse_moge_hcl_zinb_frame_predictions.parquet`
- `tables/region_clusters_and_gates.csv`
- `tables/area_etas_parameters.csv`
- `tables/learned_mark_transition.csv`
- `metrics/moge_hcl_zinb_summary_metrics.csv`
- `metrics/moge_hcl_zinb_daily_metrics.csv`
- `metrics/moge_hcl_zinb_training_history.csv`
- `metrics/frequency_quantile_metrics.csv`
- `metrics/gate_summary.csv`
- `metrics/summary_metrics.csv`などの共通150m評価。
- `metrics/plots/*.png`

### 3.11 退役した実験

以下は退役または参考用です。

- 固定パラメータ・週次・全犯罪の `ETAS_full`
- 週次セル系列の `Transformer_v1`
- 3区域だけを使った `ETAS_v1` / `ETAS_v2`
- 静的診断プロット
- metric CSVやbacktest summary

今回のファイル整理では、古い結果を保持するよりも、参照モデル再現と本プロジェクト独自モデルの改良履歴を分けることを優先しました。

### 3.12 評価指標実験

追加文献 `Predictive Policingモデルの数値実験で採用可能な評価指標` に基づき、共通150m格子を使う各実験に対して評価指標をまとめて計算します。

STMGNN-ZINBとHCLは3km・犯罪種別・日次という異なる出力を持つため、この共通evaluatorへ無理に変換しません。STMGNN-ZINBはMAE、PICP、MPIW、F1、true-zero rate、KL、NLLを `ref_models/experiments/stmgnn_zinb/results/metrics/` に、HCLは論文の数量予測指標であるMAEと正の観測に対するMAPEを `ref_models/experiments/hcl/results/metrics/` に直接保存します。

実行スクリプト:

```text
metrics/evaluate_current_results.py
```

対象モデル:

- `Reference ETAS`
- `Our vol1 STNPP-GAT`
- `Our vol2 ETAS-enhanced STNPP-GAT`
- `Our vol3 STNPP-GAT-ZINB`
- `Our vol4 STNPP-GAT-HCL-ZINB`
- `Our vol5 STNPP-GAT-MoGE-HCL-ZINB`

主な出力:

各モデルの `results/metrics/` に、モデル単位の評価表と単体プロットを保存します。モデル横断の比較プロットは生成せず、比較が必要な場合は各モデルの `summary_metrics.csv` を後から読み込んで行います。

- `ref_models/experiments/etas/results/metrics/`
- `our_experiment/vol1/results/metrics/`
- `our_experiment/vol2/results/metrics/`
- `our_experiment/vol3/results/metrics/`
- `our_experiment/vol4/results/metrics/`
- `our_experiment/vol5/results/metrics/`

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

2024年12月後半には公開データ末尾の明確な右打ち切りが見られます。STMGNN-ZINB実験では2024年11月30日で切りましたが、既存のETAS/STNPP-GAT GIFは12月の表示を含みます。既存3モデルを再評価するときも同じcutoffへ揃える必要があります。

さらにlegacy dataでは、`BURGLARY` が2024年1月1,344件、3月1,022件から4月564件、5月222件、6月50件へ不連続に減少する一方、車両系2犯罪は同程度で推移しています。犯罪情勢だけで説明するより、記録方式・分類・収録範囲の変更を疑うべき系列です。このためST-MoGE実験は論文NYCと同じ2020-07-31から2022-11-30を使い、2024年の不連続をtrain/validation/testから除外しました。原因の一次資料による確定とNIBRS-era dataとの接続は未実施です。

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

### 4.4 STMGNN-ZINB再現の限界

文献 `2408.04193v1` は5ページで、著者実装も公開されていません。DGCN、MTCN、Hadamard fusion、ZINB NLL、3km格子、日次解像度、7:1 split、30日validationは本文に従いましたが、履歴長、将来step数、層数、hidden width、dropout、optimizer、learning rate、batch size、epoch数、隣接グラフの作り方は記載がありません。これらは再現可能な実験設定として本リポジトリ側で定めたもので、著者実装の完全再現ではありません。

論文はNYCとChicagoの各4犯罪種を使いますが、今回の実験は既存のLAPD対象3犯罪を使います。データ期間、都市形状、zero rate、crime taxonomyが異なるため、論文Table 2の数値と直接比較できません。HA、STGCN、STMGNN-NB、Gaussian、Truncated Normalのbaselineも今回は再実行していません。

論文のPICP説明には、10%-90% intervalとしながらcoverageを90%に近づけるという不整合があります。本実験はquantile境界を優先し、nominal coverageを80%として併記しました。またKL divergence、F1、true-zero rateの厳密な集計方法も本文にありません。本実装の定義は結果とともに明示していますが、論文値との同一定義は保証できません。

3km格子は既存150mモデルより粗く、同じhotspot面積率で直接比較できません。3km予測を150mへ見かけ上補間すると精度を誤認させるため、現在の共通150m metric evaluatorには加えていません。将来比較する場合は、全モデルを同じ3km・日次・犯罪種別カウントへ再学習する必要があります。

テスト時は直前28日の実観測を毎日入力するrolling one-step-ahead予測です。未来予測を自己回帰的に次の入力へ戻すmulti-step forecastや、online fine-tuningはまだ試していません。extra-zero parameter `pi` の時空間的解釈、区間calibrationの再調整、異なる履歴長に対する感度も未確認です。

### 4.5 HCL再現の限界

HCL論文はframeworkをplug-and-play moduleとして記述し、数量予測にはST-HSL、発生有無予測にはST-SHNを使いますが、HCL自体の著者実装は確認できませんでした。今回の実装はHCL固有のHawkes強調、type相関、neighbor相関を数式に沿って実装した一方、backboneはLAPDの不規則な境界付き格子でも動くcompact structural hypergraphです。したがって、論文のST-HSL + HCL完全再現ではありません。

論文はNYC/Chicagoの2年間・各4犯罪種を使います。今回のLAPD実験は2010-2024年の3犯罪種であり、期間、都市、crime taxonomy、セル形状が異なります。NYC向けの `lambda_type=0.15`, `lambda_neighbor=0.15`, `delta=0.01`, `s=3` を移植しましたが、LAPD上でgrid searchしていません。論文Table 3と数値を直接比較できません。

論文では5 runsの平均を報告しますが、現状はCPU上のseed 42による1 runです。また15年分の全日を25 epochs学習すると計算量が大きいため、学習target日は7日間隔にしています。各sampleの30日履歴には中間日も含まれますが、全日をtargetにした学習とは異なります。type/neighbor contrastive lossも、重複rolling windowで同じ日を何度も数えないよう各windowの直近日だけに適用しています。この2点は再現性のため設定表へ保存しています。

3kmセルはLAPD境界でclipされるため、論文の完全な矩形格子と異なり、端部では上下左右の一部が存在しません。neighbor係数 `beta_c` は学習期間だけから計算して情報漏洩を避けていますが、区域境界や道路ネットワークは考慮しません。

現在は数量予測だけです。ST-SHNをbackboneにしたbinary occurrence版、Micro-F1/Macro-F1、HCL各要素を外したablation、論文の全grid search、5 seedsの平均と信頼区間は未実施です。またMAPEはST-HSL公式コードに合わせ、観測countが正の要素だけで計算しています。

### 4.6 ST-MoGE再現の限界

ST-MoGEの著者コードは確認できませんでした。MGE、attentive spatial gate、regional-aware predictor、CECL、HALR、ST-block数、層数、hidden width、cluster数、temperature、optimizer、batch size、epoch数は論文本文と公式Appendixに従いましたが、実装細部を著者コードと照合できないため完全再現ではありません。

特にCECLの印刷式は分母にnegative termsだけを示し、positive pairを明示していません。そのままでは標準的なInfoNCEと異なるため、本実装はpositive-inclusive denominatorで補完しました。universal expertへのcorrupted input、2つのaugmentation、attention heads、dropout、prediction/contrastive loss比率、weight decay、cosine scheduleも再現可能性のため本実験側で定めた設定です。

論文の地域clusterは学習中のnode embeddingに基づき更新され、HALRはiterationごとのloss低下率を使います。本実験はCPU実行時のcluster割当を安定させるため、K-meansを最初の2 epochsで固定し、HALRをepoch単位で更新しました。また各epochはtrain targetを7日間隔で使い、offsetを循環して7 epochsで全日を訪問します。50 epochsで全train targetは複数回使われますが、論文の50 full passesとは異なります。

論文はNYC 225地域・Chicago 140地域と各都市固有の犯罪種を使いますが、本実験はLAPDの205個の3km格子と3犯罪種です。NYCと同じ暦期間・spatial layer数を使っても、region定義、crime taxonomy、count分布が異なるため、論文Table 1-2と数値を直接比較できません。

現状はseed 42の1 runで、論文の12 baselinesやablationをLAPD上で再実行していません。specific expert、universal expert、attentive gate、regional predictor、CECL、HALRを個別に外した比較がないため、LAPDでどの部品が改善へ寄与したかは未確認です。低頻度Q1ではabsolute MAEが小さくても正例MAPEが0.95前後であり、HALRがdata-scarce regionsの相対誤差を十分に解消したとは言えません。

逐次評価は直前7日の実観測を毎日入力するone-step-ahead方式です。online fine-tuning、予測値を次の入力へ戻すmulti-step forecast、concept driftへの適応は未実施です。gate weightとregion clusterも、時空間パターンの説明として安定しているか、seedや期間を変えた検証が必要です。

### 4.7 our_model vol3の限界

vol3の逆分散融合では、fine branchの期待件数をPoissonとみなし、分散が平均に等しいと仮定しています。しかし、STNPP-GAT/ETAS側のparameter uncertaintyやmodel uncertaintyを直接推定したものではありません。ZINB branchとの独立性も仮定しているため、`w_zinb` は厳密なBayesian model averagingではなく、再現可能なuncertainty-aware heuristicです。

ZINBの不確実性は3km・犯罪種別・日次にのみ定義されています。150mセルへは期待件数だけを配分しており、fine-cell prediction interval、fine-cell `pi`、fine-cell dispersionは出していません。これらを出すには、coarse countをfine cellsへ確率的に配るhierarchical count modelが必要です。

DGCN/MTCN branchとGAT/ETAS branchは別々に学習しています。end-to-endでjoint likelihoodを最適化していないため、ZINB補正がSpearmanを低下させる場合があります。今回も同一期間比較でSpearmanが0.078521から0.065191へ低下しました。

評価期間は右打ち切りを避けて2024年11月末までとしたため、2024年12月まで含む既存vol1/vol2の公開summaryとは期間が異なります。本文のvol2/vol3比較は両方を2024-11-27までに切った129 frameで計算しています。

### 4.8 our_model vol4の限界

vol4はHCL論文の再現モデルではなく、HCLの相関正則化をvol3の分布予測へ移植した独自モデルです。HCL論文の数量予測はST-HSL + MSEですが、vol4はcompact structural hypergraph + ZINB NLLです。そのため、HCL論文の改善率をvol4へ期待できるとは限りません。

HCLのtype lossは、同一セル・同一日の発生犯罪種表現と非発生犯罪種表現を近づけます。これは犯罪種相関を共有する一方、ZINBが必要とする「発生と構造的ゼロの識別」と競合する可能性があります。neighbor lossも同様に、anchor発生セルと非発生neighborを近づけるため、空間平滑化が過剰になる可能性があります。type、neighbor、Hawkesを個別に外すablationで確認する必要があります。

学習はCPU計算量を抑えるためtrain targetを7日間隔にし、seed 42の1 runです。validation/testは毎日評価しますが、全日target・複数seedの学習結果ではありません。`lambda_type=0.15`, `lambda_neighbor=0.15`, `delta=0.01`, `s=3` もNYC向けHCL設定の移植で、LAPD上のgrid searchは未実施です。

HCL-ZINBのvarianceはaleatoricなcount dispersionを表し、parameter uncertaintyやmodel uncertaintyではありません。fine branchをPoisson近似する逆分散融合、coarse/fine branchの独立性、150mでは予測区間を出せないというvol3の制約も残ります。

実測した80%予測区間はPICP 0.967402、MPIW 0.856528で、名目被覆率より大幅に高い保守的な区間でした。平均extra-zero probabilityも0.006185まで縮小しています。現状の分布headは点予測、ゼロ過剰、区間幅を十分に分離できておらず、PIT/coverage calibration、ZINB対NBの比較、補助損失ablationが必要です。

fine GAT/ETAS branchはvol3と同じなので、HCL-ZINBが直接変えられるのは3kmセル・犯罪種別の総量と融合weightです。同じ3kmセル内の150m rankingはfine branchに依存し、HCLだけではstreet-network距離やfine-scale neighbor correlationは改善しません。また両branchは別々に学習され、end-to-end objectiveではありません。

### 4.9 our_model vol5の限界

vol5はST-MoGEの完全再現ではなく、ST-MoGEのspecific/universal expert、attentive gate、regional predictor、HALRをvol4のHCL-ZINBへ移植した独自モデルです。ST-MoGEのpoint MSE headをregional ZINB NLLへ変更し、universal expertもST-MoGEのgraph expertではなくHCL encoderです。したがって、ST-MoGE論文の改善率やablation結果をvol5へ直接当てはめられません。

CECLは意図的に採用していません。expert分離とHCL type alignmentの目的競合を避ける判断ですが、実際に同時利用が性能を悪化させるかは未検証です。`HCLのみ`, `+specific expert`, `+gate`, `+regional head`, `+HALR`, `+CECL` の段階的ablationが必要です。

地域clusterはspecific expertのnode embeddingを最初の2 epochsだけK-meansし、その後固定しています。HALRも論文のiteration単位ではなくepoch単位です。さらに7日strideをepochごとに循環して全target日を訪問しますが、25 full passesではありません。cluster割当、HALR重み、gate値がseedや更新頻度に対して安定しているかは確認できていません。

今回のgateは同一犯罪種内のcluster間差が小さく、regional mixtureが実質的に類似したexpert比率へ収束した可能性があります。cluster別ZINB headが地域差を吸収したためgate差が不要だった可能性もありますが、head出力・cluster構成・gate entropyの追加診断が必要です。

vol5はcoarse MAEと区間幅を改善した一方、NLLとOccurrence F1を悪化させました。150mでもMAE/Brier/ECEは改善した一方、count O/E、PR-AUC、Spearmanは悪化しています。単一指標で採否を決められる結果ではなく、複数seedの信頼区間と用途別の評価基準が必要です。

HCL-ZINB由来のaleatoric variance、fine branchのPoisson近似、coarse/fine branchの独立性、150m prediction intervalがないこと、両branchをjoint trainingしていないことはvol3/vol4から残ります。fine GAT checkpointもvol4から再利用したため、coarse表現に合わせたfine allocationの再最適化はしていません。

学習・評価はseed 42の1 runです。2024年BURGLARY系列の不連続、公開犯罪データの通報・記録バイアス、report delay、street-network距離不足という上位のデータ・モデル限界も解消していません。

### 4.10 実験・評価の限界

現在はGIFとオフライン評価指標を併用しています。GIFはモデル挙動を観察するには有用ですが、予測性能や社会的妥当性を単独で判断するものではありません。評価指標も、観測犯罪データへの当てはまりを測るものであり、犯罪抑止効果や住民影響を直接測るものではありません。

まだ弱い評価:

- AIC / BICやtime-rescaling KSのような、連続時間点過程としての厳密な適合診断。
- 次イベントの時刻・位置・犯罪種を直接予測する指標。
- 犯罪種別に分けた予測性能。現行保存結果は総リスク中心。
- demographic属性に基づく公平性評価。現状はLAPD areaを代理群として使った負担評価に留まる。
- analyst baselineやpatrol allocationを仮定した実運用比較。
- 不確実性区間やbootstrap 95%信頼区間。

また、Predictive Policingの有効性は、犯罪予測の当たり外れだけでは決まりません。警察活動の配置、住民への影響、既存の通報・取締りバイアス、地域負担の偏り、feedback loopを含めて検討する必要があります。

### 4.11 計算・公開上の限界

`datas/` はGit管理外であり、GitHubには公開していません。したがって、第三者が完全再現するには、LA City Open Dataから同じデータを再取得し、ローカル準備スクリプトを実行する必要があります。

予測表やモデル状態も大きいため、多くはGit管理外です。現在GitHubに公開する想定の成果物は、主にREADME類、実験コード、参照文献、選定したGIFです。

CPU実行を前提にしているため、論文通りの毎日再推定や完全な連続時間NLLを全期間・全セルで行うには計算量が大きくなります。今後、GPU利用、空間インデックス最適化、期間分割、キャッシュ設計が必要です。

## 5. 今後の課題

優先度が高いもの:

- ETASのEM推定をより論文に忠実に実装する。
- ETASに隣接セルまたは距離減衰の空間核を入れる。
- STNPP-GATに道路ネットワーク距離を導入する。
- STNPP-GATを経験的遷移行列近似ではなく、連続時間event-sequence NLLで学習する。
- STMGNN-ZINBの公開実装または追加仕様を確認できた場合、未記載ハイパーパラメータとgraph constructionを更新する。
- STMGNN-ZINBについて、HA、STGCN、NB、Gaussian、Truncated Normal baselineを同じLAPD splitで再現する。
- 3km日次共通benchmarkを作り、ETAS/STNPP-GAT系とSTMGNN-ZINBを同じ空間・時間単位で比較する。
- HCLを全日target・5 seedsで再学習し、Hawkes/type/neighbor各要素のablationを行う。
- HCLのST-HSL完全backboneと、ST-SHNによるbinary occurrence版を再現する。
- HCLの `lambda_type`, `lambda_neighbor`, `delta`, `s` をLAPD validationでgrid searchする。
- ST-MoGEを全train target x 50 full epochs・複数seedで再学習し、平均と信頼区間を出す。
- ST-MoGEのspecific/universal expert、attentive gate、regional predictor、CECL、HALRを個別に外すablationを行う。
- ST-MoGEのloss係数、attention heads、dropout、cluster更新頻度をLAPD validationで探索する。
- ST-MoGEのNYC/Chicago元データを用意できた場合、論文と同じregion・crime taxonomyでTable 1-2を再現する。
- 2024年のBURGLARY系列不連続の原因を一次資料で確認し、NIBRS-era dataとの接続方針を決める。
- vol4でHCL-ZINBのHawkes、type loss、neighbor lossを個別に外すablationを行う。
- vol4を全日target・複数seedで再学習し、vol3との差に信頼区間を付ける。
- coarse HCL-ZINBとfine GAT/ETASをjoint objectiveで学習する方法を検討する。
- vol5でspecific expert、gate、regional ZINB head、HALRを段階的に外し、各要素の寄与をablationする。
- vol5へCECLを単独追加し、HCL type alignmentとの勾配競合、性能、gate entropyを比較する。
- vol5を複数seed・全日targetで再学習し、cluster割当とgateの安定性および評価値の信頼区間を確認する。
- vol5のcluster数、更新頻度、HALR temperature、specific expert容量をLAPD validationで探索する。
- vol5のcoarse branchとfine GAT/ETASをend-to-endまたは交互最適化し、総量校正とfine rankingを同時に学習する。
- vol3をend-to-endのjoint likelihoodで学習し、GAT/ETAS側にもepistemic uncertaintyまたはcount varianceを導入する。
- coarse ZINB分布を150mへ確率的に分解するhierarchical allocation modelを検討する。
- vol4を検証し、空間核、report delay、network distanceのいずれかを取り込んだ次版を作る。
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
