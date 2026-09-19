# Kaggriculture 模倣学習・強化学習コード一覧

`D:\kaggriculture` にある、模倣学習（BC）と強化学習（RL）のコードの地図です。
何がどのファイルにあり、どの順に動かし、何が効いて何が効かなかったかをまとめています。
作業の経緯は [WORKLOG_2026-09-17.md](WORKLOG_2026-09-17.md)、SpaTaro 期の詳細は [REPORT.md](REPORT.md) にあります。

実行環境は Anaconda の `ml311`（`C:\Users\Notak\anaconda3\envs\ml311\python.exe`）、GPU は手元の RTX 3060（12GB）。
コマンドはすべて `D:\kaggriculture` から実行します。

---

## 0. 全体像

```
 Kaggle 日次データ (episodes zip)
        │ extract_tapes.py / spataro_meta.py / extract_team_replays.py
        ▼
 試合記録 ──► build_dataset.py ──► BC 教師データ (npz + meta.csv)
                  (kag/replay.py, kag/dataset.py, kag/features.py)
                                        │
                                        ▼
                               train.py  (kag/model.py = BCNet)
                         汎用モデル ──► 専門化 (--init)
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
     calibrate.py              強化学習（3系統）              評価
     export_np.py            rl_train.py  (GRPO)           screen.py
     package.py              rl_ppo.py    (PPO+GAE)        (kag/evaluate.py)
        │                    self_imitate.py (選別自己模倣)
        ▼                        └ kag/rl_batch.py（まとめ実行）
 submission/*.tar.gz
 (kag/agents/bc.py + kag/np_model.py, torch 不要)
```

### 結果の要約

| 手法 | コード | 結果 |
| --- | --- | --- |
| SpaTaro 単独 BC | `train.py` | ラダー 1285.7（現在値） |
| Majkel1337 単独 BC（d256） | `train.py` | ラダー 1001.6 |
| **汎用 → Majkel 専門化** | `train.py --init` | **ラダー 1599.0**（BC の最高） |
| GRPO 型 PPO（価値関数なし） | `rl_train.py` | 0.506 → 0.405〜0.440（**悪化**、960試合） |
| PPO＋価値関数＋GAE | `rl_ppo.py` | 評価 0.677 → 0.693（**横ばい**、5,808試合・11時間） |
| 選別自己模倣 | `self_imitate.py` | 0.422 → 0.031〜0.133（**大幅悪化**） |

効いたのは「幅広いチームで汎用モデル → 目的のチームで専門化」の一点だけでした（§2-7）。
強化学習は3系統とも、1試合約7,000回の判断に手がかりが試合結果1つしかない、という同じ壁に当たりました（§3-5）。

---

## 1. フォルダ構成（模倣学習・強化学習に関係するもの）

```
kag/                       ランタイムと共通部品（提出物にも同梱される）
  replay.py                リプレイ読み込み、座席ごとの (観測, 行動) 取り出し
  features.py              観測 → テンソル、行動 → intent ラベル（候補 1,707）
  dataset.py               1座席ぶんのリプレイ → BC 学習配列
  model.py                 BCNet（Transformer ＋ 候補スコア ＋ 市場ヘッド ＋ 価値ヘッド）
  np_model.py              BCNet の numpy 版（提出用、torch 不要）
  agents/bc.py             BC エージェント本体（build_ctx / decide、実行層、ガード）
  agents/tape.py           テープ（録画再生）エージェント
  agents/registry.py       エージェント指定（JSON）→ 呼び出し可能オブジェクト
  evaluate.py              座席入れ替えの対戦評価、Wilson 区間
  rl_rollout.py            GRPO 用のロールアウトワーカー（torch 非依存）
  rl_batch.py              まとめ実行：K 試合を 1 回の GPU 推論で進める

scripts/
  extract_tapes.py         日次 zip → 座席ごとのテープと索引 index.csv
  spataro_meta.py          1チームの試合一覧を index.csv 群から集める
  extract_team_replays.py  1チームのリプレイ本体を zip から取り出す
  build_dataset.py         BC 教師データを作る（チーム・勝ち試合で絞り込み可）
  train.py                 BC 学習（--init で専門化、--resume で再開、--value-weight で価値ヘッド）
  calibrate.py             市場ヘッドのしきい値と PASS 補正を検証データで決める
  export_np.py             torch チェックポイント → numpy 重み
  package.py               提出用 tar.gz を組み立てる
  rl_train.py              強化学習①：GRPO 型 PPO（価値関数なし）
  rl_ppo.py                強化学習②：PPO ＋ 価値関数 ＋ GAE
  self_imitate.py          強化学習③：選別自己模倣（生成と選別）
  si_merge.py              ③の並列シャードを統合し、全体で選別
  si_label.py              ③の古い all.csv に対戦相手の列を後付け
  screen.py                複数候補を同じ seed・同じ相手で一括比較（主力の評価）
  eval_suite.py, eval.py   単一候補の評価（初期に使用）
  eval_server_runs.py      サーバー学習分の較正・書き出し・評価をまとめて実行

server/                    IS 計算サーバー（A100）用一式。手順は server/README.md
```

---

## 2. 模倣学習（BC）

### 2-1. データ取得

Kaggle の日次データセット（`kaggriculture-episodes-<日付>.zip`）から作ります。

| スクリプト | 入力 → 出力 |
| --- | --- |
| `extract_tapes.py <zip> <out_dir>` | zip → `<out_dir>/tapes/<試合>_s<座席>.json`（行動列）と `index.csv`（チーム名・所持金など） |
| `spataro_meta.py [チーム名]` | `data/d09XX/index.csv` 群 → `data/spataro/<チーム>_episodes.csv` |
| `extract_team_replays.py [チーム名]` | 上の CSV → `data/spataro/replays/<日付>/<試合>.json`（リプレイ本体） |

他チームのエージェントログは API で取れません（403）。日次 zip が唯一の情報源です。

### 2-2. 特徴量とラベル：intent 方式（`kag/features.py`、`kag/dataset.py`）

1手ごとの移動（北・南・東・西）を当てるのではなく、**ユニットが次に行う操作とその対象タイル（intent）**を当てます。
移動と合法性は実行層（`kag/agents/bc.py`）が受け持ちます。フォーラムで「生の移動の一致率は7%」と報告されていたための設計です。

- **候補空間 `N_CAND = 1707`**：`タイル × 17操作`（1,700）＋ 倉庫操作 6 ＋ PASS 1
- 毎回、合法な候補だけをマスクで残し、その中での順位付け（listwise）として学習
- ステップ単位の入力：100 タイルの量子化特徴、ユニット配置、既に狙われているタイル数、合法マスク、全体特徴 112 次元
- ユニット単位の入力：位置・所持品などの特徴 16 次元、所持品ゲート 6 ビット
- 市場のラベル：品目ごとの売買数量を 21 個の分類ヘッドのクラスに変換（`qty_class`、`sell_class`）

`kag/replay.py` の注意：`steps[t][seat]` の観測は t−1 の処理後、行動は t−1 の観測への応答です。
したがって s 手目の判断は `(steps[s].observation, steps[s+1].action)` の組になります。

### 2-3. データセット作成（`scripts/build_dataset.py`）

```
python scripts/build_dataset.py <episodes.zip> <index.csv> <out_dir> [最低所持金] [チーム名]
```

- `<out_dir>/<試合>_s<座席>.npz` と `meta.csv`（episode, seat, path, reward, opp_reward, steps, rows）を出力
- 第5引数でチームを絞り込み（`"*"` で全チーム）。環境変数 `BUILD_WINS_ONLY=1` で勝ち試合のみ

作成済みのデータ：

| フォルダ | 中身 |
| --- | --- |
| `data/bc_spataro` | SpaTaro 510 試合 |
| `data/bc_majkel` | Majkel1337 886 試合 |
| `data/bc_top110` | 89 チームの上位試合 3,309 軌跡（所持金 11 万以上、7 日分、約 13GB をメモリに載せる） |
| `data/bc_mmpq` | M & M & P & Q 336 試合（真似できず失敗） |

### 2-4. モデル：BCNet（`kag/model.py`）

- **エンコーダ**：100 タイル ＋ 全体トークン 1 個の Transformer（d=256・6 層・4 ヘッドで 6.14M パラメータ）
- **ユニットの候補スコア**：タイル埋め込み ＋ ユニット埋め込み ＋ ユニットとタイルの組の特徴（`pair_feats`）を分解した形で計算。倉庫操作と PASS も同じ枠で採点
- **拾う個数ヘッド**：PICKUP の個数を分類
- **市場ヘッド**：品目ごとの分類ヘッド 21 個（全体埋め込みから）
- **価値ヘッド `v_mlp`**：全体埋め込みとタイル平均から tanh で −1〜+1。強化学習の critic 用に後から追加
- **スタイル条件付け**（`n_style > 0`、未使用）：どのチームを真似ているかの埋め込み

`value()` を持たない古いチェックポイントも読めるよう、`train.py --init` と `kag/agents/bc.py` は
`strict=False` で読み込み、欠けているのが `v_mlp.*` だけのときに限り許容します。

### 2-5. 学習（`scripts/train.py`）

```
python scripts/train.py <data_dir> <out_dir> --d 256 --layers 6 --bs 256 --epochs 12 --lr 6e-4
```

損失は 4 つの和です（`losses()`）。

| 項 | 内容 |
| --- | --- |
| ユニット | 合法候補内のクロスエントロピー（listwise）。行ごとに試合の重みを掛ける |
| 拾う個数 | PICKUP の行だけのクロスエントロピー |
| 市場 | 21 ヘッドのクロスエントロピーの平均（`--market-weight`） |
| 価値 | `(自分の所持金 − 相手の所持金) / 50,000` を ±1 に丸めた目標との二乗誤差（`--value-weight`、既定 0） |

- **成績で重み付け**：`clip(所持金 / 中央値, 0.6, 1.6)`、負け試合はさらに ×0.7。ゼロにはしない
- **検証は試合単位で分割**（判断単位で分けると同じ試合の手が両側に漏れる）。少量データで検証が空になるときの退避処理あり
- `--init <ckpt>`：既存の重みから開始（専門化・自己模倣で使用）
- `--resume`：`<out_dir>/last.pt` から再開（サーバーの `low` QoS は中断・再投入されるため）
- `--max-traj N`：所持金上位 N 軌跡だけを使う
- bf16 autocast、出力は `best.pt`（検証一致率が最良）・`last.pt`・`val_best.json`

**効いた構成（汎用 → 専門化）**。`server/submit_general.sh` の中身と同じです：

```
# 1. 汎用：89 チーム 3,309 軌跡（A100 で 1 時間 20 分、手元では RAM 不足で不可）
python scripts/train.py data/bc_top110 runs/gen_d256 --d 256 --layers 6 --epochs 12 --bs 256 --lr 6e-4
# 2. 専門化：Majkel1337 886 試合（10 分）
python scripts/train.py data/bc_majkel runs/spec_majkel_d256 --d 256 --layers 6 --epochs 6 --bs 256 --lr 1.5e-4 --init runs/gen_d256/best.pt
```

`runs/v_lr03/best.pt` は、専門化を弱めた版（学習率 0.3 倍）に価値ヘッドを付けて学習し直したもので、強化学習の出発点に使いました。

### 2-6. 較正・書き出し・提出

```
python scripts/calibrate.py <run>/best.pt data/bc_majkel      # しきい値を best.pt に書き込む
python scripts/export_np.py <run>/best.pt <run>/final.npz
python scripts/package.py <run>/final.npz submission/<名前>.tar.gz --kwargs "{\"calibrated\": false, \"feed_topup\": false}"
```

- `calibrate.py`：市場ヘッドは「何か起きるか」の F1 が最大になる確率しきい値で発火（argmax だと稀な売買を出さない）。
  PASS は予測率が教師の率に合うようロジットを補正。**メモリ対策で PASS の較正は 30 万行に間引き**（全行だと 13.7GB 要求して落ちる）
- `package.py`：`main.py` ＋ `kag` のランタイム ＋ `weights.npz` ＋ `config.json` を tar.gz に。提出物は numpy のみで動き、1 手 約 40ms
- 提出時の実体は `kag/agents/bc.py` の `BCPolicy`。`act()` は `build_ctx()`（観測→入力）と `decide()`（出力→行動）に分かれています。
  実行層にはユニットを目標へ歩かせる処理と、餌やり・世話・最終日のガードが入っています

### 2-7. サーバー（IS 計算サーバー A100）

`server/README.md` に手順と安全規則があります。要点：

- 必ず `--qos=low`、ジョブは **GPU 学習だけ**（自己対戦・評価は投げない。ガイドに「GPU 使用が主体でないジョブは停止される場合があります」）
- `uv` はログインノードで環境構築だけ。同時実行は 2 本まで
- `train_bc.sbatch` が `--resume` を自動付与、`submit_general.sh` は汎用→専門化を `--dependency=afterok` で連結
- 学習済みの `best.pt` だけを持ち帰り、`scripts/eval_server_runs.py runs_server` で較正・書き出し・評価を一括実行

### 2-8. 模倣学習で分かったこと

1. **一致率は強さを予測しない**：d128 と d256 は一致率 0.9142 と 0.9154 なのに、実戦は 2.5 倍違った
2. **データを増やすだけ・モデルを大きくするだけでは効かない**：Majkel 単独 886 試合は 0 勝 / 64 試合
3. **幅広いチームで下地を作ってから専門化する**と効く（13% → 36〜61%）
4. **お手本の強さより真似しやすさ**：平均所持金が 7k 高い M & M & P & Q は一致率 0.777 で全敗
5. 評価は同じ seed で同時に比べた順位だけを信用する（同じモデルが seed の組で 35.9%〜65.6% まで振れた）

---

## 3. 強化学習

3 系統とも、模倣学習済みの方策を出発点にしています。

### 3-1. GRPO 型 PPO（`scripts/rl_train.py`、`kag/rl_rollout.py`）

```
python scripts/rl_train.py --init <ckpt> [--iters 60 --groups 4 --k 6 --procs 8]
```

- 1 反復 = G グループ × K 試合。1 グループは (seed, 相手, 座席) を固定し、K 試合はサンプリングだけが違う
- **アドバンテージ = グループ内で標準化した試合結果**（価値関数なし）。その試合の全判断に同じ値を配る
- 更新前の対数確率で PPO のクリップ（0.2）＋ 模倣方策からの KL ペナルティ（β=0.02、合法分布上で厳密計算）
- ロールアウトは `rl_rollout.py` のワーカーが numpy で 1 試合ずつ実行。**torch を読み込まない**（numpy の MKL と torch の OpenMP が同居すると落ちる）

**結果：40 反復・960 試合で 0.506 → 0.405〜0.440（悪化）**。1 更新あたり 24 試合、価値関数なしでは手がかりが足りない。

### 3-2. まとめ実行（`kag/rl_batch.py`）

1 試合ずつだと 1 手の 97% が GPU 推論（110.7ms、特徴量は 1.2ms）で、小さすぎる行列計算の待ち時間でした。

- K 試合を横並びで 1 手ずつ進め、K 試合分の入力を **1 回の GPU 推論**にまとめる
- 方策の判断コードは `BCPolicy.build_ctx()` / `decide()` をそのまま使うので、**学習中の試合と提出時の挙動が一致**
- `opp="self"` で自分の複製と対戦し、相手側の推論も同じまとめ呼び出しに同乗
- 価値ヘッドの出力 V(s) も同時に返す（GAE 用）

```
python -m kag.rl_batch <ckpt.pt> 8 --check     # 速度測定と、1試合ずつ実行した場合との行動一致の確認
```

**1 試合 33〜170 秒 → 約 5〜6 秒**。200 手先まで行動が完全一致することを確認済み。
まとめた後のボトルネックは Python 側（`env.step` 34.5%・推論 25.9%・`build_ctx` 20.1%・`decide` 19.0%）で、GPU 使用率は約 25% です。
公式シミュレータのゲームルール本体は 1 手 0.12ms しかなく、`env.step` の大半は `jsonschema` 検証と `structify` の梱包でした。

### 3-3. PPO ＋ 価値関数 ＋ GAE（`scripts/rl_ppo.py`）

```
python scripts/rl_ppo.py --init runs/v_lr03/best.pt --out runs/ppo1 --iters 50 --games 48
```

- ロールアウトは `rl_batch.play_batch`（1 反復 48 試合、半分は自己対戦、半分は上位 8 チームのテープ）
- **GAE**（γ=1.0、λ=0.95）でアドバンテージを計算し、正規化
- 損失 = クリップ付き方策損失（0.2）＋ 価値損失（係数 0.5）＋ 模倣方策からの KL（β=0.02）
- 5 反復ごとに固定パネル（4 チーム × 8 試合）で評価、チェックポイント `it005.pt`〜
- 出発点に価値ヘッドがないと停止する（`v_mlp.*` を要求）

**結果：121 反復・5,808 試合・約 11 時間で、評価 0.677（前半）→ 0.693（後半）**。振れ幅 0.594〜0.781 の中に埋もれて改善なし。
一方で KL は 0.009 → 0.236 まで広がった：**方策は動いたが、良くはならなかった**。チェックポイントは `runs/ppo1/` に残っています。

### 3-4. 選別自己模倣（`scripts/self_imitate.py`、`si_merge.py`、`si_label.py`）

勾配ではなく**選別**を学習信号にする方法です。ランダム性を入れて試合を生成し、良かった試合だけを通常の模倣学習にかけます。

```
# 1. 生成（2 シャード並列。15.7GB の手元 PC では 2 本が上限）
python scripts/self_imitate.py --init runs/v_lr03/best.pt --out data/si_round1/sh0 --games 400 --batch 24 --no-select --seed-start 2000000 --seed 3
python scripts/self_imitate.py --init runs/v_lr03/best.pt --out data/si_round1/sh1 --games 300 --batch 16 --no-select --seed-start 2100000 --seed 4
# 2. 統合して選別（相手ごとに上位 25%）
python scripts/si_merge.py data/si_r1_strat --shards data/si_round1/sh0 data/si_round1/sh1 --stratify --keep 0.25
# 3. 微調整
python scripts/train.py data/si_r1_strat runs/si1_strat --d 256 --layers 6 --heads 4 --bs 256 --epochs 3 --lr 2e-5 --init runs/v_lr03/best.pt
```

- 記録はバッチごとに npz へ書き出してメモリから捨てる（全試合を抱えていた初版は 192 試合で 10GB を要求しスワップした）
- `all.csv` をバッチごとに更新するので、途中で止めても失われず、再実行すると済んだ試合を飛ばす
- `si_merge.py` の選び方：既定（全体の所持金差上位）／`--stratify`（相手ごとに上位）／`--only self`（自己対戦のみ）／`--random SEED`（対照実験）
- `all.csv` に対戦相手の列 `opp` を記録。記録前のデータは `si_label.py` が乱数の引き直しで復元

**選別の落とし穴**：所持金差で上位を取ると、700 試合中 174 試合がテープ相手、自己対戦は 341 試合中 1 試合だけでした。
所持金差は「自分がうまく打ったか」ではなく「相手が崩れたか」を測っていたためで、層別選別を追加しました。

**結果（seed 840000〜、各 128 試合）**

| モデル | 選別 | 更新量 | スコア |
| --- | --- | --- | --- |
| v_lr03（出発点） | — | — | 0.422 |
| si1_gentle | 層別 25% | 1 エポック 5e-6 | 0.133 |
| si1_self | 自己対戦のみ 25% | 3 エポック 2e-5 | 0.078 |
| si1_strat | 層別 25% | 3 エポック 2e-5 | 0.070 |
| si1_rand | ランダム 25% | 3 エポック 2e-5 | 0.031 |

選別ありはランダムの 2 倍で信号は存在するが、**出発点から動かすこと自体の損失**のほうが大きい。
生成時はランダム性（τ=0.5）で打たせ、提出時は最尤を選ぶため、抽出行動を真似ると確率分布がなだらかになり切れ味が落ちます。

### 3-5. 強化学習で分かったこと

- 3 系統とも同じ壁：**1 試合 約 7,000 回の判断に対して、手がかりは試合結果 1 つ**
- 参考にした上位解法は 150 万〜1,250 万試合。手元の速度（2 並列で 1 試合 3.3 秒）では 150 万試合に 57 日かかる
- フォーラムの外部データ：3 億環境ステップ（約 42 万試合）回した参加者が最終資金 8 万で飽和（手元の BC は 9.2 万、公開ルールベース v47 は 9.5 万）
- 残された唯一の筋：**強化学習は日単位の計画（約 30 回の判断）だけを決め、実行は決定論的に行う**。複数の参加者が独立に到達した形で、手がかりの密度が約 240 倍になる。未実装

---

## 4. 評価

| ファイル | 役割 |
| --- | --- |
| `kag/evaluate.py` | 対戦評価の本体。seed ごとに座席を入れ替えて 2 試合、勝ち・負け・引き分けだけを数える。`alternate=True` で 1 seed 1 試合（座席を交互） |
| `scripts/screen.py` | **主力**。複数の候補を同じ seed・同じ相手で 1 つのプール（11 並列）にかけて比較 |
| `kag/agents/registry.py` | JSON 指定 → エージェント。`bc`（`weights` 必須）、`tape`、`file`（提出形式の `main.py`）、`starter` など |
| `kag/agents/tape.py` | テープエージェント（上位チームの行動列を再生）。**seed が変わると崩れる**ので物差しとしては甘い |
| `scripts/eval_server_runs.py` | サーバーで学習した複数モデルを一括で較正・書き出し・評価 |

```
python scripts/screen.py \
  --cand '{"kind":"bc","weights":"runs/v_lr03/best.pt","kwargs":{"calibrated":false,"feed_topup":false}}' \
  --tapes none \
  --opp-file sub1_2619=opponents/user_sub1_2619/main.py \
  --opp-file v41_2378=opponents/user_v41_2378/main.py \
  --alternate --seeds 64 --seed-start <未使用のseed> --save runs/<名前>.json
```

- 候補の JSON に二重引用符が入るので、**PowerShell の `Start-Process` ではなく Git Bash から実行**する（引用符が落ちて JSON が壊れる）
- 32〜64 試合の比較は接戦を見分けられない。5 ポイントの差を示すには約 385 試合、2 ポイントなら約 2,400 試合
- 別の seed の組同士の数字は比べない

---

## 5. 使っていないもの・デバッグ用

| ファイル | 用途 |
| --- | --- |
| `scripts/debug_feed.py` | 1 試合を再生し、指定日にユニットが何を選び FEED が何位だったかを表示（終盤の餓死の調査用） |
| `scripts/diagnose.py` | 1 試合を両座席の時系列で表示 |
| `scripts/tape_ladder.py` | テープ同士の総当たり |
| `scripts/final_eval_spa.py` | SpaTaro 期の最終比較（REPORT.md §8-5） |
| `scripts/eval.py`、`scripts/eval_suite.py` | 初期の単一候補評価（`screen.py` に置き換え） |
| `scripts/ladder_audit.py`、`scripts/ladder_diverge.py` | 模倣学習・強化学習ではなく、提出した v47 のラダー試合の分析用（WORKLOG §17） |

---

## 6. 既知の注意点

- **古いチェックポイントと価値ヘッド**：`v_mlp.*` のない `best.pt` は `train.py --init` と `kag/agents/bc.py` では読めるが、`rl_ppo.py` は停止する（critic が必要なため）
- **`calibrate.py` のメモリ**：PASS 較正を間引かないと 210 万行 × 1,707 候補で 13.7GB を要求して黙って落ちる。落ちたモデルはしきい値なし（argmax）で動く
- **`self_imitate.py` のメモリ**：1 プロセスで 3〜5GB。15.7GB の PC では 2 並列まで
- **ロールアウトワーカーで torch を読まない**：numpy の MKL と torch の OpenMP が衝突する（`rl_rollout.py` が torch 非依存なのはこのため）
- **Kaggle の API**：competitions 系は OAuth が必要で、古い `kaggle.json` があると OAuth が効かない。
  `credentials.json` だけを入れたフォルダを `KAGGLE_CONFIG_DIR` に指定し、`python -m kaggle ...` で実行する（`kaggle.exe` は Windows のアプリ制御でブロック）
