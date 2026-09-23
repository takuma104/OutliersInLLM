# Phase 2 実施計画（案 C）: Qwen3.5 縮小版 Track A → Qwen3-0.6B に GA＋GatedNorm 後付け

- 作成: 2026-09-23 / 状態: **承認済み（案 C、2026-09-23）**
- 元計画: [20260923-gatednorm-retrofit-plan.md](./20260923-gatednorm-retrofit-plan.md) §5
- 根拠: [Phase 1 レポート](../reports/phase1-outliers.md) §4

元計画 §5 の設計（KL 蒸留＋外れ値正則化、恒等初期化、学習設定、評価）はそのまま使う。変更点と追加点だけをここに書く。

---

## 1. 段階と狙い

| 段階 | 対象 | 問い | 規模 |
|---|---|---|---|
| **C1** | Qwen3.5-0.8B-Base | dim 0 のシンクは「バイアス的」なので、**ゲート（GatedNorm）よりバイアスで置き換わる**か | 3 アーム × λ 2 水準 × 200M token（＋パイロット） |
| **C2（主実験）** | Qwen3-0.6B-Base | 論文の機構（再スケール因子のシンク＋MA＋attention sink）の上で、**GA＋GatedNorm の後付けがパレート優位**になるか（元計画 H2） | 3〜4 アーム × λ 2 水準 × 200M token（＋パイロット） |

## 2. C1 のアーム（Qwen3.5）

| ID | 構造 | 学習対象 |
|---|---|---|
| A1 | 元のまま | 全重み |
| A2 | ＋GatedNorm（input / post_attention / final norm の 49 個、rank 16、`2σ(·)` で恒等初期化） | 全重み＋ゲート |
| **A5** | ＋**ゼロ初期化バイアス**を、残差 Norm の出力を読む全 Linear に追加（GDN の in_proj_qkv / z / a / b、Attention の q / k / v_proj、gate / up_proj、lm_head） | 全重み＋バイアス |

- λ は A1 のパイロットで、正則化項 R が初期値から約 60% / 90% 下がる 2 水準を選び、A2 / A5 にも同じ値を使う。
- 予測（Phase 1 から）: A5 ≧ A2 ≈ A1。A2 が A5 に勝てば、GatedNorm がバイアス以上の働きをしている証拠になる。

## 3. C2 のアーム（Qwen3-0.6B）

| ID | 構造 | 学習対象 |
|---|---|---|
| B1 | 元のまま | 全重み |
| B2 | ＋GA | 全重み＋GA ゲート |
| B3 | ＋GA＋GatedNorm | 全重み＋ゲート |
| B4（任意） | ＋GA＋GatedNorm＋バイアス | 全重み＋ゲート＋バイアス |

- **GA の後付け**: Attention 出力（o_proj の入力）に `2σ(W_g · x̂ + b)` を掛ける。x̂ は Attention の入力（input_layernorm の出力、GatedNorm がある場合はその出力）。W_g と b はゼロ初期化で、初期状態は厳密に恒等。ゲートの粒度は **head-wise（W_g: 1024→16）と element-wise（1024→2048、Qwen3.5 と同じ）をパイロットで比べて決める**。
- 正則化 R は全トークン（先頭トークンを含む）にかける。Qwen3 の dim 35 は先頭トークンの MA と attention sink を兼ねているので、先頭トークンの MA を消すには、GA が sink の代わりを担える必要がある（B2 / B3 と B1 の比較がその検証）。
- B2 と B3 の差が、GA とは独立した GatedNorm の寄与になる。

## 4. 共通の設定（元計画 §5.2〜5.4 から）

- 損失: `L = KL(p_orig ‖ p_θ) + λ · R`、`R = mean_{token, norm} Σ_j ReLU(|u_j| − τ)²`、`u = x / rms(x)`（残差 Norm の入力）、τ = 8。
- データ: FineWeb-Edu sample-10BT。2048 token に packing。全アームで同じ順序の同じデータ。held-out は別シャードから取る。
- 学習: global batch 128 × 2048（約 262K token/step）、200M token ≈ 763 step。AdamW β = (0.9, 0.95)。元の重みは LR 2e-5、新規パラメータ（ゲート・バイアス）は LR 1e-3。warmup 3%、cosine で 10% まで減衰。WD 0.1 は元の Linear 重みだけにかける。スケジュールは全アームで共通にする。
- 精度: fp32 のマスター重み＋bf16 autocast、gradient checkpointing（non-reentrant）。R はフォワード中にだけ集計し、再計算時は集計しない。
- lm_head の KL / CE は、チャンクごとに勾配を計算する fused 実装で、248K 語彙のロジットを一度に作らない。
- ログ: 100 step ごとに固定プローブ（C4 の 16×2048）で M1〜M3・Linear 入力・ゲート統計・held-out KL を計算し、JSONL と wandb（project `OutliersInLLM`、group `phase2-c1` / `phase2-c2`）に送る。

## 5. 評価（各チェックポイント）

- 機能: WikiText-2 / C4 の PPL、元モデルとの held-out KL、lm-eval zero-shot（ARC-e / c、HellaSwag、PIQA、WinoGrande、LAMBADA）。
- 外れ値: Phase 1 の M1〜M8 一式（`phase1_measure.py` をチェックポイントに適用）。
- 量子化: RTN ΔPPL（W8A8、W4A16、W4A4 NVFP4、A4 / A4-INT、per-kind）。
- ゲートの働き: (i) g ≡ 1 に固定したときの KL 増加、(ii) |y_j| と g_j の相関、(iii) シンク次元の λ_eff の変化。A5 ではバイアスの大きさと、元のシンク方向との対応。
- 推論オーバーヘッド: prefill / decode のレイテンシ比。

## 6. 判定ゲート

- **G1（各段階のパイロット後）**: 恒等性（初期ロジットがビット一致）、スループット、λ に応じて R が動くこと。
- **G2（C1 後）**: A1 / A2 / A5 のパレート比較を中間レポートにまとめ、C2 に進む（C2 は案 C の一部として承認済みなので、判定ではなく報告）。
- **G3（C2 後）**: B3 が B1 に対してパレート優位なら PTQ フェーズ（別計画）の候補にする。

## 7. 日程の目安

| 期間 | 作業 |
|---|---|
| D1–2 | 実装（注入と恒等性テスト、fused KL/CE、正則化 hook、学習スクリプト、プローブ記録）、データ準備、スループット計測 |
| D3 | C1 パイロット（λ、LR、安定性） |
| D4–6 | C1 本番（6 本）＋評価、中間レポート |
| D7 | C2 実装（GA 後付け）とパイロット（ゲート粒度、λ） |
| D8–11 | C2 本番＋評価 |
| D12–14 | 最終レポート、余力があれば延長（最良条件の延長学習、シード反復） |
