# Phase 1 レポート: 学習済み小型 LLM の外れ値評価（Qwen3.5-0.8B-Base / Qwen3-0.6B-Base）

- 作成: 2026-09-23 / 対象計画: [docs/plans/20260923-gatednorm-retrofit-plan.md](../plans/20260923-gatednorm-retrofit-plan.md) §4
- 生データ: `results/phase1/<model>/*.parquet`（git 管理外）、wandb: project `takuma104/OutliersInLLM`, group `phase1`
- 集計表の全体: `scripts/phase1_summary.py` の出力（本文の表はその抜粋）

---

## 0. 要約

<!-- SUMMARY -->

---

## 1. セットアップ

| 項目 | 内容 |
|---|---|
| モデル | Qwen/Qwen3.5-0.8B-Base（`Qwen3_5ForCausalLM`、テキスト部 0.75B。visual / MTP 重みは読み込まない）、Qwen/Qwen3-0.6B-Base。bf16、sdpa（M7 のみ eager） |
| プローブ | C4 validation shard 0 から、両トークナイザで 2048 token 以上ある文書を 128 本（seed 0）選び、**文書先頭から** 2048 token を切り出す。文書 index は `data/probe_c4_docs.json` |
| LM 評価 | WikiText-2 test（`"\n\n".join` → 2048 token の非重複窓）。PPL と bits-per-byte（BPB: バイト数はバイトレベル BPE のトークン文字列長から厳密に計算） |
| トークン種別 | first = 位置 0、delim = 空白と `.,;:!?` だけから成り `.` か改行を含むトークン、other = それ以外 |
| 計測 | forward hook によるストリーミング統計（`src/outliers/{hooks,stats}.py`）。残差ストリームは各 RMSNorm（input_layernorm / post_attention_layernorm / 最終 norm）の入力として読む |
| fla | `flash-linear-attention` 0.5.2 の GDN カーネルは sm_120（RTX 5090）で動作。torch 参照実装との差は fp32 参照に対する KL で 4.2e-4（fla） vs 4.1e-4（torch）と同等、backward も一致（相対誤差 0.5%）。forward 78K tok/s、grad-ckpt 付き fwd+bwd 19K tok/s（lm_head 抜き、8×2048） |

### 計画からの変更・補足

- **M6 の SQNR / 層出力誤差はトークン平均**（各トークンの雑音/信号比を平均してから dB 化）を主指標にした。全トークンのエネルギー加重だと、MA を持つ少数のトークンが和を支配し（例: Qwen3 L2 down_proj の入力は 3888 の値 1 個で SQNR が 33 dB に見える）、per-token 量子化の難しさを表さないため。加重版は `*_global` 列に残した。
- **M5 の「6σ を超えるチャネル数」**は、262K トークンにわたる absmax を全体 σ と比べると裾の重い入力ではほぼ全チャネルが該当してしまい（down_proj で 3584 中 3540）、情報がない。代わりに **チャネル RMS が中央値の 6 倍を超えるチャネル数**（系統的な外れ値チャネル）を主に用いる。
- Qwen3.5 の `o_proj` 種別は、Full Attention の `o_proj` と GDN の `out_proj` を合わせたもの。性質が大きく違うので、図表では `o_proj (attn)` と `out_proj (GDN)` に分けた。
- M1 の "other" トークンの最大値は、Qwen3 で**位置 1 の 2 トークン**（`"44"`, `"11"` で始まる文書の 2 桁目。1 桁目と同じトークンが続き、先頭トークンのように振る舞う）に支配されていた。図には p99.9 も併記した。

---

## 2. 結果

### 2.1 LM 品質

<!-- LM -->

WikiText-2 の BPB は 0.851 と 0.845 でほぼ同じ。以下の外れ値の差は「LM としての品質差」によるものではない。

### 2.2 深さ方向プロファイルと massive activation（A, M1, H）

![depth profile](figs/phase1/depth_profile.png)
![block outputs](figs/phase1/block_outputs.png)

<!-- M1 -->

- **Qwen3-0.6B**: 典型的な MA。先頭トークンの dim 35 が **L2 MLP（step-up）で一気に 7200** になり、L27 MLP（step-down）で打ち消されるまで残差に残る。128 系列すべての先頭トークンが MA の基準（|h|>100 かつ中央値の 1000 倍超）を満たす。step-up の MLP では、down_proj 入力の ch 55 が 3888 に達し、**super weight** W_down[35, 55] = 0.58（行列 std の 24 倍）と、二次形式ノルム ‖U_35‖_F が中央値の 8.2 倍（step-down の L26/L27 でも 6.3 倍 / 17.6 倍）になる（2603.05498 Fig.3 と同じ構図）。区切りトークンの最大値は 10 → 650 と深さとともに増えるが、他のトークンの p99.9 とほぼ重なり、区切りトークンだけが特別に大きいわけではない。
- **Qwen3.5-0.8B**: **MA は無い**。残差の最大値は全層で 21 以下、先頭トークンは 7 以下（MA 基準を満たすトークンは 0）。step-up / step-down ブロックも無い。ただし ‖U_k‖_F の最大は**全 24 層で dim 0**（中央値の 1.9〜3.6 倍）で、MLP が一貫して dim 0 に書き込んでいる。L1 と L3 の down_proj には dim 0 行に super weight 的な要素（std の 21.6 倍、11.6 倍）がある。

<!-- H -->

### 2.3 残差シンク（B, M2〜M4）

![residual sink](figs/phase1/residual_sink.png)
![sink dims](figs/phase1/sink_dims.png)
![peak histogram](figs/phase1/peak_hist.png)

<!-- SINK -->

- **両モデルとも固定次元の残差シンクがはっきりある。** Qwen3 は dim 35（MA と同じ次元）、Qwen3.5 は dim 0。M2 スコアの中央値は 38 と 18、10 を超える残差読み出しは 55/56 と 46/48。
- **M3（u = x/rms(x) のピーク）**: 非先頭トークンの p50 の中央値は Qwen3 17.2、Qwen3.5 10.6（最大 19.5）。上限 √d = 32 に対して、どちらも「1 次元で RMS の 10〜20 倍」を常に持つ。上位 4 次元のエネルギー比は Qwen3 0.43、Qwen3.5 0.26。Qwen3 の先頭トークンは peak ≈ 31.8（ほぼ one-hot）で、RMS も他トークンの 86 倍。
- **M4（実効 Norm 重み）は両モデルで挙動が違う。**
  - Qwen3: シンク次元 35 の λ は **post_attention_layernorm（MLP 入力）で中央値 2.8e-5**（1e-6〜1e-3）とほぼ 0。input_layernorm（Attention 入力）では 0.6 程度で、シンクは Attention 側には通す（先頭トークンの key を sink にするため）が MLP には通さない。GatedNorm 論文の「外れ値次元の λ は極端に小さい（0.006 vs 1）」に一致する。
  - Qwen3.5: シンク次元 0 の λ_eff = 1+w は **input_layernorm で 1.18、post_attention_layernorm で 0.82**（中央値）。**Norm 重みでは抑えられていない**。論文が Qwen3-Next で報告した λ ≈ 0.004 のパターンは、Qwen3.5-0.8B では再現しない。
- その結果、Qwen3.5 では **dim 0 が Linear 入力の外れ値チャネルとしてそのまま現れる**。qkv / in_proj_z / gate_up の入力で ch 0 の absmax は 15〜32（他チャネルの中央値 4〜5）、RMS 比 3〜4.7。重み側では ch 0 の入力列ノルムが全チャネル中で最小（中央値の 0.3〜0.7 倍）で、弱い「重み側での抑制」はあるが、λ ≈ 0 ほど極端ではない。§3.4 で想定した「residual sink は Norm 重みで潰されるので A 量子化には直接効きにくい」は、**Qwen3 では成り立つが、Qwen3.5 では成り立たない**。

### 2.4 Attention（D, M7）

![attention sink](figs/phase1/attention_sink.png)

<!-- ATTN -->

- Qwen3: sink ratio 0.75（Gu et al. の基準 >0.3）、先頭トークンへの平均注意 0.46。先頭トークンの value ノルムは他の 0.14 倍で、典型的な「何もしない」シンク。
- Qwen3.5: **sink ratio 0**（先頭トークンへの注意は平均 0.04、最大 0.18）。先頭トークンの value ノルムはむしろ他の 1.8 倍。代わりに出力ゲート（GA）の値は平均 0.2 と小さく、33% の要素が 0.1 未満。「何もしない」をゲートが担っている（GatedNorm 論文の gating-based rescaling）。

### 2.5 Linear 入力と重み（C, M5, M6, M8）

![linear inputs](figs/phase1/linear_inputs.png)
![weights](figs/phase1/weights.png)

<!-- LINEAR -->

- 絶対値: Qwen3 の Linear 入力は MA の影響で最大 3888（L2 down_proj）、qkv / gate_up でも最大 600 程度。Qwen3.5 は最大 71（L14 down_proj の先頭トークン）で、多くは 20〜40。
- 系統的な外れ値チャネル（チャネル RMS が中央値の 6 倍超）: Qwen3.5 は **GDN out_proj 入力（中央値 15 本、RMS 比 22.5）と o_proj（11.5 本）** が多く、qkv / gate_up（3〜4 本、主に ch 0）、down_proj（1 本）は少ない。Qwen3 は qkv（12 本、RMS 比 30）、gate_up（6.5 本、RMS 比 31）が多い。
- GDN out_proj 入力 = RMSNormGated(core)·SiLU(z) の外れ値チャネルは、特定の (head, dim) にだけ現れ、ヘッド間で共有の Norm 重み（128 次元）とは相関しない（相関 −0.6〜0）。GDN 内部の per-head 正規化の後に生じるもの。計画ではこの Norm を GatedNorm の対象外にしている（§5.1）ので、Phase 2 では監視対象に加える必要がある。
- per-token INT4 の SQNR（中央値）はどの種別も 5〜8 dB で、種別間の差は小さい。後述の ΔPPL の差（数十倍）を SQNR の中央値だけでは説明できない。
- 重み（M8）: 両モデルとも大差ない。超過尖度の中央値は 1〜3、INT4 g128 の相対誤差は 0.12〜0.13、NVFP4 は 0.095。

### 2.6 RTN fake-quant ΔPPL（F）

![rtn](figs/phase1/rtn.png)

<!-- RTN -->

- 全体: W8A8（INT8）で Qwen3.5 +1.3% / Qwen3 +2.7%、W4A16（INT4 g128）で +24% / +46%、W4A4（NVFP4）で +18% / +26%。**Qwen3.5 のほうが一貫して量子化に強い**（パラメータ数が多いことの影響は切り分けていない）。
- 活性だけ 4bit: NVFP4（block 16）なら +7% / +10% で済むが、**per-token INT4 では両モデルとも崩壊**（PPL 2436 / 12048）。
- 1 種別ずつ INT4 化したときのボトルネックがモデルで違う:
  - Qwen3: **gate_up（PPL 3911）≫ qkv（159）≫ down_proj（51）≫ o_proj（14.3）**。Norm 出力を読む Linear が壊れる。
  - Qwen3.5: **o_proj/out_proj（110）≫ down_proj（28）> gate_up（23）> qkv（15.8）> in_proj_z（13.8）**。Norm 出力側（qkv / gate_up）は比較的軽い。

<!-- RTN_LAYERS -->

### 2.7 因果プローブ: シンク次元のアブレーション（E）

<!-- ABLATION -->

### 2.8 Base と Instruct の比較（G）

<!-- INSTRUCT -->

---

## 3. 仮説の検証

<!-- HYPOTHESES -->

---

## 4. G0 判定と Phase 2 への示唆

<!-- G0 -->

---

## 5. 再現手順

```bash
uv sync
uv run python scripts/check_fla.py                      # fla の動作確認 → results/phase1/check_fla.json
uv run python scripts/phase1_make_probe.py              # data/probe_c4_docs.json（git 管理済み）
for m in qwen3.5-0.8b qwen3-0.6b; do
  uv run python scripts/phase1_measure.py --model $m     # M1〜M8, PPL/BPB
  uv run python scripts/phase1_quant_probe.py --model $m # RTN ΔPPL
  uv run python scripts/phase1_ablate_sink.py --model $m # シンクのアブレーション
  uv run python scripts/phase1_quant_layers.py --model $m
  uv run python scripts/phase1_step_up.py --model $m     # ‖U_k‖_F, super weight
done
uv run python scripts/phase1_measure.py --model qwen3.5-0.8b-instruct   # G（任意）
uv run python scripts/phase1_measure.py --model qwen3-0.6b-instruct
uv run python scripts/phase1_plots.py                   # docs/reports/figs/phase1/*.png
uv run python scripts/phase1_summary.py                 # 集計表（Markdown）
uv run pytest                                           # 単体テスト + 実モデルの統合テスト（GPU）
```
