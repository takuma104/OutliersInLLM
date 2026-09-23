# 実験計画: 学習済みLLMの外れ値評価と GatedNorm 後付けFT

- 作成: 2026-09-23 / 状態: **承認済み（§8 の決定事項を参照）**
- 前提資料: [ChatGPT-GatedNormのPTQ応用検討](./ChatGPT-GatedNormのPTQ応用検討-20260923-1216.md)
- 参照論文（`docs/papers/`）
  - 2601.22966 — Outlier-Driven Rescaling / GatedNorm（以下「GatedNorm論文」）
  - 2306.12929 — Quantizable Transformers（B.6 に Gated Attention 後付けFTの例）
  - 2603.05498 — The Spike, the Sparse and the Sink（massive activation と attention sink の解剖）

---

## 0. 要約

| フェーズ | 期間 | 内容 |
|---|---|---|
| Phase 1 | 2〜3日 | Qwen3.5-0.8B-Base（Hybrid GDN＋Gated Attention）と Qwen3-0.6B-Base（全層Softmax＋QK Norm）の Weight/Activation 外れ値を、同じツール・同じデータで計測する。ここで作る計測ライブラリは Phase 2 でも使う。最後に Phase 2 に進むか判定する（G0） |
| Phase 2 | 1〜2週 | Qwen3.5-0.8B-Base に**恒等初期化の GatedNorm** を後付けしてFTし、外れ値を抑えられるかを検証する |

**Phase 2 の設計の核**:
通常のLM損失（CE）だけでFTしても、既存の外れ値を消す圧力はほとんど掛かりません。既存の外れ値は、すでに損失の低い解の一部だからです（重み減衰の効果も、FTのLRとステップ数ではごく小さい）。
そこで主実験（Track A）は「**元モデルへのKL蒸留＋外れ値正則化**」とし、同じ正則化強度のもとで **元アーキテクチャ vs GatedNorm付き** の「外れ値削減量 vs 機能劣化」のパレート曲線を比べます。
GatedNorm論文の主張は「外れ値は再スケーリングを担う機能部品であり、GatedNorm はその代替経路になる」というものです。これが後付けでも成り立つなら、GatedNorm 側のほうが少ない劣化で外れ値を落とせるはずです。
ChatGPT案の「通常FTとの比較」は副実験（Track B）として残します。

---

## 1. 問いと仮説

**RQ1（Phase 1）**: Gated Attention (GA) を持つ Qwen3.5 と持たない Qwen3 では、外れ値の種類（attention sink / massive activation (MA) / residual sink）と所在（層・次元・どの Linear 入力か）がどう違うか。

- H1a: Qwen3-0.6B は、先頭トークンと区切りトークンの中間層に MA（数百以上）を持ち、attention sink も強い（2603.05498 の Qwen3-8B と同じ傾向）。
- H1b: Qwen3.5-0.8B は attention sink と MA が弱い。一方で固定次元の residual sink は残り、その次元の実効Norm重み `1+w` は極端に小さい（GatedNorm論文の Qwen3-Next の観察と同じ）。
- H1c: 活性の量子化誤差は、両モデルとも down_proj 入力が支配的。residual sink は Norm 重みで潰されるので、通常の A 量子化への直接の影響は限定的（§3.4）。

**RQ2（Phase 2 主）**: 学習済みの Qwen3.5 に GatedNorm を後付けすると、residual sink を**より少ない機能劣化で**除去できるか。

- H2: 外れ値の削減量を揃えて比べると、GatedNorm 付きのほうが元モデルからの KL / ΔPPL が小さい（パレート優位）。

**RQ3（Phase 2 副）**: 外れ値抑制の圧力を掛けない通常FTでも、GatedNorm を入れるだけで外れ値は減るか。

- H3: 0.5B token 程度では差はほぼ出ないと予想する。差が出れば強い結果になる。

**RQ4（Phase 2 副）**: 元の重みを凍結し、ゲートだけを学習させても効果はあるか。

---

## 2. 対象モデル

| | Qwen3.5-0.8B-Base | Qwen3-0.6B-Base |
|---|---|---|
| 層構成 | 24層 = GDN 18 ＋ Full Attn 6（3:1） | 28層（全層 Full Attn） |
| hidden / FFN | 1024 / 3584 | 1024 / 3072 |
| Full Attention | 8 heads × 256, KV 2, **出力ゲート (GA)**, QK Norm, partial RoPE 25% | 16 heads × 128, KV 8, QK Norm |
| 線形注意 | Gated DeltaNet（out_proj の前に `RMSNormGated`＝Norm×SiLU(z)） | なし |
| RMSNorm | zero-centered: `x̂ · (1 + w)` | 通常: `x̂ · w` |
| 語彙 / tie | 248,320 / tied | 151,936 / tied |
| パラメータ | テキスト部 約0.75B（うち埋め込み 0.25B）＋vision/MTP | 0.60B（うち埋め込み 0.16B） |
| 読み込み | `AutoModelForCausalLM` → `Qwen3_5ForCausalLM`（visual/mtp は無視） | `Qwen3ForCausalLM` |

- 主対象は Base 版にする。Instruct 版は後学習（RL 等）の影響が大きく、FT時の挙動が読みにくいため。Instruct 版は Phase 1 で参考として計測するだけ（任意）。
- 両モデルとも hidden=1024 なので、比率系の指標はそのまま比較できる。
- **交絡に注意**: 両モデルは GA 以外にも、Hybrid（線形注意）の有無、学習データとトークン数、マルチモーダル事前学習、zero-centered Norm（と Norm への weight decay）、head_dim、語彙が違う。Phase 1 のモデル間比較は記述的な比較にとどめ、差を「GA の効果」とは断定しない。GatedNorm論文の Table 1 でも、Hybrid 化だけで最大活性は 6000→1800 に下がっている。
- 語彙が違うので、LM品質をモデル間で比べるときは PPL ではなく **bits-per-byte (BPB)** を使う。同じモデルの前後比較は PPL / ΔPPL でよい。
- Qwen3.5 は GatedNorm論文の「Hybrid＋GA」ベースライン（Table 1 の row 5）と同じ構成で、後付け実験は row 5→22 の比較に当たる。ただし論文のスクラッチ学習でも、最大活性は 1100→780 程度しか下がっていない。そのため**最大値だけでなく residual sink に固有の指標**（M2〜M4）を主に見る。

---

## 3. 共通基盤

### 3.1 環境

- RTX 5090 (32GB) ×1、RAM 62GB、uv 環境（torch 2.14 / transformers 5.17）
- 追加予定: `datasets`, `pandas`, `pyarrow`, `matplotlib`, `pytest`, `wandb`, `flash-linear-attention`（GDN 用 Triton カーネル）。Phase 2 では `lm-eval` と、必要なら `liger-kernel` も入れる。
- **Phase 1 の初日に、`fla` が sm_120 で動くかと、torch フォールバックと数値が一致するかを確認する。** Phase 1 は forward だけなのでフォールバックでも間に合うが、Phase 2 の学習ではカーネルが必須になる見込み。

### 3.2 データ

| 用途 | データ | 備考 |
|---|---|---|
| 外れ値計測用のプローブ | C4 validation (en) 128文書 × 2048 token | **文書の先頭から切り出す**（先頭トークンの効果を自然な形で観測するため）。C4 はキャッシュ済み |
| PPL / BPB / fake-quant ΔPPL | WikiText-2 test | |
| 入力非依存性の確認（任意） | 日本語・コード 各32本程度 | residual sink が入力に依存しないかを見る |
| Phase 2 学習 | FineWeb-Edu sample-10BT（必要なシャードだけ取得） | held-out シャードを別に確保する |

### 3.3 指標定義

| ID | 名称 | 定義 | 用途 |
|---|---|---|---|
| M1 | 最大活性 | 各層の残差 h の max\|h_j\|。先頭トークン / 区切りトークン / その他に分けて集計。各ブロック出力（attn/GDN, MLP）についても同様 | MA の検出、step-up/step-down ブロックの特定 |
| M2 | 残差シンクスコア | 次元ごとの平均 \|h_j\|（先頭トークンを除く）の max/median。加えて、各トークンの argmax 次元が固定次元と一致する割合 | residual sink の有無と強さ |
| M3 | Norm入力のピーク比 | u = x / rms(x)（‖u‖² = d, \|u_j\| ≤ √d = 32）について、max_j\|u_j\| のトークン分布（p50/p99）と、上位k次元のエネルギー比 Σ_topk u_j² / d | GatedNorm論文 App. A.1 でいう「再スケールのレバー」そのもの。Phase 2 の正則化対象 |
| M4 | 実効Norm重み | λ_eff = 1+w（Qwen3.5）/ w（Qwen3）。最小値と、シンク次元での値 | 論文の特徴的なパターン（シンク次元で λ≈0.004） |
| M5 | Linear入力の分布 | 全 nn.Linear 入力について absmax、チャネル別 absmax の max/median、トークン単位の超過尖度、6σ を超えるチャネル数 | 実際に量子化される活性の難しさ |
| M6 | 量子化誤差 | 活性 fake-quant の SQNR [dB]（INT8/INT4 per-token, FP8 E4M3 per-token, NVFP4 block16）。層出力の相対誤差 ‖XWᵀ − Q(X)Q(W)ᵀ‖ / ‖XWᵀ‖（W4A16 / W8A8 / W4A4） | 外れ値指標と量子化の難しさの対応をとる |
| M7 | Attention sink | Full Attn 層の sink ratio（先頭トークンへの平均注意が 0.3 を超えるヘッドの割合; Gu et al. 2025）、先頭トークンの value ノルム。Qwen3.5 では GA ゲート値の分布も見る | Attention 側の再スケール |
| M8 | 重みの外れ値 | 行列ごとの超過尖度、absmax/std、入力チャネル別ノルム。W-only fake-quant の相対誤差（INT8/INT4 per-channel, INT4 g128, NVFP4） | Weight 側の評価 |

- MA の判定は Sun et al. 2024 の基準（絶対値 100 超、かつ中央値の約1000倍）を参考にし、小型モデル向けに相対基準も併用する。
- 量子化の評価対象から外すもの: 埋め込み / lm_head、GDN の小さなパラメータ（in_proj_a/b, conv1d, A_log, dt_bias）。これらは計測だけ行う。

### 3.4 注意: 「residual sink が大きい」と「量子化が難しい」は同じではない

通常の W×A 量子化では、残差ストリーム自体は量子化しません。また、次元 d の residual sink は Norm 後に λ_eff≈0 を掛けられるので、qkv / gate_up の入力には現れにくくなります。residual sink が量子化に効いてくるのは、次のような経路です。

1. RMSNorm の分母を通して、他のチャネルのスケールを縮めている
2. **λ を重みに畳み込む手法**（QuaRot / SpinQuant などの回転系）では、量子化される活性が u = x / rms(x) になり、シンクがそのまま現れる
3. 残差加算や KV を低精度で扱う場合

そのため、常に **M3（u のピーク比）と M5 / M6（実際の Linear 入力）の両方**を計測し、「residual sink が減った＝量子化が改善した」とは扱わない。

---

## 4. Phase 1: 外れ値評価（2〜3日）

### 4.1 計測・解析の項目

- **A. 深さ方向のプロファイル**: 層ごとの上位3活性（M1）とブロック出力の分解から、step-up / step-down ブロックを特定する（2603.05498 Fig.1 と同じ形式）。
- **B. 残差シンクの特定**: M2〜M4 から、シンク次元・それが現れる層・Norm 重みとの対応を調べる。
- **C. Linear 入力と重み**: M5・M6・M8 をモジュール種別ごとに集計する。種別は qkv/in_proj_qkv, in_proj_z, o_proj/out_proj, gate_up, down_proj。
- **D. Attention**: M7。eager attention での計測は、このときだけ行う。
- **E. シンクの機能を確かめる安価な因果プローブ**: 各 Norm 入力のシンク次元に対して、次の3つを行い ΔPPL を測る。
  - (i) 平均値で置き換える（mean ablation）
  - (ii) 0 にする（zero ablation）
  - (iii) τ·rms でクランプする

  (i) では無害なのに (ii) で壊れるなら、そのシンクは「入力に依存しない再スケール因子」だと言える（論文の主張と一致）。この結果から、Phase 2 でシンクを除去する難しさを見積もる。
- **F. RTN fake-quant ΔPPL（推奨・軽量）**: W8A8 (INT8)、W4A16 (g128)、W4A8、W4A4 NVFP4（活性は論文と同じく dynamic per-token）。加えて、モジュール種別ごとに活性だけを量子化し、ボトルネックの場所を特定する。**PTQ 手法の比較ではなく**、外れ値指標が量子化の難しさと対応しているかを確認するためのもの。
- G. （任意）Instruct 版と Base 版の比較。
- H. （任意）step-up ブロックの MLP で、super weight と U_k の Frobenius ノルム（2603.05498 Fig.3）を調べる。

実装方針: forward hook でストリーミング統計を取り、活性の全量は保存しない。ヒストグラムと SQNR 用には小さなサンプルだけ保存する。

### 4.2 日程

| 日 | 作業 |
|---|---|
| Day 1 | 環境整備（uv add、fla の動作確認）、モデル DL、Qwen3.5 のテキスト部だけの読み込み確認、プローブデータ作成、hook / 統計 / fake-quant ライブラリとテスト |
| Day 2 | M1〜M8 の計測と図の作成、RTN sweep（F） |
| Day 3 | 因果プローブ（E）、（任意）G/H、レポート作成、G0 判定 |

### 4.3 成果物

- `docs/reports/phase1-outliers.md`（図表付きレポート）
- `results/phase1/*.parquet`（生の統計）
- Phase 2 で再利用する計測ライブラリ

### 4.4 判定ゲート G0（閾値は暫定）

- **Go**: Qwen3.5 に residual sink がはっきりある。目安は、多くの層で M2 ≥ 10、M4 でシンク次元の λ_eff ≪ 1、M3 の p50 ≳ 10。
- **弱い / 無い場合**は、次のどちらかに方針を変えるか相談する。
  - (a) Qwen3-0.6B に GA＋GatedNorm を後付けする（Quantizable Transformers B.6 に近い設定。外れ値が大きいので改善の余地も大きい）
  - (b) 目的を down_proj 入力など、別の外れ値の抑制に絞る

---

## 5. Phase 2: GatedNorm 後付けFT（1〜2週）

### 5.1 GatedNorm モジュール

```python
class GatedNorm(nn.Module):
    """既存の RMSNorm の出力に、低ランクの要素ごとゲートを掛ける（恒等初期化）。"""

    def __init__(self, norm: nn.Module, hidden_size: int, rank: int = 16) -> None:
        super().__init__()
        self.norm = norm
        self.down = nn.Linear(hidden_size, rank, bias=False)  # 通常の初期化
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.zeros_(self.up.weight)  # 初期状態で g ≡ 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        g = 2.0 * torch.sigmoid(self.up(F.silu(self.down(y))))
        return y * g
```

- 2σ(z)⊙y = σ(z)⊙(2y) なので、これは論文の GatedNorm で Norm 重みを2倍にしたものと等価。**アーキテクチャの変更ではなく、初期化の工夫**にすぎない。
- sigmoid(0) = 0.5 は厳密に表現できるので、初期状態の出力は元モデルとビット単位で一致するはず。これをユニットテストで確認する。
- 配置（主設定）: 各層の `input_layernorm` と `post_attention_layernorm`（48個）、および最終 `norm`（1個）。論文の「全 Norm の後」に合わせる。q_norm / k_norm と GDN 内の `RMSNormGated` は対象外とし、必要なら別実験にする。
- rank 16 の場合、1個あたり 2·1024·16 = 32.8K パラメータ、計 約1.6M（テキスト部の約0.2%）。
- W_up をゼロ初期化するので、ゲートの LR は元の重みより高くする。

### 5.2 損失

**Track A（主）**: `L = KL(p_orig ‖ p_θ) + λ · R`

- 教師は元モデル（bf16, no grad, オンライン推論）。初期状態では KL = 0 で勾配も 0 なので、**正則化を掛けない限りモデルは動かない**。そのため、学習中の変化はすべて「外れ値を減らす圧力」と「機能維持」の綱引きで決まり、学習データのドメインシフトの影響を受けにくい。
- 外れ値正則化:

  ```
  u = x / rms(x)   （対象 Norm の入力 x、トークンごと）
  R = mean_{token, norm} Σ_j ReLU(|u_j| − τ)²     τ = 8（暫定。ガウスなら1024次元の最大値は ≈3.5）
  ```

  - スケールに依存しない（残差全体を拡大・縮小しても R は変わらない）ので、論文 App. A.1 の「外れ値の割合 r = |x_d| / ‖x‖₂」を直接抑えることになる。
  - **逃げ道**: シンクを複数の次元に分散させれば、ピークを下げながら再スケールのレバーを残せる。この場合も量子化には有利だが、M3 の top-k エネルギー比で監視する。
  - 代替案: 尖度ペナルティ、あるいは目標の外れ値水準を制約とした双対更新（外れ値水準を揃えたうえで KL を直接比べられる）。

**Track B（副）**: `L = CE` のみ（通常の継続事前学習）。

### 5.3 実験条件

**Track A（主）**

| ID | アーキテクチャ | 学習対象 | λ | tokens |
|---|---|---|---|---|
| A0 | 元モデル | – | – | – |
| A1-{lo,mid,hi} | 元のまま | 全重み | 3水準 | 各 ~200M |
| A2-{lo,mid,hi} | ＋GatedNorm | 全重み＋ゲート | A1 と同じ | 各 ~200M |
| A3 | ＋GatedNorm | ゲートのみ（元の重みは凍結） | mid | ~200M |
| A4（任意） | ＋PreAffine（λ₁ = 1 で初期化） | 全重み | mid | ~200M |

- λ の水準はパイロットで決める。A1 で、R が初期値から約30% / 60% / 90% 下がる値を探し、同じ λ を A2 にも使う。
- 元アーキテクチャの Norm 重みは、すでに「入力に依存しない要素ごとのスケール」なので、**A1 は無条件ゲートの対照も兼ねる**（2603.05498 Table 7: 無条件ゲートでは sink は消えない）。A4 は、論文でいう「パラメータへの吸収」系の対照。

**Track B（副）**

| ID | アーキテクチャ | 学習対象 | 損失 | tokens |
|---|---|---|---|---|
| B1 | 元のまま | 全重み | CE | ~500M |
| B2 | ＋GatedNorm | 全重み＋ゲート | CE | ~500M |

参考: Quantizable Transformers B.6 の後付けFTは約520M token。

### 5.4 学習設定（初期値。パイロットで調整する）

- seq 2048 で packing、global batch 128 seq（約262K token/step）。200M token ≈ 760 step。
- AdamW β = (0.9, 0.95)
  - LR: 元の重み 2e-5（sweep: 1e-5 / 3e-5 / 1e-4）、ゲート 1e-3（sweep: 3e-4 / 1e-3 / 3e-3）
  - スケジュール: warmup 3%、cosine で 10% まで減衰
  - WD 0.1 は Linear 重みだけに掛ける（Norm とゲートは 0）
- **LR スケジュールは全条件で共通にする**。論文 Fig.4 では LR の減衰とともに外れ値が減っており、スケジュールが交絡要因になるため。
- 精度: fp32 master weight ＋ bf16 autocast、gradient checkpointing（`use_reentrant=False`。hook で正則化項を集めるので、再計算時にフックが二重に呼ばれる点に注意）。
- 語彙が 248K あるので、ロジットを一度に作ると巨大になる（micro-batch 8×2048 で bf16 約8GB）。CE / KL はシーケンス方向にチャンク分割して計算する（または Liger Kernel の fused linear CE / JSD を使う）。
- 100 step ごとに固定のプローブバッチ（16×2048）で、M1〜M3・M5（down_proj ほか）・ゲート統計・held-out KL を記録し、外れ値の推移を曲線として残す。記録先はローカルの JSONL と wandb の両方。

### 5.5 評価（各 checkpoint）

- **機能**: WikiText-2 / C4 val の PPL、元モデルとの held-out KL、lm-eval の zero-shot（ARC-e/c, HellaSwag, PIQA, WinoGrande, LAMBADA）。任意で MMLU 5-shot。lm-eval には `HFLM(pretrained=model)` でゲート注入済みのモデルを渡す。
- **外れ値**: Phase 1 の M1〜M8 一式（学習前後で比較）。
- **ゲートが実際に使われているか**:
  - (i) g を 1 に固定したときに KL がどれだけ上がるか（ゲートが本当に再スケールを担っているか）
  - (ii) |y_j| と g_j の相関（論文: |y| が大きい次元ほどゲートが小さい）
  - (iii) シンク次元の λ_eff が 1 に戻るか
- **量子化プローブ**: M6 と RTN ΔPPL（Phase 1 と同じ設定）。
- **推論オーバーヘッド**: prefill / decode のレイテンシ比。論文では hidden 2048 で 8% なので、hidden 1024 ではそれ以上になる見込み。

### 5.6 成功基準（暫定。Phase 1 の結果を見て数値を確定する）

- **主**: A2 のパレート曲線が A1 を支配する。目安として、ΔPPL が同程度（WikiText-2 で元モデル比 +2% 以内）のときに M3 の p99 や M2 が2倍以上小さい。または、外れ値水準が同じときに KL が半分以下。
- **機構**: A2 で §5.5 の (i)〜(iii) を確認できる。
- **量子化**: A2 で qkv / in_proj / gate_up 入力の NVFP4 SQNR が改善し、RTN W4A4 の ΔPPL が縮む。
- **副**: B2 と B1 の差（RQ3）、A3 の到達点（RQ4）。

### 5.7 計算予算と日程

**スループットの見積もり**: テキスト部 0.75B で、学習コストは約4.5 GFLOP/token（勾配チェックポイントの再計算と教師の forward を足すと、その約1.5倍）。RTX 5090 の実効性能を 60〜80 TFLOPS と仮定すると、CE のみで約1万 tok/s、教師ありで約7千 tok/s（1日あたり 0.6〜0.9B token）。**実装初日に実測して置き換える。**

| 項目 | tokens | GPU時間の目安 |
|---|---|---|
| パイロット ×約8本 | 各 30〜50M | 約12h |
| Track A ×6本 | 各 200M | 約45h |
| A3 | 200M | 約6h |
| Track B ×2本 | 各 500M | 約28h |
| 評価（約12 checkpoint） | – | 約10h |
| **合計** | **約2.3B** | **約100h（約4日）** |

**2週間版の日程**

| 日 | 作業 |
|---|---|
| D1–2 | 実装（GatedNorm の注入と恒等性テスト、学習スクリプト、チャンク分割 KL/CE、正則化 hook、プローブ記録）とスループット計測 |
| D3 | パイロット（LR、λ の範囲、安定性、そもそも外れ値が動くかの確認） |
| D4–6 | Track A（夜間も連続実行） |
| D7–8 | Track B と A3 |
| D9–10 | 評価・解析・中間レポート |
| D11–14 | 延長: 最良条件を 1B token まで延ばす（外れ値がまだ減り続けているか）、λ=mid のシード反復、A4（PreAffine）、rank / τ のアブレーション、最終レポート |

**1週間版**: Track A を λ 2水準（4本）に、Track B を各 300M token に縮め、延長は行わない。

### 5.8 判定ゲート

- **G1（パイロット後）**: 初期ロジットが元モデルと一致すること（恒等性）、スループット、λ に応じて R が動くことを確認する。
  - A1 でも KL がほぼ 0 のまま R が大きく下がる場合、「後付けの状況では外れ値は機能的ではない」ことを示唆する。論文の仮説に反する方向の結果だが、それ自体も成果になる。
- **G2（Track A 後）**: A2 がパレート優位なら、PTQ フェーズ（SmoothQuant / GPTQ / NVFP4 など、別の計画書で扱う）に進む。

---

## 6. リスクと対策

| リスク | 対策 |
|---|---|
| fla（Triton）が sm_120 で動かない / 遅い | Phase 1 の Day 1 に確認する。動かない場合は torch フォールバックで、token 数や seq 長を縮める |
| Qwen3.5 のチェックポイントが VLM 形式 | テキスト部だけを読み込む。GatedNorm 付きモデルは `save_pretrained` の往復に頼らず、state_dict と注入コードで保存・復元する |
| 248K 語彙のロジットでメモリが足りない | CE / KL をチャンク分割、または fused kernel を使う |
| 外れ値が別の場所へ移るだけ（down_proj 入力、GDN out_proj 入力、複数次元への分散） | 全 Linear 入力と M3 の top-k エネルギー比を常に監視する |
| シード1本では差がノイズに埋もれる | 主比較（λ=mid の A1/A2）はシード反復する。プローブセットでの分散も報告する |
| ゲートを重みに畳み込めず、推論時にも残る | レイテンシを実測する。回転系 PTQ（QuaRot / SpinQuant）は Norm 重みの融合を前提にしており、要素ごとのゲートは回転と可換でないので相性が悪い。PTQ フェーズで要検討 |

---

## 7. コード構成案

```
src/outliers/
  models.py      # モデル読み込み（Qwen3.5 はテキスト部のみ）、モジュール種別の正規化
  data.py        # プローブ / 評価 / 学習データ
  hooks.py       # forward hook の登録
  stats.py       # ストリーミング統計（M1〜M5, M7）
  quant.py       # fake-quant（INT8/INT4/FP8/NVFP4）、SQNR、層出力誤差
  gatednorm.py   # GatedNorm / PreAffine の注入と取り外し
  losses.py      # チャンク分割 CE/KL、外れ値正則化
scripts/
  phase1_measure.py, phase1_ablate_sink.py, phase1_quant_probe.py
  train_retrofit.py, eval_lm.py
tests/           # 恒等性、統計の正しさ（numpy と照合）、fake-quant の既知値、チャンク KL = 全体 KL
results/         # .gitignore 対象
```

---

## 8. 決定事項（2026-09-23 承認）

1. 主対象は Base 版とする（Instruct 版は Phase 1 の参考計測のみ）。
2. Track A（KL＋外れ値正則化によるパレート比較）を主実験とする。
3. RTN fake-quant の簡易プローブ（§4.1 F と §5.5）を入れる（PTQ 手法の比較はしない）。
4. 学習データは FineWeb-Edu（英語）のみとする。
5. ログはローカル（JSONL / parquet）に加えて **wandb にも記録する**。Phase 2 の学習曲線とプローブ指標は必ず wandb に送る。Phase 1 の計測結果も run として残す。
