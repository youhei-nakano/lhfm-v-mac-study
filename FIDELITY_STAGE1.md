# 現在の3000更新計画と原LHFM-Vの対応

FIDELITY_INHERITED.mdは前段の40更新smoke記録であり、そこにある6run・validation8・test16という数は今回の3000更新計画の条件ではない。現在の条件は以下とPLAN.md、RECOVERY_PLAN.md、RECOVERY_CPU_PLAN.mdによる。

| 項目 | 今回 | 忠実性・差異 |
|---|---|---|
| 元資料 | 原論文・公式commit979a41de542f7b190549fc94a8803a56530e4a00 | 公式sourceは読取参照のみ。実行して同重み照合したものではない |
| データ | 公式MNIST28px、2数字、64px、観測10+未来10 | 原設定を維持。原generatorの手順を自作対応。公式実装との画素単位一致は未検証 |
| train/validation | 原55k/5k index split、seed271100 | index集合disjoint。writer/公式testとのidentity独立は未確認 |
| 学習系列 | 3000系列、absoluteID0..2999、data seed270829 | 各モデル同系列順・batch1。元1.25M更新/batch16より縮小 |
| validation/test | val32系列seed271109、35固定検証点。新officialtest16..143の128系列 | 元val1024/test10000より小さいsubset。公式test0..15は前段で見たため除外。両モデル選択後のみ開く |
| 構造 | history画像/差分/xy、attention7slot4head、future planner、fine/coarse motion、controller、feedback、source/transport、aux coarse | ConvGRU等への置換なし。depth・構成意味を保持 |
| 幅 | (8,16,24)、latent/controller/field16、rank8、key/value8、motion16、coarse/plan8、correction16 | 65,893params。原幅の自作instantiation18,637,493は公式報告と一致。幅の縮小が大きい |
| 空間operator | 二次upwind、zero-exterior、unbounded transport | 一次upwind・tanh上限・周期境界へ置換しない |
| 積分 | 半frame dt0.5、source込み24次shifted exponential、adaptive subdivisions | Eulerへ置換しない。zero-wは厳密にJ+dt r。シンプレクティック保存保証は主張しない |
| loss | predictionMSE + .001 source energy + .0001 transportTV + .05 coarseMSE | 全項を保持。offではtransportがゼロのためTVもゼロ |
| optimizer | AdamW、beta2.999、wd1e-4、eps1e-8、clip1 | 元lr/beta1式をabsolute original index1..1250000へ線形写像して3000へ再尺度化。EMA.999を保持 |
| checkpoint | 事前35検証点の最小EMA raw pixel MSE | testで選ばない。EMA初期値寄与は3000で4.97%、早い選択点ではそれ以上 |
| 指標 | raw spatial-sumMSE/MAE、range2/FP32/uniform7/samplecov SSIM、clip後frame PSNR | 新しい都合のよい正規化へ変更しない。SSIMのtranspose/crop3/FP64空間平均→FP32channel平均を保持 |
| 実行 | eagerFP32/NCHW、M1 8GB、2CPUthreads | 元CUDA/BF16/channels-last/compileとは異なる。transport1..787MPS、その後CPU。offはCPU。FP32でもdevice数値差あり |
| 速度 | 両選択checkpointを同CPU、batch1、3warmup+10synchronizedforward、入力転送除外 | 自作evalではauxも計算し予測には不使用。そのコストを含む。このlatencyを原FLOPsやCUDA速度と直接比較しない |
| baseline | transport-offの学習時ablationとlast-frame | SOTA比較ではない。総65,893同数。offはtransportMLP816+correction544+history offset32channel800=2,160 scalarがprediction/lossに影響せず、有効63,733。計算量同一ではない |

## 独立検証

二次stencilのscalar/dense照合、source込みaugmented matrix_expとのFP64照合、image/source/transport gradient、zero限界、adaptive pieces、構造cadence、原幅param count、schedule、metric normalization、split/EMA、CPU save/resumeをテスト。MPS復旧では元の重複74更新のloss/gradient normが一致。これらは公式全モデルの同重み数値一致を証明しない。

## 復旧と証拠の扱い

元650更新のswap停止とMPS復旧787更新のswap停止を残す。最終の一つの軌跡は元1..576、MPS復旧577..787、CPU復旧788..3000。probeや失敗した重複更新を独立seedや追加の性能証拠に数えない。全体swapは他処理も含み、guard停止を学習単独OOMと断定しない。RSS値は高水位peakであり現在residentと区別する。

single seed・小幅・小subset・短いbudget・混在した学習deviceの制約がある。元benchmarkスコアの再現、統計的有意、一般的優位性は主張しない。
