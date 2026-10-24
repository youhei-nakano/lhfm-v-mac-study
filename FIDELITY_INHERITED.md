# Fidelity実測後の状態（凍結計画FIDELITY.mdは変更しない）

## 検証済み

- 自作原幅モデルcount18,637,493と公式値一致。縮小幅65,893。history encoder、3residual/MLP、attention7slot/4head、fine/coarse motion各2block、future planner4block、全head/aux/state feedback/paired rotations/low-rank/grouped mixを保持。
- cadenceテスト: planner1回、controller10回、source readout20回、feedback19回、qkv10回、aux1回。最後の不要state更新だけ省略する公式挙動と一致。
- 二次upwind operatorは独立scalar stencilと一致。24次source込みaffine積分は独立augmented dense matrix_expとdoubleで一致。source/w/image gradient、zero-w極限、適応内部分割、schedule endpoints/半減境界をテスト。小operatorのCPU/MPS最大差0（atol/rtol2e-5）。
- 64×64/10+10/batch1 pilot成功。縮小のためConvGRU/一次upwind/Euler/損失省略/transport gateへ置換していない。
- 学習6runは全件完了、validation最小EMA選択。公式MNIST train SHA256一致、元splitと生成seed/位置更新/RNG順を自作で対応。公式test full819,200,096 bytesのSHA256が固定commitの値に一致。
- 評価settingsはrange2、FP32 map、uniform7、sample covariance、transpose、crop3、FP64空間平均→FP32 channel平均、frame/sequence平均。MSE/MAEはrawspatialsum、PSNRはclip後frame単位。

## 意図した縮小・差

- widths=(8,16,24)、latent16、controller/field16、rank8、key/value8、motion16、coarse8、plan8、correction16。その他depth、expansion、memory/head数、64px/10+10は原設定を保持。
- 40update/batch1、3初期化seed。元1.25M/batch16ではない。元schedule式へのindex線形写像を明示。EMAは元通り.999/optimizer updateのため40更新では時定数1000より短く、ほぼ初期EMAの性能である点を隠さない。
- validation先頭8生成系列（元1024）、検証は10/20/30/40へ間引き。選択則は最小validation rawMSEのまま。testはofficial先頭16/10000だけ。
- FP32/MPS/NCHW/eager、PyTorch2.14.1。元BF16/CUDA/channels-last/compile/記載2.14.0とは違う。
- SciPy1.15.3の公式Mac wheelはmacOS27でloader error。新venvだけ1.16.3へ更新して同一定義を使用。NumPy1.26.4は保持。公式依存版とのbitwise再現は未検証。
- 公式coarse headはtraining-only条件付き。自作はeval時もcoarseを計算するが予測には使わないため、forecast数式には影響しない。計測コストはこの実装のscopeであり公式inference FLOPsと比較しない。
- 同じ初期化分布規則だがmodule生成/RNG順・加算順が異なる。公式の同seed初期tensorやbitwise出力は未再現。

## 未検証・実行していない項目

- 公式repositoryのコードは.txt読取参照のみ。import/run/公式checkpoint読込はしていない。独立式/構造照合とパラメータ数一致は、公式全modelの同一重みnumerical equivalenceの証明ではない。
- 公式testと訓練digitの画像identity/writerの独立性は未確認（元protocolをそのまま使用）。train/validationのindex集合はdisjoint。
- 公式10000 test全体、原幅の学習、原1.25M更新、別環境、長期bit再現性、原benchmark性能は未検証。

強い公式照合を行う具体的追加actionは、commit979a41de542f7b190549fc94a8803a56530e4a00のsrc/lhfm_v/frozen_expm.py::frozen_exponential_stepを隔離CPUで固定FP64入力に対し実行し、output/gradientを自作と比較すること、および同commit model.py::PhysicalFastPlanMovingMNISTへ意味対応で同一重みを設定して小固定clipのforwardを比較することです。第三者公式ソースを実行する具体的承認がまだないため実施していません。今回の自作による縮小確認の完了と、この未検証事項を分けて報告します。
