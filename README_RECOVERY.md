# 忠実な縮小LHFM-V：保護停止からの復旧

このディレクトリの原論文対応・数式は FIDELITY_INHERITED.md、モデル・訓練計画は PLAN.md、復旧変更は RECOVERY_PLAN.md を参照。過去の簡易video-miniとは異なる自作の原構造縮小実装であり、原論文のbenchmark性能を再現したとは主張しない。

## 実行環境

Python3.12、requirements-lock.txt の公式レジストリ依存版を専用venvへ導入。macOS MPS/FP32。公式研究repoのコードは読み取り参照で、import/runしていない。license付与・公開pushは行っていない。

```
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python smoke.py --output smoke-new
```

## フォルダと主要関数

- architecture.py: Faithful がhistory/difference/xy encoder、attention memory、future planner、source/transport、coarse head、half-frame recurrent feedbackを接続。
- operators.py: second-order upwind zero-exterior operator、sourceを含む24次 shifted exponential、適応内部分割、全loss、原scheduleの3000更新への再尺度化。
- dataset.py:公式MNISTのindex splitと絶対系列IDによるMoving-MNIST生成。download_data.py は公式URL・checksumを検証。
- metrics.py:原定義の空間和MSE/MAE・range2 SSIM・clip PSNR。
- stage1.py:凍結した元実行。runs/stage1は650更新でswap保護停止した元試行。
- recovery.py:同じ数学の再開runner。576完全状態からtransportを再開、offはscratch。毎更新の原子的保存とgraph/cache解放のみ変更。
- runs/recovery1:採用する一つの最終軌跡。originalの1..576をprefixとし、577以降は復旧で再計算。元の失敗577..650は別監査として残す。
- recovery_probe.py:CPU save/resumeの完全一致と実状態からの保護付きMPS replay。recovery_replay_comparison.json:元試行との10更新損失・勾配ノルム一致。
- summarize_recovery.py:全raw配列から再集計、固定例の画像・GIF、曲線・機械可読表。

## CPU復旧への追記

MPS復旧1も787更新で全体swap保護停止。以降はRECOVERY_CPU_PLAN.mdに従い同一数学のCPU実行へ切替。transportは1..787がMPS、788以降CPU、offはCPU。両者の最終評価・latencyはCPU。同一デバイス訓練の完全な対照とは主張しない。CPU復旧は recovery2.py --device cpu --output runs/recovery-cpu-new、元の787完全状態もZIPへ同梱。結果はruns/recovery2、再集計はsummarize_recovery2.py。

## データ取得と復旧実行

公式データはZIPに含めない。download_data.pyの引数は --help を確認し、提供元とSHAは同ファイルとdata/test_source.jsonを参照。ダウンロードは約819MBの公式testを含む。新しい場所に置いた場合は --official-test で明示する。

```
.venv/bin/python download_data.py --mnist --official-test
.venv/bin/python recovery2.py --device cpu --output runs/recovery-cpu-new --official-test data/mnist_test_seq.npy
.venv/bin/python summarize_recovery2.py runs/recovery-cpu-new
```

上記は同梱の787完全状態と元ログから同じseed・schedule・系列IDで再開する。保存先は未存在の新規ディレクトリを使う。元のstopped資料を上書きしない。時間上限なし、RSS1.5GiB/driver1GiB/全体swap増加256MiBの保護は継続。自動リトライループはない。CPUを使う場合も同じ保護だが、device数値差・速度を別に報告する必要がある。

## 小さく検証・画像再生成

小さなfixtureのsmoke.pyは16px、元の64px実験の代替ではない。checkpointと単例の再生成は外部データ不要。

```
.venv/bin/python regenerate.py runs/recovery1/transport/best_ema.pt runs/recovery1/example_official_16.npz --output regenerated.npz
```

学習更新数・幅・データ等を変える際は別の計画と出力にする。最終testを設計選択に使わない。元650更新・40更新の結果を完成した比較に混ぜない。

可視化は0〜1へclipして保存する。MSE/MAEはraw予測、PSNR/SSIMは原設定のclipを使う。raw範囲外画素率と全128系列予測はローカルに保持し、clipされた図だけで品質を判断しない。Conv/Linear MACはarithmetic_scope.pyの明示した部分計数であり、全FLOPsではない。
