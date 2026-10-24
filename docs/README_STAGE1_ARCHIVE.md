# Faithful scaled LHFM-V: stage1 learning experiment

40更新の機能確認とは別に、原構造・数値operator・損失・評価を固定した1seed/3000update×2modelの学習段階です。以前の出力は不変に保持。6時間の3seed案は自動実行しません。原benchmarkスコア/一般優位性を再現したという意味ではありません。

## 固定条件と開始前の記録

`PLAN.md`、`configs/stage1.json`、`core_identity.json`、run開始時`source_hashes.json`を参照。core4ファイルは前の検証済み自作実装とbyte一致。64×64/28×28 MNIST2digits/10+10、history encoder/differences/coords、attention memory、future planner、half-frame feedback、source/transport/aux heads、二次upwind/zero外部/unbounded transport/24次適応shifted exponential/source同時積分、全4項lossを保持。変更は学習予算とsingle-seedの固定stage設計です。

seed1017、scratchから初期化し40update checkpointを読みません。両modelは同一初期weights・同一3000訓練系列。batch1、縮小幅65,893params。baselineはw=0の解析的source-only極限で、未使用transport paramを含むため実効capacity完全一致ではありません。

元AdamW/weight decay/epsilon/beta2/clip、original schedule/beta1軌道へのindex写像、EMA.999は保持。3000更新後の初期EMA寄与は約4.97%。raw modelとEMAを分けてvalidationを測ります。

validationはoriginal heldout pool/seed271109の先頭32系列。元検証milestoneをceil(step×3000/1250000)で再尺度化した35時点を固定し、raw pixel MSE最小EMAでcheckpoint選択。既視test0–15は使わず、未使用official16–143(128系列)を両modelの選択完了後にだけ開きます。testで条件選択しません。single seedはrobust superiorityを示しません。

## 実行

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --index-url https://pypi.org/simple -r requirements-lock.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python download_data.py --mnist
.venv/bin/python download_data.py --official-test
.venv/bin/python stage1.py --official-test data/mnist_test_seq.npy
.venv/bin/python summarize.py
```

出力先は新規のみ、上書き・自動再開なし。`--config`/`--output`で別の凍結設定・出力先を指定できます。公開候補にはrawdownloadを含めず、SHA256と明示download commandを提供。公式source codeはimport/runせず、標準ライブラリ/PyPI依存と自作のみです。

MPSが利用できない場合は未計測CPUで2時間枠を開始せず停止。OS設定・管理者権限・課金・公開pushは不要。初期global swapから256MiB増加、RSS1.5GiB、MPS driver1GiB、全段階7200秒で停止。元operator128internal-piece guard/非有限値も保持。勝敗による中断やstep削減をしません。

`runs/stage1/progress.jsonl`が開始時刻からの実測・update・validation・資源の生記録、`current_status.json`は直近完了update。`runtime.json`にPID/UTC開始/scratch初期化を記録。checkpointとoptimizer rolling stateは保存しますが、自動再開はありません。

## 結果の定義

MSE/MAEはraw spatial sum（pixel平均×4096）。SSIMはpaperのrange2/FP32 uniform7/sample covariance/transpose/crop3/FP64 spatial mean→FP32 channel mean、PSNRはclip後frame別計算。SciPy1.16.3は旧公式wheelの現Mac互換問題を受けた明示差（NumPy1.26.4は保持）。元コード同一重みforward equivalenceは未検証。

latencyはbatch1、3warmup後10反復、MPS synchronizeを含むmodel forward、input transfer除外。history処理/20field/計算するが予測に使わないaux headを含む本自作のscope。128系列のend-to-end推論時間（転送含む）も別記録。元FLOPsや他hardware速度と比較しません。

## 配布と再集計

lean ZIPはsource/config/hash/raw metric CSV/JSON/小checkpoint/固定example/図のみを含み、raw MNIST/official819MB/test全体npz/大きな予測配列/rolling optimizer state/venv/cacheを除外します。Library保存成功だけでクライアント配布可能と判断せず、目標5MB未満。全raw predictionsはローカル保存し、必要なら32系列ごと8MB未満のNPZ shardsを作成します。

`summary.py`相当の結果再集計・図生成は`summarize.py`。`build_release.py`がallowlist/hash監査、`audit_release.py`が独立展開/テスト/全中核CPU fixture smoke/checkpoint再生成/結果照合を行います。既存専用venvは再利用、新OS/新依存導入の完全検証とは区別。

今回の学習の完了・停止状態は`RESULTS.md`と`runs/stage1/completed.json`または`stopped.json`で確認します。元実装との忠実性の継承範囲は`FIDELITY_INHERITED.md`、追加stageの凍結設計は`PLAN.md`。公開push/ライセンス付与は行いません。
