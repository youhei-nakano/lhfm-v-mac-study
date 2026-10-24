# 忠実な縮小LHFM-V：3000更新×2の結果

両者3000更新・全validation・予約公式test128系列・同CPU latency評価を完了。原構造と演算を保つ小幅実装の検証であり、原benchmark性能の再現ではない。単seedの記述的比較で、統計的有意・一般的優位性は主張しない。

| model | raw空間和MSE ↓ | raw空間和MAE ↓ | SSIM range2 ↑ | clip PSNR ↑ | CPU latency ms ↓ |
|---|---:|---:|---:|---:|---:|
| transport | 130.300531 | 309.159378 | 0.496393 | 15.143122 | 323.934 |
| transport_off | 146.544421 | 338.656171 | 0.440113 | 14.757244 | 34.782 |
| last_frame | 280.055249 | 331.154181 | 0.665462 | 11.836533 | 未測定 |

transportはoffよりMSEが11.08%低い一方、推論は9.31倍遅い。固定例16〜18の実視ではtransportに強いぼけ・数字形状の喪失、offに残像・格子状ノイズ・移動予測失敗がある。last-frameのSSIMが両者より高い。MSE低下を良質な動画予測の成功と解釈しない。

## 条件・評価

seed1017、batch1、両者3000更新、64px、2数字、10観測+10未来。同じtrain系列ID/順序、初期化規則、optimizer・全loss・原schedule式の再尺度化・EMA.999。固定val32系列と35検証点で最小EMA raw pixel MSEを選択。両者3000更新が選択、初期EMA寄与4.9712%。train objectiveと独立validation pixel MSEを混同しない。

testは公式10000系列の16〜143。両者選択後に初めて開いた。全128raw予測から再集計しJSONと1e-10以内で一致。128系列は128独立学習ではない。paired_sequence_mse.csvを保存するが学習seed間のばらつきや有意差は未推定。

MSE/MAEはclip前raw pixel誤差×4096の空間和。SSIMはclip後range2/FP32/uniform7/sample covariance49/48/transpose/crop3/FP64空間平均→FP32channel平均、PSNRはclip後frame単位。画像は表示用clip。raw範囲外率transport34.92%、off33.56%。強さはrecovery_visual_review.jsonのmin/max参照。

## 復旧・時間・資源

CPU復旧から全処理完了5099.87秒。transportのCPU継続788..3000（検証込み）4291.62秒、off全3000更新（検証込み）741.97秒。transport採用prefix込み計測区間合計7439.29秒（元576到達elapsed+MPS復旧elapsed+CPU継続）。失敗重複更新/probeを二重加算せず、元ログは別保持。

元MPS650更新で全体swap保護停止→完全状態576からMPS復旧→787で再度swap停止→787完全状態からCPU継続。transport1..787はMPS、以降CPU、offは全CPU。この運用差・丸め差のため完全同一device対照ではない。CPU save/resume fixtureのmodel/EMA/loss完全一致、元と重なる74 MPS更新のloss/gradient norm一致。全重みの跨device bitwise一致は未検証。最終軌跡は各更新番号一回、probeや失敗重複は独立証拠でない。

時間制限のみ解除。RSS高水位1.5GiB、driver1GiB、全体swap増加256MiBは維持。CPUプロセスpeak1562771456 bytes（約1.46GiB）は両モデル等の高水位で、off単独値ではない。1119更新時の現在resident1308229632 bytesと区別。最終swap6686706237.44はCPU開始7114525245.44より少ない。swapは全体値で単独OOMと断定しない。毎更新完全状態を原子的に保存。

## latency・計算量・容量

同CPU2threads、batch1、10観測→10予測、3warmup+10同期wall-clock forwardの中央値。入力転送・データ読込・lossを除外、evalでも計算する未使用aux headを含む。生10値をJSON保持。

Conv2d/Linearのみ両者50534400 MAC、1MAC=2FLOPsで101068800。attention tensor積/norm/activation/interpolation/物理operator等を除く部分計数で、総FLOPsや原13.1G再現値でない。固定test16のtransportは20half-step・計101内部区間・2424stencil評価、offは0。この追加計算をMAC同数で隠さず、FLOPsと速度を同一視しない。

総65893params。offはtransport MLP816+correction544+history offset800=2160scalarが予測/lossへ影響せず、有効63733。重み変更でもoff出力が不変のテストを通した。完全等容量/等計算量や独立SOTAとは呼ばず、学習時transport-off ablationと呼ぶ。

## 検証・再実行・限界

14テスト成功。二次zero-exterior stencil、source込み24次shifted exponentialと独立dense matrix_exp/gradient、zero限界、適応分割、原幅count/cadence、schedule/metric、resume、inactive容量と部分MACを検証。公式ソースは固定commit読取参照のみ、公式全モデル同重み数値一致は未検証。FIDELITY_STAGE1.mdに保持/縮小/未検証を整理。

README_RECOVERY.md、固定依存版、公式download URL/SHA、CSV/JSON全記録、選択EMAと小さな完全状態576/787を含む軽量ZIP。公式819MB data・venv・巨大配列は除外しローカル保持。再集計・クリーン展開テスト/smoke/再生成・secret/path監査はaudit JSONを参照。

可視品質は不十分。小幅・小batch・短い学習・単seed・小test subset・混在deviceが制約。原benchmarkスコアと複数seed頑健性は未達。testを見た追加学習/選択はせず、別の3seed長時間計画を自動開始しない。
