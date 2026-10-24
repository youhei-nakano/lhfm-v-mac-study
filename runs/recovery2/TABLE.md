# Single-seed stage1 heldout comparison

One initialization seed; no significance or robust superiority claim. Same128new official sequences after validation-only selection.

| model | selected update | MSE sum raw | MAE sum raw | SSIM range2 | PSNR | latency ms |
|---|---:|---:|---:|---:|---:|---:|
| transport | 3000 | 130.300531 | 309.159378 | 0.496393 | 15.143122 | 323.934 |
| transport_off | 3000 | 146.544421 | 338.656171 | 0.440113 | 14.757244 | 34.782 |

Transport minus off MSE: -16.243889
