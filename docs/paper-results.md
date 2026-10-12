# Paper results and figure sources

The README showcases reported results from **A Few More Tokens Ahead: Reinforcing Parallelism in Diffusion Language Models**, using the 23-page paper draft built on October 12, 2026. The figures below come from that paper's source PDFs. The displayed results are historical paper experiments, distinct from the code checks documented in [release validation](validation.md).

## Method and experiment setup

RePara samples short consecutive token blocks from a shared masked state, then reconstructs each recorded state to optimize the block's joint probability. Fix2 commits blocks of up to two tokens; Adaptive2-0.95 accepts the longest confident prefix of a two-token proposal using normalized-entropy confidence at threshold 0.95. SDAR also caps proposals at its native attention boundaries.

The mathematical experiments train each backbone separately on GSM8K and MATH. JustGRPO, Fix2, and Adaptive2-0.95 use 128 full-parameter optimizer updates, a constant learning rate of `5e-6`, 64 prompts per update, and 16 completions per prompt. Rollouts use temperature one and a fixed 256-token response budget. Evaluation uses step-128 checkpoints and one temperature-zero response per problem at response budgets 128, 256, and 512, excluding the prompt.

The plots additionally show base models and released baseline checkpoints re-evaluated under those response budgets.

LLaDA evaluation uses confidence-based decoding in 32-token windows with fixed quotas `q ∈ {1, 2, 4, 8, 16}` and `S = L/q` total denoising steps. SDAR evaluation uses native four-token blocks and dynamic low-confidence selection at token-probability threshold 0.85. Its nominal quota is `q = 4/D`, where `D ∈ {4, 2, 1}` is the denoising budget per block. Threshold selection can commit more than the nominal quota. The training block cap and inference quota are separate parameters.

## Selected LLaDA results

Accuracy (%) at generation budget 512:

| Nominal quota | Training | GSM8K | MATH-500 |
| ---: | --- | ---: | ---: |
| 8 | JustGRPO | 75.06 | 30.00 |
| 8 | RePara Fix2 | 81.27 | 37.40 |
| 8 | RePara Adaptive2-0.95 | 81.35 | 35.40 |
| 16 | JustGRPO | 28.20 | 12.80 |
| 16 | RePara Fix2 | 68.54 | 26.80 |

The quota-eight Adaptive values are stated in the paper's main text. The LLaDA main-results table provides the full generation-budget-512 sweep for Fix2, JustGRPO, and released baselines. The figures also show Adaptive and all three response budgets.

## SDAR results at generation budget 512

Accuracy (%) under the native four-token dynamic confidence protocol:

| Training | GSM8K q=1 | GSM8K q=2 | GSM8K q=4 | MATH-500 q=1 | MATH-500 q=2 | MATH-500 q=4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SDAR-Chat | 90.75 | 55.12 | 74.45 | 50.00 | 36.20 | 28.20 |
| JustGRPO | 82.11 | 55.12 | 46.63 | 54.20 | 34.20 | 35.40 |
| RePara Fix2 | 85.37 | 72.63 | 81.80 | 52.40 | 39.80 | 40.40 |
| RePara Adaptive2-0.95 | 74.37 | 58.00 | 59.82 | 51.00 | 32.00 | 41.00 |

The paper's SDAR table also reports budgets 128 and 256. The plotted Fix2 GSM8K curve is non-monotonic: its quota-four score exceeds its quota-two score. The result supports better accuracy retention at the evaluated reduced budgets, rather than a uniformly flat curve or an improvement at every setting.

## Training efficiency

The paper compares LLaDA mathematical training on 64 NVIDIA H200 GPUs, using 1,024 completions of 256 tokens per optimizer update. After resolving continuation records, the measurements average 127 optimizer steps shared by all three methods, including their initial timing steps:

| Training | Core time (s/update) | Full step time (s/update) | Core speedup |
| --- | ---: | ---: | ---: |
| JustGRPO | 422.64 | 427.89 | 1.00× |
| RePara Fix2 | 213.39 | 218.60 | 1.98× |
| RePara Adaptive2-0.95 | 372.26 | 377.66 | 1.14× |

Core time includes rollout, rewards, learner computation, and optimizer updates. Full step time also includes checkpointing and metric preparation; Fix2's full-step speedup is 1.96×. These measurements describe cost per optimizer update in that setup, not inference latency or time to a matched accuracy. Adaptive's variable block counts introduce synchronization overhead, so average accepted block length is not a measured speedup.

## Assets and table snapshot

| Repository asset | Paper source | Caption |
| --- | --- | --- |
| [Method overview](../assets/method-overview.png) | `figures/main/0311-method_overview.pdf` | One-token JustGRPO rollout/replay versus consecutive-block RePara rollout/replay. |
| [LLaDA MATH-500](../assets/llada-math500.png) | `figures/main/0411-results_llada_math.pdf` | Accuracy versus nominal decoding rate for response budgets 128, 256, and 512. |
| [LLaDA GSM8K](../assets/llada-gsm8k.png) | `figures/main/0412-results_llada_gsm8k.pdf` | The same panel layout and legend as LLaDA MATH-500. |
| [SDAR MATH-500](../assets/sdar-math500.png) | `figures/main/0414-results_sdar_math.pdf` | Accuracy versus nominal quota under native four-token dynamic decoding. |
| [SDAR GSM8K](../assets/sdar-gsm8k.png) | `figures/main/0415-results_sdar_gsm8k.pdf` | The same panel layout and legend as SDAR MATH-500. |
| [Reported accuracy values](../assets/paper-results.csv) | `sections/050Experiments.tex` | 152 accuracy cells transcribed from the LLaDA generation-budget-512 table and the SDAR three-budget table. |

The PNGs are rendered with `pdftoppm`; their plots, legends, and labels are preserved. The CSV records backbone, training setting, generation length, nominal quota, benchmark, reported accuracy, and evaluation protocol. It is a snapshot of reported table values, not raw benchmark predictions or a new evaluation run. The CSV's LLaDA table includes the paper's Fix2 setting; Adaptive is shown in the source figures and selected main-text values above.

The draft PDF's SHA-256 is `38d46ea7214c7c159c190dbf6a5fcde08ef7108b72a6b0b3ca922c934ba275a9`. The original paper and its source figure/table construction pipelines are not bundled in this repository. Model checkpoint download locations and a public paper URL can be added when available.

The paper does not establish consistent accuracy–efficiency benefits on code generation. The repository implements code training and evaluation while leaving the detailed coding diagnostic experiments outside the release scope.
