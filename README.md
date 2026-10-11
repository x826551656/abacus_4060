# Abacus Embeddings：RTX 4060 单卡一周复现工程

这是《Transformers Can Do Arithmetic with the Right Embeddings》的一套**资源缩减、机制导向的加法复现**。默认假设 RTX 4060 8GB；16GB 的 4060 Ti 可以使用同一设置。源码、数据生成、训练、恢复、真实自回归评估、对照矩阵和结果绘图均包含在此工程。

**不是原论文训练配置的逐项复刻，不保证达到 99% 的 100 位加法准确率。** 这是交给另一台电脑使用的便携项目包，不包含安装好的第三方环境。解压后先读 [`USER_GUIDE.md`](USER_GUIDE.md)，按指令在接收者电脑安装依赖。目录分类与每个文件用途见 [`DIRECTORY_GUIDE.md`](DIRECTORY_GUIDE.md)，第三方环境见 [`ENVIRONMENT.md`](ENVIRONMENT.md)。本次交付核对以 `LOCAL_VALIDATION.md` 为准，`TEST_REPORT.md`保留此前CPU历史记录；本次未运行CUDA训练和GPU测试。

## 1. 一周内要回答什么问题

主问题：在数字内容、训练样本、优化器、有效深度和训练更新数保持一致时，给数字提供“数内位置”能否改善加法学习和长度外推？

三个递进问题：

1. NoPE/FIRE → Abacus：显式数位对齐是否有效？
2. 固定起点 Abacus → 随机起点 Abacus：长数字受益是否来自训练时见过更广的 embedding 索引？
3. Abacus → 输入注入 → 循环层：在数位表示改善后，结构改动是否仍有额外价值？

用一周单卡预算先把这条证据链跑完整。乘法、排序和自然语言混合任务是下一阶段，不在当前加法代码中伪装成“已复现”。它们需要不同数据、训练预算及指标，见 `PROTOCOL.md`。

## 2. 推荐规模与理由

| 项目 | 本工程默认值 | 理由 |
| --- | --- | --- |
| 显卡 | RTX 4060 8GB，单卡 | 不假设 16GB 显存或多卡 |
| 训练数字长度 | 两个操作数各 1–20 位 | 保留论文核心长度外推场景 |
| 训练样本池 | 200,000 条，有放回生成 | 20×20 个长度组合每格恰好 500 条 |
| 隐藏维度 / FFN / 头数 | 256 / 512 / 8 | 先降低模型开销，保留完整 Transformer 机制 |
| 标准模型 / 循环模型 | 8×1 / 4×2 | 有效深度都是 8，循环模型共享参数 |
| 微批 / 累积 / 有效批 | 32 / 4 / 128 | 梯度按答案 token 数加权；并非简单平均微批 loss |
| 起点范围 | β∈{1,…,100} | 同一次前向中所有数字共享 β；测试 β=1 |
| 建议更新上限 | 每个模型 24,000 步，按本机测速缩减 | 曝光 3,072,000 次，约 15.36 个小数据池遍历 |
| 训练精度 | auto | CUDA 支持 BF16 则用 BF16，否则 FP16 + GradScaler |
| 最大上下文 | 512 | 足够覆盖本工程 100 位加法输入与输出 |
| 验证 / 保存 | 每 2,000 / 1,000 步 | 验证集只含训练长度，不窥视 OOD 测试集 |

数据池行宽为 65 个 uint8，`train.npy` 大约 13MB（约 12.4MiB），不是 200,000 个 Python 大对象。训练只输入最多 64 个 token。100+100 位加法的输入前缀含 BOS 后长 203，最多生成 102 个 token，最后一次前向上下文长 304。小池重复曝光较多，需观察ID验证过拟合；预算允许可在新目录改为800,000或2,000,000条，并给全部对照使用同一新数据协议。

标准 Abacus 模型参数实数为 **4,344,064**；NoPE 为 **4,212,992**，FIRE 为 **4,215,632**，4×2 Abacus 循环模型为 **2,239,744**。这里比较的是匹配宽度、有效深度及更新数，不是严格参数量匹配，也不是完全相同 FLOPs。

24,000 步是建议上限，不是测速结果。4060 笔记本功耗、台式机散热、操作系统、CUDA 内核和可用 GPU 小时都影响耗时。`budget.py` 必须使用你本机的实测文件。

## 3. Windows 安装：不用修改执行策略

支持范围按本工程依赖选择Python3.11–3.12；接收者推荐安装Python3.12。使用Windows PowerShell或PyCharm Terminal。打开整个`abacus_4060`文件夹。以下命令全部在工程根目录执行。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
nvidia-smi
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
.\.venv\Scripts\python.exe -m pytest -q
```

后文 Windows 示例用 `$py` 指向虚拟环境解释器，避免调用到另一个 Python；不必激活虚拟环境，也不必降低 PowerShell 安全设置。

```powershell
$py = '.\.venv\Scripts\python.exe'
```

PyTorch 2.5.1 + CUDA 12.1 安装方式来自官方历史版本页。此工程不编译 CUDA 扩展，一般不必另装完整 CUDA Toolkit，但需要与该运行时兼容的 NVIDIA 驱动。若 `torch.cuda.is_available()` 为 False，先修复驱动/安装环境，不要把一周实验偷偷改成 CPU 训练。固定旧版本是为了减少环境漂移，不代表它是最新版本。

Linux/WSL2 等价安装：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pytest -q
```

本工程不需要torchvision、torchaudio、transformers、datasets或FlashAttention第三方编译包。接收者应按`USER_GUIDE.md`实测自己的CUDA环境；本次只做交付静态核对，见`LOCAL_VALIDATION.md`。

## 4. 第一步：生成并审计数据

```powershell
& $py run.py prepare --output data/add20 --samples 200000 --train-max 20 --per-cell 64
& $py run.py audit --data data/add20
```

默认训练数据种子为 2026，模型种子独立。训练集对每一对长度等量采样；不能只生成“20 位+20 位”然后声称覆盖 1–20 位。

数字、加号、等号是单字符 token。两个输入和答案都倒序：

```text
普通十进制：123 + 45 = 168
训练序列：BOS 3 2 1 + 5 4 = 8 6 1 EOS
推理前缀：BOS 3 2 1 + 5 4 =
```

`PAD` 仅用于数组存储和批内右侧补齐，是特殊 token，不是十进制 `0`。不对操作数或结果补十进制零，不在人为对齐位置插入空格。有效 token 的因果注意力看不到右侧 PAD；PAD 位置损失被屏蔽。

为了防止 train/test 泄漏，对 `(min(a,b),max(a,b))` 的固定哈希分配 train/val/test，比例近似 80/10/10。因此 `a+b` 与 `b+a` 不会跨 split；短数字训练重复是有放回采样的自然结果，并非泄漏。测试集不重复相同有序题目；极小单元格可用题目不足 64 时保留真实 n，不重复凑数。

默认验证长度：1、2、4、8、12、16、20，每格至多 8 条，只用于选 checkpoint；固定seed实际386条。默认测试长度：这些长度加 24、32、40、60、80、100，每格至多 64 条，共 169 个长度格，固定seed实际10,761条。这是**稀疏长度网格**，不是论文 1–100 的全网格。1×1格验证仅有2条、测试9条，故必须保留每格真实n；验证集只是低成本选模工具，不是最终充分的ID性能估计。

`audit` 检查全部训练行的算术、倒序、split 和每格计数，并核对文件 SHA256。生成程序拒绝覆盖已有数据目录；修改数据方案时创建新目录。准备和审计可能花几分钟，这是 CPU 工作，不需要显卡。

## 5. 第二步：实测 4060，再决定统一预算

运行每个核心方法的测速；测速包含真实梯度累积、反向传播与 optimizer 更新。默认预热 5 次、计时 30 次。使用真实训练数据，比合成满长度测速更有代表性。

```powershell
$methods = @('nope','fire','abacus_fixed','abacus','abacus_ii','abacus_loop')
foreach ($m in $methods) {
    & $py run.py benchmark --preset $m --data data/add20 --updates 30 --output "bench/$m.json"
    if ($LASTEXITCODE -ne 0) { throw "Benchmark failed: $m" }
}
& $py budget.py --benchmarks bench/nope.json bench/fire.json bench/abacus_fixed.json bench/abacus.json bench/abacus_ii.json bench/abacus_loop.json --gpu-hours 100 --reserve-hours 20 --output budget.json
```

这里 100 小时指**一周可支配 GPU 小时**，不是“7 天”自动等于“168 小时”，也不是累计人工工作时长。20 小时预留给评估、重启、异常和部分验证；实际验证/长序列评估可能更慢，必须再检查一次 100 位推理成本。预算脚本对训练计时再乘 1.25 安全余量，并将步数向下取整到 1,000。

若你一周只能运行 28 GPU 小时，应把 `--gpu-hours` 改为 28，并适当调整预留。若八组预算少于 4,000 步，优先保留 NoPE/Abacus 两模型×两种子，不要用几十步的结果评价论文。

若测速 OOM，统一改为微批 16、累积 8：所有方法仍为有效批 128，重新测速并重新冻结计划。不能只有一个模型减少样本曝光或训练步数。评估 OOM 时可独立降低评估批大小，不改变模型训练预算。

## 6. 第三步：冻结并运行对照矩阵

从 `budget.json` 读取 `matched_steps_per_run` 后设置共同更新数。示例假设测得足够跑 24,000；如果算出来是 12,000，就把所有命令中的值改为 12,000。

```powershell
$steps = 24000
& $py suite.py --data data/add20 --output runs/week1 --steps $steps
```

这一步只打印 dry-run，不训练。确认后执行：

```powershell
& $py suite.py --data data/add20 --output runs/week1 --steps $steps --execute
```

八组实验：种子 42 跑 NoPE、FIRE、固定起点 Abacus、随机起点 Abacus、Abacus+输入注入、Abacus+输入注入+4×2 循环；种子 43 再跑 NoPE 和随机起点 Abacus。主结论以 NoPE↔Abacus 两种子对照为主，其他单种子只视为探索性消融。

若想分天执行，不改变冻结计划：

```powershell
& $py suite.py --data data/add20 --output runs/week1 --steps $steps --only nope_seed42 abacus_seed42 nope_seed43 abacus_seed43 --execute
& $py suite.py --data data/add20 --output runs/week1 --steps $steps --only fire_seed42 abacus_fixed_seed42 --execute --continue
& $py suite.py --data data/add20 --output runs/week1 --steps $steps --only abacus_ii_seed42 abacus_loop_seed42 --execute --continue
```

第一次执行就写入完整八组 `plan.json`；后续子集只能使用相同配置。`--continue` 跳过已经完成的 run，并从未完成 run 的 `last.pt` 恢复。未加 `--continue` 时遇到已有 run 会拒绝覆盖。

### 单模型手动运行与夜间续训

```powershell
& $py run.py train --preset abacus --seed 42 --steps $steps --data data/add20 --run runs/manual_abacus42 --max-hours 10
& $py run.py train --preset abacus --seed 42 --steps $steps --data data/add20 --run runs/manual_abacus42 --resume runs/manual_abacus42/last.pt --max-hours 10
```

`--max-hours` 只限制当前会话，不改变总计划或学习率日程。停止后 `status.json` 会标记 incomplete；不要把会话停止当成训练已完成。正常时间预算结束在完整更新步保存 checkpoint。Ctrl+C 仅可恢复最近完整保存点，未保存步可能需重跑。

恢复保存了模型、AdamW、GradScaler、方法随机数、Torch RNG 和数据顺序。要求同一 run、同一配置、同一数据，避免“恢复”时偷偷改变预算。只加载自己信任的 `.pt` 文件；PyTorch pickle 不适合接收不可信外部模型。

## 7. 模型具体怎样计算：不是硬编码竖式

普通模型和 Abacus 模型都用数字倒序，不应把“倒序”归功于本文新方法。训练时是 teacher forcing：模型在每个输出位置预测下一个数字，损失只计算答案数字和 EOS；推理时只喂问题，逐 token 贪心生成。

例如 `321+54=`，模型第一次输出 `8`，然后读取自己生成的 `8` 再输出 `6`，再输出 `1`，最后 EOS。它并没有被源码告知“3 和 5 必须相加”，注意力、FFN 和表示都是从训练数据学习的。不能把教学动画里的竖式步骤误认为已被机制分析证明的内部算法。

Abacus 的唯一额外位置信息示例：

```text
token：      3 2 1 + 5 4 = 8 6 1
数内索引：   1 2 3 0 1 2 0 1 2 3
若 β=37：   37 38 39 0 37 38 0 37 38 39
```

同数位数字拥有同一个额外向量，token 自身的内容向量仍不同。非数字使用可学习的第 0 行，不是完全不加向量；这一点与作者 standalone 实现相同。训练对所有数字 span 共享随机 β，测试固定 β=1。答案的数内位置在生成时根据**模型自己产生的 token**更新，不读取真实答案。

这并不意味着可以泛化到无限长。训练最大输出 21 位，β 最大 100，因而正索引最多约 120；100 位测试所用的大部分索引已在短序列随机位移中受到训练。固定 β=1 的消融只训练前面约 21 个正索引，长位置表行没有相同训练机会。它可以区分“长度外推”与“未知索引外推”。

输入注入是每层再次加原始 embedding，不是再次读 Python 算术结果；4×2 循环是 4 个不同层重复调用 2 遍，不是自回归生成两个 token，也不是推理时无限增加层数。本工程默认循环实验**不加 progressive loss**，这样可以单独考察权重共享。论文式训练变体及偏差见 `DEVIATIONS.md`。

## 8. 第四步：真正的自回归评估

正式 test 只在计划完成、best checkpoint 根据 ID 验证选好后评估，不能拿 OOD test 调学习率和挑 seed。

```powershell
& $py run.py eval --checkpoint runs/week1/abacus_seed42/best.pt --examples data/add20/test.jsonl --output evals/abacus42 --batch 16
& $py run.py eval --checkpoint runs/week1/nope_seed42/best.pt --examples data/add20/test.jsonl --output evals/nope42 --batch 16
& $py run.py eval --checkpoint runs/week1/abacus_seed43/best.pt --examples data/add20/test.jsonl --output evals/abacus43 --batch 16
& $py run.py eval --checkpoint runs/week1/nope_seed43/best.pt --examples data/add20/test.jsonl --output evals/nope43 --batch 16
& $py report.py --evals evals/nope42 evals/abacus42 evals/nope43 evals/abacus43 --output reports/main
& $py analyze_errors.py --eval evals/abacus42 --output reports/abacus42_errors.json
```

同样对 FIRE、fixed、II、loop 执行 eval，然后将 eval 目录加入 report 参数。结果保留每个长度格的正确数、样本数、EOS 率、Wilson 区间，以及逐题预测与错误。只给前缀不给答案，禁止用 teacher-forced argmax 准确率代替算法外推效果。

ID：两个长度均≤20。OOD：至少一个长度>20、两个都≤100。主指标是长度格等权的 macro exact match，同时给 sample-weighted micro；整个答案和 EOS 都正确才计一题正确。EOS 口径可能比论文摘要的“所有数字正确”更严格，不能混为一谈。

每格 64 条属于课程项目的探索性评估，不能支持十分精细的百分点结论。两个 seed 只能初步检验稳定性；FIRE 等单 seed 更不能声称统计显著。短格里可能同时出现交换次序的两题，置信区间未建模这种相关性；micro Wilson 是描述性近似，不代表独立的多次训练不确定性。

热力图只画被测试的 13×13 位置，轴不等间距，并且不插值填充未测长度。图中的红框表示 ID。图的形状不应被包装成论文原图 1 的完全复刻。`report.py` 检查数据 hash、更新预算、宽度和有效深度，还拒绝把同一 seed 的两个 checkpoint 当独立实验。

验证最佳点可能早于最后训练步，这不代表 run 未完成。评估 provenance 同时记录 checkpoint 选择步数和 run 完成步数。未完成 run 的结果不能混入正式比较。

单题演示：

```powershell
& $py run.py demo --checkpoint runs/week1/abacus_seed42/best.pt --a 123 --b 45
```

训练标签和评估真值用 Python 大整数产生，这是离线数据/计分工作。**网络生成过程不调用计算器、Python 算术、搜索或代码解释器**；`demo` 是先生成再显示真值。Python 得到的真值不是模型预测。

## 9. 一周操作表

| 时间 | 工作 | 验收点 |
| --- | --- | --- |
| 第 1 天 | 环境、pytest、生成/全量审计、六方法 GPU 测速、预算冻结；小试验仅看训练/ID 验证 | CUDA 可用、无 OOM、审计通过、参数和预算一致 |
| 第 2–4 天 | NoPE/Abacus × seeds 42/43，四个核心 run；保存日志与恢复点 | completed=true，loss 有限，ID 验证趋势可解释 |
| 第 5 天 | FIRE 与固定 β 消融，种子 42 | 同数据、同有效批、同更新数 |
| 第 6 天 | 输入注入与循环消融，种子 42 | 有效深度都是 8，循环参数更少 |
| 第 7 天 | 完成后统一测试、绘图、错误分类、整理失败案例与差异说明 | 完整 test provenance 与不夸大的结论 |

这是工作组织方案，不是 GPU 速度保证。如果预算只够四组，删掉探索性四组并明确报告删减，不能把主模型训练到 24k、基线只训练 4k 后比较“优势”。训练结束早也不要将剩余时间全部用来反复窥视同一个 OOD test 调参。

## 10. 不收敛或 OOD 很低时怎么办

先依次检查：环境与 NaN → loss mask → token/倒序 → train/val 隔离 → ID 是否已经学会 → beta 范围 → checkpoint/训练是否完成 → 自回归与 EOS。`tests/test_core.py` 与 `audit` 专门覆盖这些容易出现伪结果的环节。

如果 ID 很低：不能直接得出“Abacus 没用”；可能缩减模型或更新数不足。用新的仅训练/ID 验证 pilot 比较 LR=1e-4 与 3e-4 或更长预算，再建立**新实验协议**。修改学习率需在 `abacus/config.py` 中设置，并给旧目录留档。最终公平比较须重跑相关基线，不能只强化 Abacus。

如果 ID 高、OOD 低：这是合法研究结果。按长度和进位链检查错例；报告模型规模、步数和与原文的差异，不抄原论文 99.1%。延长训练、增加 k、组合 FIRE 或增加共享计算深度可作为后续假设，不作为已经被本次实验确认的事实。

## 11. 工程入口

| 文件 | 用途 |
| --- | --- |
| `run.py` | prepare / audit / benchmark / train / eval / demo |
| `USER_GUIDE.md` | 从零安装、PyCharm社区版配置、每一步运行与故障处理 |
| `DIRECTORY_GUIDE.md` | 数据/代码/第三方/结果目录及文件职责 |
| `ENVIRONMENT.md` | Python解释器和第三方依赖版本、原仓库环境区别 |
| `check_environment.py` | 当前解释器、CUDA、GPU与第三方库检查 |
| `audit_project.py` | 官方commit、逐文件用途/哈希、相对路径检查 |
| `suite.py` | 冻结八组对照、分批运行与恢复 |
| `budget.py` | 六个实测 benchmark 推算共同训练步数 |
| `report.py` | 真实指标 CSV、稀疏热力图、seed 汇总 |
| `analyze_errors.py` | 缺 EOS、非法 token、长度错误、数字错误和进位链分组 |
| `abacus/data.py` | 字符 tokenizer、隔离采样、mask、确定性数据顺序 |
| `abacus/model.py` | Abacus / APE / FIRE / NoPE、输入注入和共享循环模型 |
| `abacus/training.py` | AMP、梯度累积、验证、checkpoint、测速 |
| `abacus/evaluation.py` | 带 KV cache 的真实贪心生成与指标 |
| `tests/test_core.py` | 编码、因果性、缓存、过拟合、梯度与恢复测试 |
| `DEVIATIONS.md` | 原论文与本工程的严谨差异表 |
| `PROTOCOL.md` | 预注册实验问题、资源决策、验收与扩展方案 |
| `TEST_REPORT.md` | 已实际执行的测试与未验证边界 |
| `SOURCES.md` | 论文、官方代码固定 commit 与安装来源 |

最终提交建议包含：`plan.json`、数据 `manifest.json/audit.json`、GPU benchmark、每个 run 的 `run.json/status.json/train.jsonl`、测试 `provenance.json/summary.json/grid.csv`、热力图与错例分析。训练数据可以按固定种子重建；checkpoint 不要删除唯一副本，课程报告应保留生成结果对应的文件 hash。
