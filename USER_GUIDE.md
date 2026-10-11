# 从零操作说明：Windows + PyCharm 社区版 + RTX 4060

适用项目：`abacus_4060`。这是给接收者使用的项目包，发送者无需本地安装第三方库。接收者解压ZIP，将整个文件夹放在电脑任意位置，再用PyCharm打开整个文件夹。以下命令都在**项目根目录**执行，数据、权重、日志和图片全部通过相对路径访问。移动项目后仍使用同样命令。

这是论文核心 Abacus 嵌入的**小规模加法复现**：20 万训练样本、1–20 位训练、最长 100 位测试，比较 NoPE、Abacus 和其他消融。完整论文还研究乘法、排序等，本周不纳入。代码、数据和实际结果的区别见 `DIRECTORY_GUIDE.md`，与原论文的差异见 `DEVIATIONS.md`。

## 1. 准备三个软件

1. **NVIDIA 驱动**：在 NVIDIA 官网安装适合 RTX 4060 的驱动，安装后按提示重启。打开 Windows Terminal/PowerShell，输入 `nvidia-smi`，应该看到显卡名称和显存。该命令显示的 CUDA Version 是驱动支持上限，不是 Python 已安装的 CUDA 版本。
2. **Python 3.12 64 位**：到 [Python 官网](https://www.python.org/downloads/windows/)下载；安装时勾选Python Launcher和Add Python to PATH。已有3.12可以直接使用；请不要为这套固定依赖选择3.13，PyTorch2.5.1的此处Windows wheel使用3.12。
3. **PyCharm 社区版**：已有社区版可以继续使用；也可以从 [JetBrains 官网](https://www.jetbrains.com/pycharm/download/)获取 PyCharm，使用免费核心 Python 功能即可。不同发行版本的按钮名称略有差异。PyCharm 是编辑器，真正执行代码的是下一步的 `.venv` Python 解释器。

Git用于重新获取和复查官方源码，下载地址为 [Git 官网](https://git-scm.com/downloads/win)。ZIP已包含`reference/upstream/`官方完整源码快照，为减小体积不包含`.git`历史。仅训练无需Git；文件审计在ZIP中核对官方源码文件哈希。

本项目使用预编译 CUDA wheel，无需为它额外安装 Visual Studio、完整 CUDA Toolkit 或 cuDNN SDK，也不需要购买专业版 IDE。

## 2. 用 PyCharm 打开项目并创建环境

1. 打开 PyCharm，点击 **Open**，选择整个 `abacus_4060` 文件夹，点击信任本项目。不要只打开 `run.py`。
2. 在 Project 文件树中看到 `run.py`、`abacus/`、`requirements.txt` 后，打开底部 **Terminal**。终端当前目录必须是项目根目录。
3. 输入下面的命令，检查3.12并创建虚拟环境。项目包不包含`.venv`，第一次使用必须创建；若你自己之前已创建且检查通过，可跳到安装/核验步骤。

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 --timeout 120 --retries 5
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

PyTorch CUDA 包下载约 2.45GB，第一次安装需要网络和数GB空闲空间；这属于运行环境大小，训练数据只有约 13MB。以后无需重复下载。安装命令依据 [PyTorch 官方历史版本页](https://pytorch.org/get-started/previous-versions/)。不要在另一个全局 Python 中运行 `pip install`。

4. 打开 **File → Settings → Project → Python Interpreter**（新版本可能显示 **Settings → Python → Interpreter**）。选择 **Add Interpreter → Add Local Interpreter → Existing environment**，选择项目内 `.venv/Scripts/python.exe`，点击 OK/Apply。如果 PyCharm 自动选中了别的全局 Python，请改成这个环境。
5. 若 `.venv` 在项目中已存在，可直接选择它，不要创建第二个环境。PyCharm界面可能显示解析后的完整解释器路径，这是 IDE 定位解释器的方式；源码及命令不写机器专属盘符。

第三方依赖：`torch==2.5.1+cu121`（模型与CUDA）、`numpy==2.3.5`（紧凑数据数组）、`matplotlib==3.10.8`（结果图）、`pytest==8.3.3`（正确性测试）。标准库 `pathlib/json/argparse/csv/random/hashlib/dataclasses` 等无需安装。具体说明见 `ENVIRONMENT.md`。

## 3. 检查解释器和显卡

```powershell
.\.venv\Scripts\python.exe check_environment.py --require-cuda
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe audit_project.py
```

验收：`cuda_available`为`true`、`gpu`包含RTX4060、CUDA矩阵计算有限、测试通过、文件审计`status=pass`。环境检查写入`evidence/windows_environment.json`；文件清单写入`evidence/FILE_INVENTORY.csv`，含每个文件的相对路径、分类、用途、大小和SHA256。发送者此次只核对源码与包结构，没有完成这些依赖/GPU运行检查，接收者需要实际执行。

`pytest` 包括 CPU 正确性测试和六种方法的实际 CUDA 前后向/缓存检查。未检测到GPU时 CUDA测试会标为 skipped；这时不能开始一周正式训练。

## 4. 先跑一个几分钟内完成的小流程

目的：熟悉数据→训练→评估→绘图，不检查论文准确率。下面10步模型准确率很低是正常的。首次执行使用以下目录，重跑请换 `smoke2` 等新名称，程序会拒绝覆盖旧数据和结果。

```powershell
.\.venv\Scripts\python.exe run.py prepare --output data/smoke --samples 400 --train-max 2 --per-cell 5 --test-lengths 1 2 3
.\.venv\Scripts\python.exe run.py audit --data data/smoke
.\.venv\Scripts\python.exe run.py train --tiny --steps 10 --micro-batch 4 --accumulation 2 --data data/smoke --run runs/smoke --device cuda
.\.venv\Scripts\python.exe run.py eval --checkpoint runs/smoke/best.pt --examples data/smoke/test.jsonl --output evals/smoke --batch 4
.\.venv\Scripts\python.exe report.py --evals evals/smoke --output reports/smoke
.\.venv\Scripts\python.exe analyze_errors.py --eval evals/smoke --output reports/smoke/errors.json
.\.venv\Scripts\python.exe run.py demo --checkpoint runs/smoke/best.pt --a 12 --b 7
```

生成的 `reports/smoke/*.png` 可以在 PyCharm 中双击查看；黑色或低值网格应与预测结果对应。`demo` 分开显示网络预测和真实答案，真实答案19不代表网络已算对。

## 5. 准备正式的小数据

```powershell
.\.venv\Scripts\python.exe run.py prepare --output data/add20 --samples 200000 --train-max 20 --per-cell 64
.\.venv\Scripts\python.exe run.py audit --data data/add20
```

默认每个长度组合500条，400个组合共20万条。`train.npy` 约13MB，`val.jsonl` 386题，`test.jsonl` 10,761题，后者测试13×13个稀疏长度格；所有默认生成参数固定，train/val/test以规范化操作数对哈希隔离。输入和答案数字均倒序，例如 `123+45=168` 保存为 `321+54=861`。

| 文件 | 类型与用途 | 是否参与训练 |
| --- | --- | --- |
| `data/add20/train.npy` | uint8数字token矩阵、20万行×65列 | 是 |
| `data/add20/val.jsonl` | 仅1–20位的验证题，选best模型 | 否，用于验证 |
| `data/add20/test.jsonl` | 固定ID与最长100位OOD测试题 | 否，正式训练完成后评测 |
| `data/add20/manifest.json` | 生成参数、划分规则、SHA256 | 元数据 |
| `data/add20/audit.json` | 全量算术、数量和隔离审计 | 证据 |

缩小数据池会增加重复采样及过拟合风险：24,000更新×有效批128=3,072,000次样本曝光，约遍历小池15.36次。若ID验证很差，用新目录生成80万或200万条再重新冻结公平对照；不能只扩大Abacus的数据，不能用OOD测试成绩挑数据规模。

已经生成且审计通过的数据可以直接使用。重新生成会拒绝覆盖，换目录后所有后续命令的 `--data`/`--examples` 也要同步修改。

## 6. 在这台4060上测速，再决定训练时间

笔记本请接电源、避免系统休眠。训练时可以最小化 PyCharm，关闭项目会停止其训练进程。不要同时运行其他占大量显存的程序。

```powershell
$methods = @('nope','fire','abacus_fixed','abacus','abacus_ii','abacus_loop')
foreach ($m in $methods) {
    .\.venv\Scripts\python.exe run.py benchmark --preset $m --data data/add20 --updates 30 --output "bench/$m.json"
    if ($LASTEXITCODE -ne 0) { throw "Benchmark failed: $m" }
}
.\.venv\Scripts\python.exe budget.py --benchmarks bench/nope.json bench/fire.json bench/abacus_fixed.json bench/abacus.json bench/abacus_ii.json bench/abacus_loop.json --gpu-hours 100 --reserve-hours 20 --output budget.json
```

`100`表示一周实际可以使用的GPU小时。如果每天只运行4小时，改为`--gpu-hours 28 --reserve-hours 6`。读取`budget.json`中`matched_steps_per_run`，它才是下面`$steps`的取值，不要无条件照抄24,000。六组测速文件由你第一次运行生成；重新测速使用新的`bench/recheck/`目录，预算输入也对应改变。

预算包含25%训练余量和单独评测预留；长序列测试及验证耗时未完全由训练测速覆盖。不能保证任何笔记本在同一小时数完成八组，更不能保证99%准确率。

OOM：将**全部正式方法**统一为 `--micro-batch 16 --accumulation 8`，重新测速，suite同样传这两项，仍保持有效批128。已有run不能改批配置直接恢复。仅eval OOM可以单独降低 `--batch`。

## 7. 按一周顺序运行实验

1. 第1天：软件环境、测试、smoke、小数据生成审计、测速预算。
2. 第2–4天：NoPE与Abacus，两者各种子42/43，共4个主实验。
3. 第5天：FIRE和固定起点Abacus，两项消融。
4. 第6天：输入注入与循环层，两项消融。
5. 第7天：统一测试、绘图、错例分析、报告。

在PowerShell设置共同步数，例如预算足够时为24,000：

```powershell
$steps = 24000
.\.venv\Scripts\python.exe suite.py --data data/add20 --output runs/week1 --steps $steps
```

该命令只预览，不训练。看到八个run后按天运行：

```powershell
.\.venv\Scripts\python.exe suite.py --data data/add20 --output runs/week1 --steps $steps --only nope_seed42 abacus_seed42 nope_seed43 abacus_seed43 --execute
.\.venv\Scripts\python.exe suite.py --data data/add20 --output runs/week1 --steps $steps --only fire_seed42 abacus_fixed_seed42 --execute --continue
.\.venv\Scripts\python.exe suite.py --data data/add20 --output runs/week1 --steps $steps --only abacus_ii_seed42 abacus_loop_seed42 --execute --continue
```

第一次就冻结完整 `runs/week1/plan.json`；后续不得改变预算/模型配置，所有run顺序执行，不会同时占显卡。预算不足优先只完成4个主实验，并在报告标明省略消融。八组预算脚本不是自动计算四组预算，四组公式见 `PROTOCOL.md`。

## 8. 暂停、恢复和读取日志

训练输出有 `step/loss/grad_norm/elapsed_session_seconds/peak_allocated_gb`，每100更新显示一行，完整记录在 `train.jsonl`。每1000更新保存last，每2000更新验证best；很久没有屏幕输出时先看GPU占用，不要重复启动第二个run。

定时结束一个单独run并恢复（两次配置必须一致）：

```powershell
.\.venv\Scripts\python.exe run.py train --preset abacus --seed 42 --steps $steps --data data/add20 --run runs/manual_abacus42 --max-hours 10
.\.venv\Scripts\python.exe run.py train --preset abacus --seed 42 --steps $steps --data data/add20 --run runs/manual_abacus42 --resume runs/manual_abacus42/last.pt --max-hours 10
```

suite中断后，用**原来的同一条suite命令**加 `--continue`：完成run会跳过，未完成run从last恢复。Ctrl+C可能丢失最近保存后的更新。`status.json` 中 `completed=true` 才表示计划训练结束；best早于last是正常的。

## 9. 评估和报告

主实验完成后运行：

```powershell
.\.venv\Scripts\python.exe run.py eval --checkpoint runs/week1/nope_seed42/best.pt --examples data/add20/test.jsonl --output evals/nope42 --batch 16
.\.venv\Scripts\python.exe run.py eval --checkpoint runs/week1/abacus_seed42/best.pt --examples data/add20/test.jsonl --output evals/abacus42 --batch 16
.\.venv\Scripts\python.exe run.py eval --checkpoint runs/week1/nope_seed43/best.pt --examples data/add20/test.jsonl --output evals/nope43 --batch 16
.\.venv\Scripts\python.exe run.py eval --checkpoint runs/week1/abacus_seed43/best.pt --examples data/add20/test.jsonl --output evals/abacus43 --batch 16
.\.venv\Scripts\python.exe report.py --evals evals/nope42 evals/abacus42 evals/nope43 evals/abacus43 --output reports/main
.\.venv\Scripts\python.exe analyze_errors.py --eval evals/abacus42 --output reports/main/abacus42_errors.json
.\.venv\Scripts\python.exe run.py demo --checkpoint runs/week1/abacus_seed42/best.pt --a 123 --b 45
```

四项消融同理分别运行eval，替换checkpoint及输出目录，把这些目录加入 `report.py --evals` 即可。测试是**只给题目、逐位生成**；整个答案及EOS正确才计一题正确。ID表示两个输入各不超过20位；OOD表示至少一个超过20位，两个都不超过100位。

报告阅读：`comparison.csv` 看每个seed的ID/OOD，`seed_statistics.json` 看均值与差异，PNG看长度格，`errors.jsonl`看错例。保留 `manifest/plan/run/status/provenance` 及 checkpoint，报告不能把原论文成绩作为本机成绩。

## 10. 在PyCharm中点击运行，而不是输命令

Terminal方式最容易批量操作。也可以使用社区版运行配置：

1. 点击 **Run → Edit Configurations → + → Python**。
2. **Name** 填 `01_prepare`；**Script path** 选择根目录 `run.py`，若支持宏可以填 `$PROJECT_DIR$/run.py`。
3. **Parameters** 填 `prepare --output data/add20 --samples 200000 --train-max 20 --per-cell 64`。
4. **Python interpreter** 选择项目 `.venv`；**Working directory** 填 `$PROJECT_DIR$` 或从界面选择项目根目录；勾选 **Add content roots to PYTHONPATH**（若有）。
5. Apply → Run。继续复制配置并修改名称和参数：

| Name | 脚本 | Parameters示例 |
| --- | --- | --- |
| `00_environment` | `check_environment.py` | `--require-cuda` |
| `02_audit_data` | `run.py` | `audit --data data/add20` |
| `03_benchmark_abacus` | `run.py` | `benchmark --preset abacus --data data/add20 --output bench/abacus.json` |
| `04_train_abacus` | `run.py` | `train --preset abacus --seed 42 --steps 24000 --data data/add20 --run runs/manual_abacus42` |
| `05_eval_abacus` | `run.py` | `eval --checkpoint runs/manual_abacus42/best.pt --examples data/add20/test.jsonl --output evals/manual_abacus42 --batch 16` |
| `06_report` | `report.py` | `--evals evals/manual_abacus42 --output reports/manual_abacus42` |

表里的24,000仍须替换成本机预算确定的共同步数。PowerShell `$steps` 不是PyCharm Parameters语法；这里输入整数。普通右键 `run.py → Run` 会因为缺少子命令而报 `command required`，必须配置Parameters。

项目不提交机器专属 `.idea` 配置。IDE中的脚本定位可由界面生成完整路径，但所有数据/权重参数必须按表使用相对路径。参照 [JetBrains 解释器说明](https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html)及 [项目运行说明](https://www.jetbrains.com/help/pycharm/creating-and-running-your-first-python-project.html)。

## 11. 常见问题

| 现象 | 处理 |
| --- | --- |
| `No module named torch` | PyCharm切换至`.venv`，用`.venv/Scripts/python.exe -m pip`安装，避免全局pip |
| `CUDA unavailable` | 执行环境检查及`nvidia-smi`，确认torch带cu121；安装CPU torch不支持GPU训练 |
| `py -3.12`找不到 | 安装Python3.12及Launcher，重新打开终端 |
| `manifest.json`不存在 | Working directory设项目根目录；先prepare，再audit/train |
| `samples must be divisible by 400` | 20位训练共400格，样本数用200000/800000/2000000等400的倍数 |
| `Refusing to overwrite` | 旧结果被保护；新实验换目录，恢复训练用resume/continue |
| 无法执行`.ps1`或激活环境 | 无需激活，不必改变系统执行策略，直接调用`.venv/Scripts/python.exe` |
| `Frozen plan mismatch` | 恢复原配置；改变配置必须创建新suite目录并重跑公平对照 |
| CUDA OOM | 全方法微批16/累积8，重新测速并新建实验；eval独立减batch |
| 无best.pt | 未到验证步或训练失败；先检查日志，按原配置恢复到验证点 |
| ID/OOD准确率低 | 先检查ID学习、数据审计、训练完成状态；不要抄99%结果，见`PROTOCOL.md` |

官方源码复查：ZIP源码快照固定于`86022a57d38c0fde46444d62e8dcbebcc0af614c`，运行`python audit_project.py`核对`reference/upstream_snapshot.json`中的文件哈希即可。若另行clone了带Git历史的仓库，可使用`git rev-parse HEAD`核对commit；重新clone请用一个新目录，不覆盖现有快照。本周运行入口为根目录脚本，官方`shells/*.sh`是原始实验参考，包含集群及路径配置，不直接用于Windows小规模实验。
