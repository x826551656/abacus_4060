# 文件夹核对与用途

完整逐文件清单为`evidence/FILE_INVENTORY.csv`：每个文件一行，含相对地址、数据/代码/说明/第三方参考等分类、用途、大小、SHA256。用Excel或PyCharm打开即可。刷新命令：`python audit_project.py`，当前目录必须为项目根目录；审计会核对官方commit、仓库是否修改，以及本工程Python源码是否写死绝对路径。

## 顶层目录

```text
abacus_4060/
├── abacus/                 [代码] 轻量复现实现
├── tests/                  [代码] CPU与CUDA正确性验证
├── reference/              [第三方源码] 官方许可证、独立Abacus快照
│   └── upstream/           [第三方源码] 官方完整源码快照，固定commit，ZIP不带Git历史
├── data_examples/          [示例数据] 三道可读格式示例，正式数据由接收者生成
├── data/                   [数据] 本地生成的小规模训练/验证/测试集
├── bench/                  [实测] 六方法吞吐和显存
├── runs/                   [输出] 实验计划、模型、优化器、训练日志
├── evals/                  [输出] 逐题预测、指标、错例、来源
├── reports/                [输出] 实测热力图、比较表、错误统计
├── evidence/               [证据] 文件审计、环境、测试验证记录
├── .venv/                  [第三方环境] 接收者创建，ZIP不包含
└── *.py / *.md             [入口代码 / 中文说明]
```

`.venv/.idea/__pycache__/.pytest_cache/.git/dist`不纳入逐源文件哈希清单：分别为安装环境、IDE配置、Python缓存、测试缓存、Git元数据、打包产物。训练数据和结果仍纳入清单。清单与它自己的JSON摘要不相互哈希，避免自引用。

## 根目录每个入口和说明

| 文件 | 作用 | 运行方式 |
| --- | --- | --- |
| `run.py` | prepare/audit/benchmark/train/eval/demo统一入口 | `python run.py --help` |
| `suite.py` | 八组实验预览、冻结、顺序训练与恢复 | `python suite.py --help` |
| `budget.py` | 根据六方法实测计算共同更新预算 | `python budget.py --help` |
| `report.py` | 读取真实测试结果生成图表及seed统计 | `python report.py --help` |
| `analyze_errors.py` | 错误类型与进位链诊断 | `python analyze_errors.py --help` |
| `check_environment.py` | 当前解释器和CUDA环境检查 | `python check_environment.py --require-cuda` |
| `audit_project.py` | 完整文件分类、作用、SHA256和来源核对 | `python audit_project.py` |
| `package_project.py` | 用Python标准库生成便携ZIP并核对完整性，不安装第三方库 | `python package_project.py` |
| `README.md` | 项目研究与执行总览 | 阅读 |
| `USER_GUIDE.md` | 从安装软件到PyCharm配置、运行和报告的逐步说明 | 从这里开始 |
| `ENVIRONMENT.md` | 第三方库、解释器、原仓库环境区别 | 阅读 |
| `DIRECTORY_GUIDE.md` | 当前文件职责与目录分类 | 本文件 |
| `SOURCE_WALKTHROUGH.md` | 数据、模型、缓存、训练原理逐步解释 | 阅读 |
| `DEVIATIONS.md` | 与原论文的模型/数据/优化/指标差异 | 报告必须参考 |
| `PROTOCOL.md` | 一周公平对照、预算、验收与失败处理规则 | 正式训练前阅读 |
| `SOURCES.md` | 论文链接、固定官方版本及实现归属 | 引用来源 |
| `TEST_REPORT.md` | 以前CPU验证记录；不是本机GPU结果 | 阅读历史证据 |
| `LOCAL_VALIDATION.md` | 本次源码/文件/便携包静态核对；GPU由接收者验证 | 交付证据 |
| `requirements.txt` | NumPy/Matplotlib/pytest固定依赖 | pip安装 |
| `LICENSE` | 本工程许可，保留作者归属 | 阅读 |
| `.gitignore` | 不提交环境、大数据、权重和生成目录 | Git使用 |

## 复现模块和测试

| 文件 | 作用 |
| --- | --- |
| `abacus/__init__.py` | Python包初始化 |
| `abacus/config.py` | 超参数，NoPE/FIRE/Abacus/fixed/II/loop等预设 |
| `abacus/data.py` | 字符tokenizer、数据生成、分层采样、哈希split、batch |
| `abacus/model.py` | Pre-LN decoder、位置表、FIRE attention bias、循环共享 |
| `abacus/training.py` | AMP、优化、梯度累积、ID验证、checkpoint/恢复、测速 |
| `abacus/evaluation.py` | 真实贪心自回归、KV缓存、exact match、Wilson区间 |
| `tests/test_core.py` | 数位/标签/隔离/因果性/cache/梯度/恢复/单题过拟合 |
| `tests/test_cli.py` | 命令入口、冻结规则、预算来源规则 |
| `tests/test_cuda.py` | 六核心方法默认模型的实际CUDA反向和cache一致性 |

## 官方仓库逐层说明

来源：`https://github.com/mcleish7/arithmetic`，固定commit `86022a57d38c0fde46444d62e8dcbebcc0af614c`，MIT许可。下载后未改作者文件；每个具体文件都在CSV中有记录。ZIP不带`.git`；`reference/upstream_snapshot.json`记录版本与每个源码文件哈希，让接收者不用Git也可核对。

| 路径/文件 | 作者原实现作用 |
| --- | --- |
| `reference/abacus_official.py` | 原独立Abacus文件快照，测试时补导入再执行 |
| `reference/LICENSE-official.txt` | 作者MIT许可副本 |
| `upstream/abacus.py` | 独立数位嵌入与随机起点机制 |
| `upstream/pretrain.py` | Hydra训练调度入口 |
| `upstream/create_data_split.py` | 算术和排序数据生成/划分 |
| `upstream/create_pos_or_variants.py` | 位置提示、输入格式变体 |
| `upstream/arithmetic_eval_quicker.py` | 算术长度网格测试 |
| `upstream/sort_eval.py` | 排序任务单独测试 |
| `upstream/pretty_plotter*.py` | 小网格、大网格和排序结果绘图 |
| `upstream/dataset_analysis.py` | 数据统计分析 |
| `upstream/gen_eval_script.py` | 分块测试命令生成 |
| `upstream/load_local_model.py` | 模型加载示例 |
| `upstream/upload_processed_dataset.py` | 作者预处理数据上传工具 |
| `upstream/cramming/architectures/` | attention、embedding、普通及循环Transformer、loss组件 |
| `upstream/cramming/data/` | 原框架分词、预处理、去重与课程排序 |
| `upstream/cramming/backend/` | 原训练后端、优化器和学习率/批日程 |
| `upstream/cramming/config/arch/` | 作者不同模型的Hydra配置 |
| `upstream/cramming/config/data/` | 算术及框架其他数据源配置；不等于下载了这些数据 |
| `upstream/cramming/config/train/` | 作者训练与优化器配置 |
| `upstream/cramming/config/impl/` | 实现、设备和后端配置 |
| `upstream/cramming/config/eval/` | 原框架评估任务配置 |
| `upstream/cramming/config/wandb/` | 有/无WandB记录配置 |
| `upstream/cramming/config/hydra/` | 运行和日志配置 |
| `upstream/shells/*.sh` | 原加法/循环/乘法/排序/OR/生成/测试命令 |
| `upstream/setup.cfg`、`pyproject.toml`、`MANIFEST.in` | 原框架包构建与全部依赖 |
| `upstream/README.md`、`LICENSE`、隐藏配置 | 作者说明、许可和开发设置 |

表中的`upstream/`完整相对前缀均为`reference/upstream/`。该仓库有继承自cramming的其他语言数据和模型配置，并非本周需要运行的实验。运行根目录轻量脚本即可。

## 生成结果各文件

| 文件 | 作用 |
| --- | --- |
| `runs/week1/plan.json` | 八组固定实验配置 |
| `runs/<name>/run.json` | 本run参数、数据hash和实际环境 |
| `runs/<name>/train.jsonl` | 每次更新loss/LR/梯度/耗时/显存，验证时附ID分数 |
| `runs/<name>/last.pt` | 最近可恢复的模型+优化器+RNG+数据cursor |
| `runs/<name>/best.pt` | 由ID验证选择的最佳checkpoint |
| `runs/<name>/status.json` | 是否完成共同预算、最终步数 |
| `evals/<name>/summary.json` | ID/OOD宏/微exact match |
| `evals/<name>/grid.csv` | 每个长度格的真实n、正确数、EOS、区间 |
| `evals/<name>/predictions.jsonl` | 全部逐题网络预测及计分 |
| `evals/<name>/errors.jsonl` | 错误子集 |
| `evals/<name>/provenance.json` | 相对checkpoint地址、hash、完成步数、数据及测试hash |
| `reports/<name>/*.png` | 真实长度格热力图 |
| `reports/<name>/comparison.csv` | 方法与seed比较 |
| `reports/<name>/seed_statistics.json` | seed统计 |
| `bench/*.json` | 更新秒数、显存峰值和实测环境 |
| `budget.json` | 一周资源估算与统一训练步数 |
| `evidence/default_data_*.json`、`cpu_smoke_*.json` | 此前Linux/CPU与200万数据的历史证据 |
| `evidence/windows_environment.json` | 本次Windows/CUDA环境检查 |
| `evidence/folder_audit.json`、`FILE_INVENTORY.csv` | 本次目录核对摘要与逐文件清单 |
| `reference/upstream_snapshot.json` | 官方固定commit及全部原源码文件hash，ZIP离线核对使用 |
| `PACKAGE_MANIFEST.json` | ZIP内每个交付文件的SHA256（由打包工具放入ZIP） |
| `dist/abacus_4060_delivery.zip`、`.sha256`、`PACKAGE_CHECK.json` | 可发送的项目包、包hash、实际打包验证记录；不嵌套进ZIP |

没有数据文件时先prepare；没有权重时先train。说明里出现的`runs/week1/`是正式训练目标目录，不意味着一周训练已经完成。
