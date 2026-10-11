# 第三方环境和解释器

## 本周使用的环境

| 层级 | 选择 | 作用 |
| --- | --- | --- |
| 系统 | 接收者Windows，64位 | 目标运行平台 |
| GPU | 接收者RTX4060，按8GB显存设计 | 单卡训练/推理，接收者自行核对 |
| 驱动 | 接收者安装兼容NVIDIA驱动 | 提供CUDA设备接口 |
| IDE | PyCharm社区版/免费核心Python功能 | 编辑、选择解释器、终端、运行配置 |
| 解释器 | Python3.12，项目内`.venv` | 真正执行.py，不使用全局3.13 |
| 深度学习 | torch2.5.1，官方cu121 Windows wheel | Tensor、SDPA、AMP、AdamW、GPU |
| 数组 | numpy2.3.5 | uint8磁盘数组和确定性批顺序 |
| 图表 | matplotlib3.10.8 | CSV结果热力图 |
| 测试 | pytest8.3.3 | 正确性与CUDA验证 |
| 源码版本 | Git，可选于训练但必需于仓库核对 | 保存/确认作者commit |

安装见`USER_GUIDE.md`。发送者无需安装第三方库，交付ZIP不带`.venv`或安装包。接收者执行指令后，`requirements.txt`固定非torch依赖，torch单独从CUDA wheel索引安装；环境检查生成`evidence/windows_environment.json`。`.venv/`是接收者安装出来的第三方库目录，既不是研究代码，也不是训练数据，不需逐个阅读数千个库文件。

驱动报告的CUDA版本高于torch的CUDA12.1并不矛盾：前者是驱动可支持的上限，后者是wheel运行时。环境检查必须同时验证`torch.cuda.is_available()`和一次CUDA矩阵计算。

## 作者原始环境与本工程的关系

官方README开发环境为Python3.10.4；官方`setup.cfg`列出torch、Hydra、datasets、tokenizers、transformers、evaluate、scipy、scikit-learn、pynvml、psutil、einops、safetensors、apache-beam、zstandard、wandb、matplotlib3.8.3、seaborn0.13.2、opencv-python。原数据生成还导入pandas；这些作者依赖大多没有固定精确版本。

本工程保存完整官方仓库供机制/配置核对，Windows小规模代码采用PyTorch重实现，所以上述完整框架依赖**不作为本周安装集合**。不要对`reference/upstream/`执行`pip install .`来覆盖当前小规模环境。若后续逐项运行作者cramming，应建立另一个独立环境，按官方配置重新处理路径和数据，而不是混用两套依赖。

不使用预训练Hugging Face模型，无需额外模型下载、API密钥或WandB账号。标准库`argparse/pathlib/json/csv/hashlib/os/random/time/platform/dataclasses/subprocess`等随Python提供。

## 相对路径约定

工作目录始终为项目根目录。数据`data/add20/`、训练`runs/`、测速`bench/`、测试`evals/`、图表`reports/`、证据`evidence/`。程序不嵌入机器盘符，不将评估checkpoint路径强制写成绝对路径。恢复用目录身份比较避免误加载其他run；文件清单用相对路径保存。

使用`.venv/Scripts/python.exe`选择本项目解释器；PyCharm界面显示的完整解释器位置不是训练数据访问配置。环境可以重新创建，项目移动无需修改源码。作者原始文件保留未改，因此可能有其自己的集群绝对路径；这些文件归为第三方参考源码，本周不执行。
