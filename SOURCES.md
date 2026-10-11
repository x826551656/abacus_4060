# 来源与实现归属

## 论文

McLeish et al., **Transformers Can Do Arithmetic with the Right Embeddings**, arXiv:2405.17399v2，2024-12-23版本，NeurIPS 2024。

- 本次核对公开NeurIPS论文和arXiv页面；当前文件夹没有论文PDF，未将其记为用户已提供文件。
- NeurIPS论文：https://proceedings.neurips.cc/paper_files/paper/2024/file/c35986bc1ee29b31c1011481b77fe540-Paper-Conference.pdf
- 公开页面：https://arxiv.org/abs/2405.17399
- 3.1：数字内部位序、随机起点、同数位共享embedding。
- 3.2：输入注入、循环层与progressive loss。
- 4.2/4.3：乘法与排序；此交付的代码只实现加法。
- 5：自然语言任务未被验证的限制。
- 附录 A.2：数据；Table 3：参数量；Table 5（PDF第24页）：默认超参数。

## 官方代码

- 仓库：https://github.com/mcleish7/arithmetic
- 本次核对固定 commit：`86022a57d38c0fde46444d62e8dcbebcc0af614c`。
- 该commit页面：https://github.com/mcleish7/arithmetic/tree/86022a57d38c0fde46444d62e8dcbebcc0af614c
- standalone Abacus：`abacus.py`，保留其原始文件在 `reference/abacus_official.py`，用于测试helper及embedding查表语义。
- FIRE及集成Abacus：`cramming/architectures/embeddings.py`。
- 循环与progressive：`cramming/architectures/crammed_depthrecurrent.py`，尤其`forward_progressive`及no-grad prefix。
- 官方MIT license副本：`reference/LICENSE-official.txt`。

standalone文件原本省略`torch/random`导入，本工程测试时在独立namespace补充imports后执行，不修改保留的原文件。官方snapshot不等于本工程模型源码：本工程重写了数据、轻量模型、训练器和评估器，并在`DEVIATIONS.md`列出区别。

官方`max_k=99`是0–99的shift，helper初始数位是1；本工程`offset_k=100`直接表示β∈1–100。二者不是差一位的错误。非数字索引0始终不加shift。

FIRE核对的是官方`max_length=0`连续位置条件，不是最强随机化FIRE全部实验。缓存解码必须使用全局query位置，不能每个生成token都从query位置0重新归一化。

## 运行环境来源

- PyTorch官方历史版本：https://pytorch.org/get-started/previous-versions/
- 选择2.5.1与cu121 wheel对应官方Linux/Windows版本命令；无需安装本任务不使用的torchvision/torchaudio。
- 安装页：https://pytorch.org/get-started/locally/
- 源码采用PyTorch原生SDPA、torch.amp、AdamW，不依赖外部自定义CUDA算子。

核对日期：2026-10-09。固定版本用于本项目可复查，不声称是当前最新版。CPU验证环境与推荐用户CUDA环境的区别见`TEST_REPORT.md`。

## 引用与报告规范

课程报告注明原论文及上述固定commit；将论文原图用于讲解时标“原论文”，将本工程输出图标“资源缩减复现结果”。不能把原文99.1%、6×外推或乘法/排序表格成绩标为本机实验成绩。

本次已通过Git下载作者完整仓库到 `reference/upstream/`，核对上述commit并保持作者源码未修改；独立Abacus快照及license仍在reference根目录。根目录运行代码是资源缩减实现，不冒称作者原始训练器。本次不包含一周训练所得研究权重或论文PDF副本；smoke权重只用于流程验证。官方逐文件清单见 `evidence/FILE_INVENTORY.csv`。
