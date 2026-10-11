# 已执行的验证记录

> 历史记录：本文件为当前文件夹原有的Linux/CPU验证说明。本次便携交付核对另见`LOCAL_VALIDATION.md`；本文件中的200万数据是旧默认方案，当前默认已经缩为20万。本次没有GPU实测，请接收者按手册运行环境与正确性检查。

验证日期：2026-10-09。本报告只记录实际运行，不把CPU耗时当4060吞吐量，不包含一周CUDA训练成绩。

## 1. 本地环境

| 项目 | 实际值 |
| --- | --- |
| 系统 | Linux x86_64 |
| Python | 3.12.14 |
| PyTorch | 2.5.1+cpu |
| NumPy | 2.3.5 |
| Matplotlib | 3.10.8 |
| pytest | 8.3.3 |
| CUDA/GPU | 未提供，未执行CUDA实验 |

`requirements.txt`固定了这次实际使用的NumPy/Matplotlib/pytest版本。推荐CUDA wheel对应同一PyTorch主版本，但GPU路径不能由CPU通过自动推出。

## 2. 单元与集成测试

实际命令：`python -m pytest -q`（工程根目录）。最后一次结果：

```text
...........................                                              [100%]
27 passed in 8.53s
```

覆盖内容：

- digit span位置和非digit索引0。
- 与保留的官方Abacus helper、查表embedding一致。
- 答案首位预测位置、EOS标签和PAD损失屏蔽。
- 交换操作数split一致；短数字split集合隔离。
- 数据生成、固定顺序恢复和拒绝覆盖。
- 八个preset的因果性与KV cache/full-context logits一致。
- evalβ=1守卫和公共参数同seed一致初始化。
- FIRE缓存全局query位置以及连续位置公式。
- 循环完整反向梯度流。
- cached/uncached真实greedy生成一致。
- 单例160步overfit后网络生成`91`与EOS（12+7=19）正确。
- Wilson区间不是将64/64写成完全无不确定性。
- 默认256维、8层、20位最大输出进位、β=100的loss/backward和索引120梯度。
- 100位输入和101位输出所需304-token上下文可前向。
- token数加权的梯度累积与整体平均梯度一致。
- 训练暂停/恢复后，模型、AdamW状态、数据cursor和RNG与连续训练完全一致（CPU）。
- 首次只执行部分方法也冻结完整八组计划，后续禁止偷偷改预算。
- 已完成run的早期best checkpoint不误标incomplete。
- 预算计算拒绝将CPU计时当作4060计时。

这些是正确性证据，不是对长数字加法准确率的研究结果。缓存一致性测试是FP32 CPU的数值容差比较，CUDA BF16/FP16应再在本机执行tests和小试验。

## 3. 推荐默认数据规模已实际生成与全量审计

实际执行了：

```bash
python run.py prepare --output ../build/add20_full_check --samples 2000000 --train-max 20 --per-cell 64
python run.py audit --data ../build/add20_full_check
```

结果：2,000,000行、20×20长度格、每格5,000行。训练数组 shape=(2000000,65)，uint8，磁盘文件130,000,128 bytes；验证386题，测试10,761题。

全量audit输出：

```json
{
  "status": "pass",
  "train_rows_checked": 2000000,
  "equal_rows_per_length_cell": 5000,
  "checksums": "pass",
  "arithmetic": "pass",
  "canonical_split_isolation": "pass"
}
```

训练数组SHA256：`9a0739da6a3ed284f12523e2821915f42286211272e26a2f9fe6327c25025ade`。完整manifest/audit保留在`evidence/`，默认数据未塞进ZIP；本机可通过相同生成命令重建。hash复核应在相同生成算法与受控环境进行，不将跨版本字节相同作为未验证保证。

## 4. CLI端到端烟雾试验

小数据：400题、train_max=2、每个长度格100题、test lengths=1/2/3。模型33,760参数；tiny、10步、microbatch4、累积2、有效batch8，CPU FP32。

已执行prepare → audit → train → ID validation → best选择 → eval → report → analyze_errors → demo。训练loss从step1约2.75407降至step10约2.68505，所有10步完成。测试ID 20题、OOD 25题，正确数均0；这符合极短训练仅用于检查程序流程的定位。

best在step5由ID验证选择，整个run在step10完成；report正确允许这种选择，不将早期best误认为未完成训练。热力图已生成并视觉检查，颜色全0与原始计分一致，未使用论文或虚构的成功结果。

`demo`对12+7实际输出`====`、未EOS、correct=false，而不是把Python算出的19偷换成网络预测；与单例overfit测试相区别。错误分析将45道测试题归到missing_EOS。真实输出字段示例保留在evidence中。

tiny CPU benchmark入口也运行成功（含累积和optimizer更新），但**该计时不用于预算4060**，也不将几毫秒的小模型速度外推到默认GPU模型。

## 5. 尚未执行 / 不能保证

- 未在真实RTX4060上测试显存峰值、CUDA SDPA、BF16/FP16、驱动兼容性及热降频。
- 未完成默认4.344M模型的24,000步GPU训练或八组研究对照。
- 未证明可达到论文99%准确率或6×外推；测试“上下文可以前向”不等于“算得对”。
- 未逐项复刻cramming、Deepnorm、GELUglu、Post-LN、batch ramp、randomized FIRE、官方TBPTT progressive。
- 未实现乘法、排序、混合自然语言或负数任务；它们在协议中是后续计划。
- 未在Windows/WSL2运行过安装与长训练；提供的是兼容设计和标准操作命令。

正式研究结果必须由用户GPU跑出的run、checkpoint、eval和provenance生成。不要将本烟雾试验、单例过拟合或原论文图表作为本机一周复现成绩。
