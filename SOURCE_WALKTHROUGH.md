# 源码操作逻辑：从一道加法到可复查的实验

## 1. 数据并非模型内部计算器

`data.example(a,b)`先倒序格式化问题，再用Python生成训练真值。训练看到输入/输出配对，而不是竖式算法标签。`prepare()`按长度格生成一个固定数据池，`BatchStream`按种子打乱遍历。哈希split在采样前决定归属，因此不存在训练完后随机切分导致交换题泄漏的步骤。

示例123+45：序列为 `[BOS,3,2,1,+,5,4,=,8,6,1,EOS]`，其中示例中的3等表示字符而非token id；真实字符id由`ENC`给出。词表大小15。

`collate()`产生：

- input：到倒序答案末位1为止。
- target：从第一个数字3到EOS，整体向左移一位。
- target mask：`=`位置之前全设为-100；`=`所在位置的logit预测第一个答案8。

注意：虽然target包含早期输入字符，mask将其排除。不能再额外shift一次，否则会训练成错位任务。有效标签含3个答案字符+1个EOS，而不是只算一个总loss的二分类任务。

## 2. Abacus位置如何生成

`model.local_positions(ids)`先标记digit范围，然后找连续digit span的起点，利用cummax求每个数字内部的位置。加号、等号、BOS/EOS/PAD都不是digit。

若β=37，token3和token5都查第37行数位embedding，token2与token4查第38行。答案8、6、1同样查37、38、39行。这让模型显式拥有“它们属于同一数位”的线索，但没有直接告诉模型结果是多少。

嵌入公式：`inp = token_embedding(ids) + abacus_embedding(local_position + β - 1)`；非数字例外，固定查0。与全局APE不同，跨过'+'后数内位置重新从1开始，而不是接着整条序列继续计数。

训练β来自独立`random.Random(seed+719)`，会保存进checkpoint；一次microbatch共享一个β。eval强制β=1，防止评估时意外随机化造成结果漂移。

## 3. 一个轻量Transformer block

本工程是Pre-LN，数学操作顺序：

1. `u = LayerNorm(x)`。
2. `Q,K,V = Linear(u)`，拆成8个head。
3. 因果attention：只能读当前位置或更早token。NoPE/Abacus不增加全局位置向量；FIRE增加head-specific bias。
4. attention输出投影后残差相加。
5. 第二个LayerNorm → Linear(256→512) → GELU → Linear(512→256) → 残差相加。

最上层LayerNorm后用与token embedding共享的输出权重得到15类logits。训练用交叉熵；greedy生成选择最大logit。这个结构没有显式的“加法模块”或预先写好的进位状态机。

不同位置编码会改变模型能访问的位置信息：

- NoPE：无额外显式位置编码，但因果mask仍提供有序结构。
- APE：每条完整序列按0、1、2…查全局位置表。
- FIRE：给每对query/key加相对位置bias，不直接提供“个位/十位”标签。
- Abacus：在每个数字span内部重置位置表，可显式对齐同数位。
- Abacus+FIRE：前者加输入向量，后者加attention bias，两者不互相替代。

FIRE连续位置的bias输入近似为：

`log(1+abs(c*(q-k))) / (log(1+abs(c*max(q,abs(L_multiplier*512)))) + eps)`。

标量经两层MLP映射为每个head的bias。`q`是全局query位置；推理中的第50个新token不能每次把q当0。该公式核对了官方max_length=0路径，但整个baseline训练方案不是论文最强randomized-FIRE配置。

## 4. 输入注入与循环

普通8×1：8个block，权重各不相同。

输入注入：每个block前令`x=x+inp`，让原始token/位置信息再次进入深层。没有新的算法监督。这里首层也加一次，与某些“从第2层才开始加”的实现不同，已固定在协议中。

循环4×2：4个block调用两遍，总共8次block变换，但只存4个block参数。完整反向传播贯穿两次调用。循环并不会自动使答案生成长度变成两倍。

Optional progressive：full recurrence loss与随机较短recurrence loss混合，两个forward共享β。它是完整BPTT的简化版本；与官方no-grad prefix/截断反传有差异，不属于默认八组。

## 5. 梯度累积为什么要按token数加权

假设微批1有10个有效答案token，微批2有30个。简单`(loss1+loss2)/2`会让微批1的每个token获得3倍相对权重。本工程使用`10/40*loss1 + 30/40*loss2`，等价于对该有效batch全部答案token取均值。

每个更新先收集4个microbatch的token计数，然后依次forward/backward，释放各自计算图；不会把4个图一起保存在显存。全部累积完成后unscale → clip grad → optimizer.step → scaler.update。测试已经比较此方法与直接global token mean的梯度。

## 6. Checkpoint与恢复

保存模型/optimizer/scaler/config/step/best score、数据cursor/epoch、方法RNG、CPU/CUDA RNG和数据manifest hash。tmp写完后原子替换同一run的目标checkpoint，避免半个保存文件被当成完整结果。

恢复时配置和数据必须完全一致，且路径属于同一run。`BatchStream`根据epoch重建同样的permutation，再跳到cursor。LR从绝对训练步恢复，而不是重新warmup。

CPU测试已验证暂停后继续到同一步时模型与AdamW状态逐项一致；不同CUDA硬件/驱动/内核之间不保证逐bit一致。中断后的日志可能含从最近checkpoint重跑的step，整理曲线时按step保留最后一条已恢复记录，不能把日志行数当更新数。

## 7. 自回归推理与KV cache

`generate()`只接收同长度问题前缀。第一次forward读完整前缀，从'='处logit产生答案首位；下一次只读刚预测的token，并缓存过去每个有效层的K/V。

循环模型cache必须是8份有效层状态，而不是4份共享block状态。同一个block第1遍与第2遍看到不同hidden states，其历史K/V不能共用。源码按`repeat*block_layers+k`索引，测试逐位比较cache与完整上下文重算。

Abacus的生成位置根据预测token更新：如果前一个预测是数字，则span继续+1；非数字则断开；如果下一token变成数字，则新span从1开始。非法预测不会被真值“纠正”。EOS之后该样本不再生成有效内容。

解码词表不限制只能选数字。预测'+'、PAD或BOS都是真错误。没有EOS，即使前缀数字对，也按错误计分。最大输出长度为输入最大位数+2（最长和的数字位数+EOS）。

## 8. 评估与图表的证据链

只以ID验证选择best；正式训练完成后冻结checkpoint，再一次性运行test。结果逐题保存预测、真值、正确与否和离线进位统计。`report.py`使用这些真实JSON/CSV，不随机造热力图。

数据hash、测试hash、checkpoint hash、配置与完成步数构成可复查链。热力图每格n不同，故同时保留macro和micro。两个seed是两个训练过程；同一个seed的best和last不是两次独立重复。

最终失败也保留：例如ID高但100位几乎全错，说明该缩减预算尚未取得原论文级外推，不允许将原文图1换成自己的实验图。
