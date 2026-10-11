# 数据格式示例

`FORMAT_EXAMPLES.jsonl`仅供人阅读，说明输入和答案倒序后的字段格式。这三题不是正式训练/验证/测试集，也未按哈希split划分；不要把这个文件传给正式eval来评价复现效果。

接收者执行`python run.py prepare --output data/add20 --samples 200000 --train-max 20 --per-cell 64`后生成真实小数据，约13MB训练数组。数据来源为程序合成正整数加法；无需下载作者20M大数据。
