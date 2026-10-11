# 交付时的实际验证证据

下列CPU与2M数据文件为项目原有历史记录。本次便携打包核对见`LOCAL_VALIDATION.md`、`folder_audit.json`和`FILE_INVENTORY.csv`，不代表已在接收者RTX4060运行。第三方环境及20万条正式数据由接收者按手册创建。

- `default_data_manifest.json`：实际生成的2M数据协议和文件hash。
- `default_data_audit.json`：2M训练行全部审计通过。
- `cpu_smoke_status.json`：tiny CPU 10步训练状态，不是4060研究训练。
- `cpu_smoke_summary.json`：tiny CPU测试准确率为0，保留真实结果。

不包含训练数据或checkpoint，也不包含伪造GPUbenchmark。正式复现时生成自己的证据并按新目录保存，勿覆盖这些交付验证记录。
