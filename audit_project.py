"""Inventory all project/source files with relative names; exclude installed libraries."""
import ast
import csv
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

COMMIT = "86022a57d38c0fde46444d62e8dcbebcc0af614c"
SKIP = {".git", ".venv", ".idea", "__pycache__", ".pytest_cache", "dist"}
PURPOSE = {
    "run.py": "主入口：数据生成、审计、测速、训练、评估、单题演示",
    "suite.py": "冻结并执行八组公平对照实验，支持分批和恢复",
    "budget.py": "根据六方法本机测速估算一周共同更新预算",
    "report.py": "读取真实评估结果，输出比较CSV和热力图",
    "analyze_errors.py": "统计预测错误类型及进位链分组",
    "check_environment.py": "检查当前解释器、第三方库、GPU和CUDA计算",
    "audit_project.py": "生成完整相对路径文件清单、哈希和源码路径检查",
    "package_project.py": "使用Python标准库打包项目，排除环境/缓存/Git元数据",
    "abacus/__init__.py": "将abacus标识为Python包",
    "abacus/config.py": "模型和训练超参数，以及位置嵌入/循环预设",
    "abacus/data.py": "数字字符编码、分层小数据生成、哈希隔离及批流",
    "abacus/model.py": "轻量Transformer、Abacus/FIRE/NoPE/APE、输入注入和循环",
    "abacus/training.py": "优化器、混合精度、加权梯度累积、保存恢复和测速",
    "abacus/evaluation.py": "KV缓存自回归生成、精确匹配指标和逐题结果",
    "tests/test_core.py": "数位、标签、因果性、缓存、梯度及恢复正确性测试",
    "tests/test_cli.py": "入口、实验冻结及预算限制集成测试",
    "tests/test_cuda.py": "本机CUDA前后向及缓存一致性测试；无GPU时跳过",
    "reference/abacus_official.py": "官方Abacus独立源码快照，仅作一致性核对",
    "reference/LICENSE-official.txt": "官方源码MIT许可证",
}
OFFICIAL = {
    "abacus.py": "作者独立Abacus嵌入实现（原文件缺少导入，仅核对）",
    "create_data_split.py": "作者算术/排序数据生成和划分脚本",
    "create_pos_or_variants.py": "作者位置提示或数据格式变体生成",
    "pretrain.py": "作者Hydra训练入口",
    "arithmetic_eval_quicker.py": "作者加法/乘法网格自回归评估",
    "sort_eval.py": "作者排序任务评估",
    "pretty_plotter.py": "作者分块结果网格绘图",
    "pretty_plotter_big.py": "作者大规模网格结果绘图",
    "pretty_plotter_sort.py": "作者排序结果汇总绘图",
    "dataset_analysis.py": "作者数据分布分析",
    "gen_eval_script.py": "作者评估命令生成",
    "load_local_model.py": "作者本地模型加载示例",
    "upload_processed_dataset.py": "作者预处理数据上传工具",
    "setup.cfg": "作者完整cramming依赖与包配置；不用于本项目安装",
    "pyproject.toml": "作者Python构建工具配置",
    "MANIFEST.in": "作者包发布文件清单规则",
    "LICENSE": "作者仓库MIT许可证",
    "README.md": "作者原仓库运行说明",
    ".gitignore": "作者Git忽略规则",
    ".pre-commit-config.yaml": "作者代码检查钩子配置",
}


def describe(path):
    if path in PURPOSE:
        return "复现代码/测试" if not path.startswith("reference/") else "官方参考", PURPOSE[path]
    if path.startswith("reference/upstream/"):
        name = path.removeprefix("reference/upstream/")
        purpose = OFFICIAL.get(name)
        if not purpose:
            if name.startswith("shells/"):
                purpose = "作者" + Path(name).stem + "实验Bash命令；Linux集群参数仅供核对"
            elif name.endswith("__init__.py"):
                purpose = "作者Python子包初始化和接口导出"
            elif name.startswith("cramming/config/"):
                purpose = "作者Hydra配置：" + name.removeprefix("cramming/config/")
            elif name.startswith("cramming/architectures/"):
                purpose = "作者模型组件：" + Path(name).stem
            elif name.startswith("cramming/backend/optimizers/"):
                purpose = "作者优化器及日程组件：" + Path(name).stem
            elif name.startswith("cramming/backend/"):
                purpose = "作者训练后端：" + Path(name).stem
            elif name.startswith("cramming/data/"):
                purpose = "作者数据预处理/分词组件：" + Path(name).stem
            else:
                purpose = "作者框架工具：" + Path(name).stem
        return "官方完整仓库（参考）", purpose
    if path.startswith("data_examples/"):
        return "数据格式示例", "少量可读加法格式示例，不用于正式训练/测试"
    if path.startswith("data/"):
        return "数据", "训练数组" if path.endswith("train.npy") else "验证/测试题或数据生成与审计记录"
    if path.startswith("runs/"):
        return "模型/训练结果", "模型权重、训练状态、配置或更新日志"
    if path.startswith("bench/") or path.startswith("benchmarks/"):
        return "本机实测", "真实更新耗时、显存与环境记录"
    if path.startswith("evals/"):
        return "评估结果", "逐题生成、指标网格、错例或来源追踪"
    if path.startswith("reports/"):
        return "报告结果", "基于真实结果生成的图表或错误统计"
    if path.startswith("evidence/"):
        return "验证证据", "历史或当前验证记录；时间和环境以具体文件为准"
    if path.endswith(".md"):
        return "说明文档", "中文说明：" + Path(path).stem
    return "工程配置", "许可证/依赖版本/版本控制规则"


def main():
    if not Path("run.py").exists():
        raise SystemExit("Run from the abacus_4060 project root.")
    rows, hardcoded = [], []
    for base, dirs, files in os.walk("."):
        dirs[:] = sorted(d for d in dirs if d not in SKIP)
        for name in sorted(files):
            p = Path(base) / name
            rel = p.as_posix()
            if rel in {"evidence/FILE_INVENTORY.csv", "evidence/folder_audit.json"}:
                continue  # Avoid self-referential hashes.
            category, purpose = describe(rel)
            digest = hashlib.sha256()
            with p.open("rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(block)
            rows.append(dict(path=rel, category=category, purpose=purpose,
                             bytes=p.stat().st_size, sha256=digest.hexdigest()))
            if p.suffix == ".py" and not rel.startswith("reference/"):
                for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        if re.match(r"^(?:[A-Za-z]:[\\/]|/(?:home|mnt|Users|workspace|tmp)/)", node.value):
                            hardcoded.append(dict(path=rel, line=node.lineno))
    upstream = Path("reference/upstream")
    commit = None
    clean = None
    if (upstream / ".git").exists():
        commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
        clean = not subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip()
    snapshot_path = Path("reference/upstream_snapshot.json")
    snapshot_ok = None
    if snapshot_path.is_file():
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        expected = snapshot["files"]
        actual = {r["path"].removeprefix("reference/upstream/"): r["sha256"]
                  for r in rows if r["path"].startswith("reference/upstream/")}
        snapshot_ok = snapshot["commit"] == COMMIT and actual == expected
        if commit is None:
            commit = snapshot["commit"]
            clean = snapshot_ok
    Path("evidence").mkdir(exist_ok=True)
    with open("evidence/FILE_INVENTORY.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "category", "purpose", "bytes", "sha256"])
        w.writeheader()
        w.writerows(rows)
    result = {"files_in_inventory": len(rows), "excluded_directory_names": sorted(SKIP),
              "official_commit": commit, "expected_official_commit": COMMIT,
              "official_worktree_clean": clean,
              "official_snapshot_hashes_match": snapshot_ok,
              "hardcoded_absolute_path_literals_in_reproduction_code": hardcoded,
              "status": "pass" if commit == COMMIT and clean and snapshot_ok is not False and not hardcoded else "review_required"}
    Path("evidence/folder_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
