"""Run from the project root: python check_environment.py --require-cuda."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Check the Python selected in PyCharm")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output", default="evidence/windows_environment.json")
    args = parser.parse_args()
    if not Path("run.py").is_file():
        raise SystemExit("Set Working directory to the abacus_4060 project root.")
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "packages": {}, "cuda_available": False}
    for package in ("torch", "numpy", "matplotlib", "pytest"):
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            raise SystemExit(f"Missing {package}; install dependencies into this interpreter.")
    import torch
    result["cuda_runtime"] = torch.version.cuda
    result["cuda_available"] = torch.cuda.is_available()
    if result["cuda_available"]:
        result.update(gpu=torch.cuda.get_device_name(0),
                      vram_bytes=torch.cuda.get_device_properties(0).total_memory,
                      bf16_supported=torch.cuda.is_bf16_supported())
        x = torch.randn(32, 32, device="cuda")
        result["cuda_matmul_finite"] = bool(torch.isfinite(x @ x).all())
        torch.cuda.synchronize()
    if args.require_cuda and not result["cuda_available"]:
        raise SystemExit("CUDA unavailable. Check NVIDIA driver and CUDA-enabled torch.")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
