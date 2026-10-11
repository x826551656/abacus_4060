"""Build and verify a portable delivery ZIP, using only Python's standard library.

Run: python package_project.py
Does not install any packages or run model training. Run from the project root.
"""
import ast
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

COMMIT = "86022a57d38c0fde46444d62e8dcbebcc0af614c"
SKIP = {".git", ".venv", ".idea", "__pycache__", ".pytest_cache", "dist",
        "runs", "evals", "reports", "bench", "benchmarks", ".downloads"}


def files():
    for base, dirs, names in os.walk("."):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.startswith(".venv"))
        for name in sorted(names):
            path = Path(base) / name
            if path.suffix in {".pt", ".pyc", ".whl", ".npy"}:
                continue
            yield path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if not Path("run.py").exists():
        raise SystemExit("Run from the abacus_4060 project root.")
    out = Path("dist/abacus_4060_delivery.zip")
    if out.exists():
        raise FileExistsError("Delivery ZIP already exists; move it aside before rebuilding.")
    snapshot_path = Path("reference/upstream_snapshot.json")
    if Path("reference/upstream/.git").is_dir():
        commit = subprocess.check_output(["git", "-C", "reference/upstream", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", "reference/upstream", "status", "--porcelain"], text=True).strip()
        if commit != COMMIT or dirty:
            raise ValueError("Official source must match the fixed clean commit before packaging.")
        tracked = subprocess.check_output(["git", "-C", "reference/upstream", "ls-files"], text=True).splitlines()
        snapshot = {"repository": "https://github.com/mcleish7/arithmetic", "commit": commit,
                    "files": {name: digest(Path("reference/upstream") / name) for name in tracked}}
        snapshot_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    elif not snapshot_path.is_file():
        raise ValueError("Official source snapshot metadata is missing.")
    subprocess.run([sys.executable, "audit_project.py"], check=True)
    audit = json.loads(Path("evidence/folder_audit.json").read_text(encoding="utf-8"))
    if audit["status"] != "pass":
        raise ValueError("Project/source audit failed.")
    paths = list(files())
    syntax_checked = []
    for p in paths:
        if p.suffix == ".py" and not p.as_posix().startswith("reference/"):
            ast.parse(p.read_text(encoding="utf-8"), filename=p.as_posix())
            syntax_checked.append(p.as_posix())
    records = {p.as_posix(): digest(p) for p in paths}
    manifest = {"scope": "portable scaled addition reproduction; dependencies installed by recipient",
                "official_commit": COMMIT, "files": records,
                "dependency_binaries_included": False, "trained_research_weights_included": False}
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in paths:
            z.write(p, "abacus_4060/" + p.as_posix())
        z.writestr("abacus_4060/PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(out) as z:
        if z.testzip() is not None:
            raise ValueError("ZIP CRC check failed")
        for name, expected in records.items():
            if hashlib.sha256(z.read("abacus_4060/" + name)).hexdigest() != expected:
                raise ValueError("ZIP file hash mismatch: " + name)
        assert not any(set(Path(n).parts) & SKIP for n in z.namelist())
    sha = digest(out)
    Path("dist/abacus_4060_delivery.sha256").write_text(sha + "  abacus_4060_delivery.zip\n", encoding="ascii")
    result = {"zip": out.as_posix(), "bytes": out.stat().st_size, "sha256": sha,
              "files_in_zip": len(records) + 1, "syntax_checked_python_files": syntax_checked,
              "zip_crc_and_file_hashes": "pass", "source_audit": "pass",
              "dependency_installation": "not performed for delivery",
              "model_runtime_tests": "not performed in this delivery turn"}
    Path("dist/PACKAGE_CHECK.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
