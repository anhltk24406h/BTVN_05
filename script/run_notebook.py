"""Execute the notebook from a clean, project-local kernel and save its outputs."""

from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    # Keep all runtime/config writes within the project, not the user's profile.
    for key, directory in {
        "JUPYTER_CONFIG_DIR": ".jupyter/config",
        "JUPYTER_DATA_DIR": ".jupyter/data",
        "JUPYTER_RUNTIME_DIR": ".jupyter/runtime",
        "IPYTHONDIR": ".ipython",
        "MPLCONFIGDIR": ".matplotlib",
    }.items():
        target = ROOT / directory
        target.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(target)
    os.environ["PYTHONUTF8"] = "1"
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    import nbformat
    from nbclient import NotebookClient

    kernel_dir = ROOT / ".jupyter" / "kernels" / "portfolio-risk"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Portfolio risk (project Python)", "language": "python",
    }), encoding="utf-8")
    manager = KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)], ensure_native_kernel=False)
    km = KernelManager(kernel_name="portfolio-risk", kernel_spec_manager=manager)
    path = ROOT / "notebook_code" / "portfolio_risk_2023_2024.ipynb"
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            if i == 0 or nb.cells[i-1].cell_type != "markdown":
                raise ValueError(f"Code cell {i} needs a preceding explanatory Markdown cell")
    client = NotebookClient(nb, km=km, timeout=180, allow_errors=False,
                            resources={"metadata": {"path": str(ROOT)}})
    # For a supplied KernelManager nbclient leaves cleanup to the caller.
    try:
        client.execute()
    finally:
        if km.has_kernel:
            km.shutdown_kernel(now=True)
        km.cleanup_resources()
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    if any(c.execution_count is None for c in code_cells):
        raise RuntimeError("Notebook execution is incomplete")
    errors = [o for c in code_cells for o in c.outputs if o.output_type == "error"]
    if errors:
        raise RuntimeError(f"Notebook contains {len(errors)} error outputs")
    nbformat.validate(nb)
    temporary = path.with_suffix(".executed.ipynb")
    nbformat.write(nb, temporary)
    temporary.replace(path)
    result = json.loads((ROOT / "data" / "processed" / "analysis_results.json").read_text(encoding="utf-8"))
    print(f"Executed and saved {len(code_cells)} code cells: {path}")
    print(json.dumps({"portfolio": result["portfolio"], "risk": result["risk"],
                      "validation": result["validation"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
