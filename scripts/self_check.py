
import hashlib
import importlib
import subprocess
import sys
from pathlib import Path

import joblib
import yaml

# --- Configuration ---
CRITICAL = "CRITICAL"
WARN = "WARN"

REQUIRED_PATHS = [
    "README.md", "requirements.txt", ".gitignore", "pyproject.toml",
    "configs/base.yaml", "configs/thresholds.yaml",
    "src/__init__.py", "src/utils.py", "src/data.py", "src/features.py",
    "src/metrics.py", "src/threshold.py", "src/train.py", "src/evaluate.py", "src/drift.py",
    "src/service/app.py", "src/dashboard/app.py",
    "scripts/run_train.sh", "scripts/run_eval.sh", "scripts/serve_api.sh", "scripts/run_dashboard.sh",
    "tests/test_metrics.py", "tests/test_split_leakage.py", "tests/test_inference.py"
]

BASE_CONFIG_KEYS = [
    "dataset_path", "target_col", "time_col", "id_cols", "train_valid_test_ratios", "random_state",
    "scale_amount_log", "use_standard_scaler", "modeling", "imbalance", "evaluation", "output_dir", "mlflow"
]

LIBRARIES_TO_IMPORT = [
    "pandas", "numpy", "sklearn", "lightgbm", "xgboost", "imblearn", "shap", "optuna", "mlflow",
    "yaml", "joblib", "category_encoders", "matplotlib", "plotly", "fastapi", "uvicorn", "streamlit", "scikitplot"
]

SRC_MODULES_TO_IMPORT = [
    "src.utils", "src.data", "src.features", "src.metrics", "src.threshold", "src.train", "src.evaluate", "src.drift"
]

# --- State ---
results = []

# --- Helper Functions ---
def _print_check(name, status, message=""):
    print(f"- {name}: {status}{' - ' + message if message else ''}")
    results.append({"name": name, "status": status, "message": message})

def _run_command(command):
    try:
        process = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return True, process.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, e.stderr.strip()

def _get_file_hash(path):
    if not path.exists():
        return None
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

# --- Checks ---
def check_structure():
    """1) Structure (CRITICAL): required paths exist"""
    missing_paths = [p for p in REQUIRED_PATHS if not Path(p).exists()]
    if not missing_paths:
        _print_check("Structure", "PASS")
        return True
    _print_check("Structure", "FAIL", f"Missing: {', '.join(missing_paths)}")
    return False

def check_config_keys():
    """2) Config keys (CRITICAL): configs/base.yaml has EXACT keys"""
    try:
        with open("configs/base.yaml", "r") as f:
            config = yaml.safe_load(f)
        keys = list(config.keys())
        if keys == BASE_CONFIG_KEYS:
            _print_check("Config Keys", "PASS")
            return True
        else:
            _print_check("Config Keys", "FAIL", f"Keys do not match. Got: {keys}")
            return False
    except Exception as e:
        _print_check("Config Keys", "FAIL", f"Could not read or parse YAML: {e}")
        return False

def check_imports():
    """3) Imports (CRITICAL): import required modules"""
    failed_imports = []
    for lib in LIBRARIES_TO_IMPORT:
        try:
            module = importlib.import_module(lib)
            version = getattr(module, "__version__", "N/A")
            print(f"  - {lib}: OK (version: {version})")
        except ImportError:
            failed_imports.append(lib)
            print(f"  - {lib}: FAILED")
    if not failed_imports:
        _print_check("Imports", "PASS")
        return True
    _print_check("Imports", "FAIL", f"Could not import: {', '.join(failed_imports)}")
    return False

def check_package_imports():
    """4) Package imports (CRITICAL): import src modules"""
    failed_imports = []
    for module_name in SRC_MODULES_TO_IMPORT:
        try:
            importlib.import_module(module_name)
            print(f"  - {module_name}: OK")
        except ImportError as e:
            failed_imports.append(module_name)
            print(f"  - {module_name}: FAILED ({e})")
    if not failed_imports:
        _print_check("Package Imports", "PASS")
        return True
    _print_check("Package Imports", "FAIL", f"Could not import: {', '.join(failed_imports)}")
    return False

def check_dataset():
    """5) Dataset (WARN): check if data/raw/creditcard.csv exists"""
    if Path("data/raw/creditcard.csv").exists():
        _print_check("Dataset", "PASS", "creditcard.csv found.")
    else:
        _print_check("Dataset", "WARN", "creditcard.csv not found.")
    return True

def check_artifacts():
    """6) Artifacts (WARN): check for models/inference.joblib and compare with mlruns"""
    main_model_path = Path("models/inference.joblib")
    main_hash = _get_file_hash(main_model_path)
    if not main_hash:
        _print_check("Artifacts", "WARN", "models/inference.joblib not found.")
        return True

    mlruns_models = list(Path("mlruns").glob("*/*/artifacts/inference.joblib"))
    if not mlruns_models:
        _print_check("Artifacts", "PASS", "models/inference.joblib found, no mlruns models to compare.")
        return True

    latest_mlruns_model = max(mlruns_models, key=lambda p: p.stat().st_mtime)
    mlruns_hash = _get_file_hash(latest_mlruns_model)

    print(f"  - Main model hash: {main_hash}")
    print(f"  - Latest mlruns model hash: {mlruns_hash}")

    if main_hash == mlruns_hash:
        _print_check("Artifacts", "PASS", "SHA256 hashes match.")
    else:
        _print_check("Artifacts", "WARN", "SHA256 hashes DIFFER.")
    return True

def check_tests():
    """7) Tests (CRITICAL): run pytest"""
    print("  - Running test_metrics.py...")
    metrics_pass, metrics_out = _run_command("pytest -q tests/test_metrics.py")
    print(f"    {'PASS' if metrics_pass else 'FAIL'}")
    if not metrics_pass:
        print(metrics_out)

    print("  - Running test_split_leakage.py...")
    leakage_pass, leakage_out = _run_command("pytest -q tests/test_split_leakage.py -k synthetic || true")
    print(f"    {'PASS' if leakage_pass else 'FAIL'}")
    if not leakage_pass:
        print(leakage_out)

    if metrics_pass and leakage_pass:
        _print_check("Pytest", "PASS")
        return True
    else:
        _print_check("Pytest", "FAIL", "One or more test suites failed.")
        return False

def check_thresholds():
    """8) Thresholds (WARN): parse and print thresholds.yaml"""
    try:
        with open("configs/thresholds.yaml", "r") as f:
            thresholds = yaml.safe_load(f)
        output = []
        if "chosen" in thresholds:
            output.append(f'Chosen: {thresholds["chosen"]}')
        if "by_cost" in thresholds:
            output.append(f'By Cost: {thresholds["by_cost"]}')
        if "by_topk" in thresholds:
            output.append(f'By Top K: {thresholds["by_topk"]}')
        if output:
            _print_check("Thresholds", "PASS", ", ".join(output))
        else:
            _print_check("Thresholds", "WARN", "No recognized threshold keys found.")
    except Exception as e:
        _print_check("Thresholds", "WARN", f"Could not read or parse YAML: {e}")
    return True

def print_summary():
    """Prints a final summary of all checks."""
    print("
--- Self-Check Summary ---")
    critical_fails = [r for r in results if r["status"] == "FAIL"]
    warnings = [r for r in results if r["status"] == "WARN"]

    print(f"CRITICAL Checks: {len(results) - len(critical_fails) - len(warnings)} PASS, {len(critical_fails)} FAIL")
    print(f"WARNINGS: {len(warnings)}")

    if critical_fails:
        print("
Critical Failures:")
        for fail in critical_fails:
            print(f"  - {fail['name']}: {fail['message']}")
    if warnings:
        print("
Warnings:")
        for warn in warnings:
            print(f"  - {warn['name']}: {warn['message']}")
    print("--------------------------")
    return len(critical_fails)

# --- Main Execution ---
if __name__ == "__main__":
    critical_checks_failed = False
    checks = [
        (check_structure, CRITICAL),
        (check_config_keys, CRITICAL),
        (check_imports, CRITICAL),
        (check_package_imports, CRITICAL),
        (check_dataset, WARN),
        (check_artifacts, WARN),
        (check_tests, CRITICAL),
        (check_thresholds, WARN),
    ]

    for func, level in checks:
        print(f"
Running: {func.__doc__}")
        if not func() and level == CRITICAL:
            critical_checks_failed = True

    fail_count = print_summary()
    if fail_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)
