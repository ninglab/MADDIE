import json
import os
from typing import Dict, Any


CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CONFIG_DIR)
DEFAULT_RUNTIME_CONFIG_PATH = os.path.join(CONFIG_DIR, "runtime_config.json")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_runtime_config(path: str = None) -> Dict[str, Any]:
    if not path:
        path = DEFAULT_RUNTIME_CONFIG_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Runtime config file not found: {path}")
    with open(path, "r") as f:
        runtime_config = json.load(f)
    if "global" not in runtime_config:
        raise KeyError("Runtime config missing top-level key: global")
    if "tmp_root" not in runtime_config["global"]:
        raise KeyError("Runtime config missing global.tmp_root")
    for method in ["lbdd", "sbdd"]:
        method_cfg = runtime_config["global"].get(method, {})
        for required_key in ["python_bin", "config", "runner_script"]:
            if required_key not in method_cfg:
                raise KeyError(f"Runtime config missing global.{method}.{required_key}")
    return runtime_config


def _deep_merge_abs(d: Dict[str, Any], keys: list) -> Dict[str, Any]:
    result = dict(d)
    for key in keys:
        if key in result and result[key] and not os.path.isabs(result[key]):
            result[key] = os.path.normpath(os.path.join(PROJECT_ROOT, result[key]))
    return result


def build_resource_from_paths(
    target_name: str,
    pdb_path: str,
    sdf_path: str,
    drug_sheet_path: str,
    runtime_config: Dict[str, Any],
) -> Dict[str, Any]:
    def _abs(path):
        if not path or os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(PROJECT_ROOT, path))

    return {
        "target": target_name,
        "pdb": _abs(pdb_path),
        "sdf": _abs(sdf_path) if sdf_path else None,
        "drug_sheet": _abs(drug_sheet_path) if drug_sheet_path else None,
        "tmp_root": runtime_config["global"]["tmp_root"],
        "lbdd": _deep_merge_abs(dict(runtime_config["global"]["lbdd"]), ["config"]),
        "sbdd": _deep_merge_abs(dict(runtime_config["global"]["sbdd"]), ["config"]),
        "virtual_screen": dict(runtime_config["global"].get("virtual_screen", {})),
        "docking": dict(runtime_config["global"].get("docking", {})),
    }
