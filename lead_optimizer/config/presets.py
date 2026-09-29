"""
Pre-configured tool presets for the lead optimizer.

Provides ready-to-use tool chest and individual tools without requiring users
to set up oracle registries, tool creation, etc. Just import and use.

Example:
    from lead_optimizer.config.presets import tool_chest, screening_tool
    
    agent = LeadOptimizerAgent(tool_chest=tool_chest)
    response, meta = agent.run(prompt, path_to_env)
"""

import os
from pathlib import Path
from typing import Optional

from ..tools import ToolChest
from ..tools import MolOptOptimizerTool
from ..tools import molopt_repo_context
from ..oracles.registry import build_oracle_registry


# Default paths—override via environment or function args
DEFAULT_MOLOPT_REPO = Path(os.getenv("MOLOPT_REPO", "mol-opt"))
DEFAULT_EVALUATOR_ROOT = os.getenv("MADDIE_EVALUATOR_ROOT")
DEFAULT_MOLOPT_OUTPUT_ROOT = Path(os.getenv("MOLOPT_OUTPUT_ROOT", "molopt_output"))


def _output_root(output_root: Optional[Path | str], algorithm: str) -> Path:
    return Path(output_root) if output_root else DEFAULT_MOLOPT_OUTPUT_ROOT / algorithm


def _get_oracle_registry(
    molopt_repo: Optional[Path | str] = None,
    evaluator_root: Optional[Path | str] = None,
) -> dict:
    """Load oracle registry once and cache it."""
    repo = Path(molopt_repo or DEFAULT_MOLOPT_REPO)
    evaluator = evaluator_root or DEFAULT_EVALUATOR_ROOT
    cache_key = (str(repo), str(evaluator))
    if not hasattr(_get_oracle_registry, "_registries"):
        _get_oracle_registry._registries = {}
    if cache_key not in _get_oracle_registry._registries:
        _get_oracle_registry._registries[cache_key] = build_oracle_registry(
            molopt_repo=repo,
            evaluator_root=evaluator,
        )
    return _get_oracle_registry._registries[cache_key]


def create_screening_tool(
    molopt_repo: Optional[Path | str] = None,
    oracle: str = "qed",
    max_oracle_calls: int = 1000,
    output_root: Optional[Path | str] = None,
) -> MolOptOptimizerTool:
    """
    Create a pre-configured MolOpt screening tool.
    
    Screening is useful for ranking molecules from a pool against an oracle.
    """
    repo = Path(molopt_repo or DEFAULT_MOLOPT_REPO)
    out_root = _output_root(output_root, "screening")
    out_root.mkdir(parents=True, exist_ok=True)
    
    return MolOptOptimizerTool(
        molopt_repo=repo,
        algorithm="screening",
        oracle=oracle,
        oracle_registry=_get_oracle_registry(repo),
        source_smi_file=None,  # Use input from request
        output_root=out_root,
        max_oracle_calls=max_oracle_calls,
        freq_log=10,
        patience=2,
        n_jobs=1,
    )


def create_graph_ga_tool(
    molopt_repo: Optional[Path | str] = None,
    oracle: str = "qed",
    max_oracle_calls: int = 1000,
    output_root: Optional[Path | str] = None,
) -> MolOptOptimizerTool:
    """
    Create a pre-configured MolOpt graph genetic algorithm tool.
    
    Graph GA is useful for structural exploration and optimization of input molecules.
    """
    repo = Path(molopt_repo or DEFAULT_MOLOPT_REPO)
    out_root = _output_root(output_root, "graph_ga")
    out_root.mkdir(parents=True, exist_ok=True)
    
    return MolOptOptimizerTool(
        molopt_repo=repo,
        algorithm="graph_ga",
        oracle=oracle,
        oracle_registry=_get_oracle_registry(repo),
        source_smi_file=None,
        output_root=out_root,
        max_oracle_calls=max_oracle_calls,
        freq_log=10,
        patience=5,
        n_jobs=1,
    )


def create_smiles_ga_tool(
    molopt_repo: Optional[Path | str] = None,
    oracle: str = "qed",
    max_oracle_calls: int = 1000,
    output_root: Optional[Path | str] = None,
) -> MolOptOptimizerTool:
    """
    Create a pre-configured MolOpt SMILES genetic algorithm tool.
    
    SMILES GA works with string-level mutations; useful for quick exploration.
    """
    repo = Path(molopt_repo or DEFAULT_MOLOPT_REPO)
    out_root = _output_root(output_root, "smiles_ga")
    out_root.mkdir(parents=True, exist_ok=True)
    
    return MolOptOptimizerTool(
        molopt_repo=repo,
        algorithm="smiles_ga",
        oracle=oracle,
        oracle_registry=_get_oracle_registry(repo),
        source_smi_file=None,
        output_root=out_root,
        max_oracle_calls=max_oracle_calls,
        freq_log=10,
        patience=3,
        n_jobs=1,
    )


def create_tool_chest(
    molopt_repo: Optional[Path | str] = None,
    algorithms: Optional[list[str]] = None,
) -> ToolChest:
    """
    Create a pre-configured tool chest with standard optimizers.
    
    Args:
        molopt_repo: Path to mol-opt repository. Defaults to DEFAULT_MOLOPT_REPO.
        algorithms: List of algorithms to include. Defaults to ["screening", "graph_ga"].
                   Options: "screening", "graph_ga", "smiles_ga"
    
    Returns:
        ToolChest with ready-to-use tools.
    
    Example:
        ```python
        chest = create_tool_chest(algorithms=["screening", "graph_ga"])
        agent = LeadOptimizerAgent(tool_chest=chest)
        ```
    """
    repo = Path(molopt_repo or DEFAULT_MOLOPT_REPO)
    selected = algorithms or ["screening", "graph_ga"]
    
    chest = ToolChest()
    
    if "screening" in selected:
        chest.add("molopt_screening", create_screening_tool(molopt_repo=repo))
    
    if "graph_ga" in selected:
        chest.add("molopt_graph_ga", create_graph_ga_tool(molopt_repo=repo))
    
    if "smiles_ga" in selected:
        chest.add("molopt_smiles_ga", create_smiles_ga_tool(molopt_repo=repo))
    
    return chest


# Lazy-loaded defaults - only instantiate when accessed
# This avoids requiring RDKit at import time for users who don't need presets
_screening_tool = None
_graph_ga_tool = None
_smiles_ga_tool = None
_tool_chest = None


def get_screening_tool():
    """Lazily create and return the default screening tool."""
    global _screening_tool
    if _screening_tool is None:
        _screening_tool = create_screening_tool()
    return _screening_tool


def get_graph_ga_tool():
    """Lazily create and return the default graph GA tool."""
    global _graph_ga_tool
    if _graph_ga_tool is None:
        _graph_ga_tool = create_graph_ga_tool()
    return _graph_ga_tool


def get_smiles_ga_tool():
    """Lazily create and return the default SMILES GA tool."""
    global _smiles_ga_tool
    if _smiles_ga_tool is None:
        _smiles_ga_tool = create_smiles_ga_tool()
    return _smiles_ga_tool


def get_tool_chest():
    """Lazily create and return the default tool chest."""
    global _tool_chest
    if _tool_chest is None:
        _tool_chest = create_tool_chest()
    return _tool_chest


# For backward compatibility, provide convenience properties at module level
# These will be populated on first use
screening_tool = None
graph_ga_tool = None
smiles_ga_tool = None
tool_chest = None
