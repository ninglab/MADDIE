from contextlib import contextmanager
from contextlib import redirect_stderr, redirect_stdout
import importlib
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import yaml

from ..oracles.specs import parse_optimization_objective
from ..schemas import OptimizationObjective, OptimizationRequest
from .base import OptimizationTool


@contextmanager
def quiet_external_output(enabled: bool = True):
    """Suppress noisy optional dependency output unless verbose mode is requested."""
    if not enabled:
        yield
        return

    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            try:
                from rdkit import rdBase
            except ImportError:
                yield
            else:
                with rdBase.BlockLogs():
                    yield


def ensure_rdkit_six_compatibility() -> None:
    """Provide the legacy rdkit.six alias expected by PyTDC 0.4.1."""
    try:
        import rdkit
        from rdkit import six as _rdkit_six
    except ImportError:
        try:
            import rdkit
            import six
        except ImportError:
            return
        sys.modules.setdefault("rdkit.six", six)
        sys.modules.setdefault("rdkit.six.moves", six.moves)
        setattr(rdkit, "six", six)


ensure_rdkit_six_compatibility()


MOL_OPT_ALGORITHMS = {
    "graph_ga": ("molopt.graph_ga", "GraphGA"),
    "screening": ("molopt.screening", "Screening"),
    "smiles_ga": ("molopt.smiles_ga", "SmilesGA"),
    "stoned": ("molopt.stoned", "Stoned"),
    "graph_mcts": ("molopt.graph_mcts", "Graph_MCTS"),
    "moldqn": ("molopt.moldqn", "MolDQN"),
    "gpbo": ("molopt.gpbo", "GPBO"),
    "reinvent": ("molopt.reinvent", "REINVENT"),
    "reinvent_selfies": ("molopt.reinvent_selfies", "REINVENT_SELFIES"),
    "mimosa": ("molopt.mimosa", "MIMOSA"),
    "selfies_ga": ("molopt.selfies_ga", "SelfiesGA"),
    "smiles_vae": ("molopt.smiles_vae", "SmilesVAE"),
    "jt_vae": ("molopt.jt_vae", "JTVAE"),
}


MOL_OPT_ALGORITHM_CONFIGS = {
    "smiles_vae": "molopt/smiles_vae/hparams_default.yaml",
    "jt_vae": "molopt/jt_vae/hparams_default.yaml",
}


MOL_OPT_SMI_FILE_OVERRIDES = {
    "jt_vae": None,
}


TOXICITY_ORACLES = {"herg", "dili", "clintox", "mutagenicity", "carcinogens"}


BENCHMARKED_MOL_OPT_ALGORITHMS = [
    "graph_ga",
    "graph_mcts",
    "gpbo",
    "mimosa",
    "moldqn",
    "reinvent",
    "reinvent_selfies",
    "screening",
    "smiles_ga",
    "selfies_ga",
    "stoned",
]


MOL_OPT_ALGORITHM_DESCRIPTIONS = {
    "graph_ga": "graph genetic algorithm for structural exploration",
    "screening": "screens and ranks molecules from an input pool",
    "smiles_ga": "SMILES genetic algorithm for string-level edits",
    "selfies_ga": "SELFIES genetic algorithm with robust molecule validity",
    "stoned": "SELFIES mutation method for local exploration",
    "graph_mcts": "graph Monte Carlo tree search",
    "moldqn": "reinforcement learning optimizer",
    "gpbo": "Gaussian-process Bayesian optimizer",
    "reinvent": "generative recurrent model optimizer",
    "reinvent_selfies": "SELFIES variant of REINVENT",
    "mimosa": "fragment and substructure optimizer",
    "smiles_vae": "SMILES variational autoencoder optimizer",
    "jt_vae": "junction-tree variational autoencoder optimizer",
}


@contextmanager
def molopt_repo_context(molopt_repo: Path | None):
    """Temporarily expose an external MolOpt checkout for imports and files."""
    if molopt_repo is None:
        yield
        return

    repo = molopt_repo.resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    old_cwd = Path.cwd()
    os.chdir(repo)
    try:
        yield
    finally:
        os.chdir(old_cwd)


class NamedOracle:
    """Small callable wrapper so MolOpt can build readable run names."""

    def __init__(self, name: str, evaluator, objective: OptimizationObjective):
        self.name = name
        self.evaluator = evaluator
        self.objective = objective
        self.__name__ = name  # For compatibility with MolOpt's TempOracle

    def __call__(self, smiles):
        score = self.raw_value(smiles)
        if self.objective.direction == "increase":
            return max(score, 0.0)
        if self.objective.direction == "decrease":
            if self.objective.property == "sascore":
                return max(10.0 - score, 0.0)
            if self.objective.property in TOXICITY_ORACLES:
                return max(1.0 - score, 0.0)
            if self.objective.threshold is None:
                return 1.0 / (1.0 + max(score, 0.0))
            return max(self.objective.threshold - score, 0.0)
        midpoint = (self.objective.minimum + self.objective.maximum) / 2
        span = max(self.objective.maximum - self.objective.minimum, 1e-12)
        if score < self.objective.minimum:
            return max(0.0, 1.0 - ((self.objective.minimum - score) / span))
        if score > self.objective.maximum:
            return max(0.0, 1.0 - ((score - self.objective.maximum) / span))
        return 1.0 - abs(score - midpoint) / (span / 2)

    def raw_value(self, smiles) -> float:
        return float(self.evaluator(smiles) or 0)


class MolOptOptimizerTool(OptimizationTool):
    """Adapter that lets the lead optimizer call one MolOpt algorithm."""

    supports_adaptive_fallback = True
    supports_agent_retry = True
    supports_property_scoring = True

    def __init__(
        self,
        molopt_repo: str | os.PathLike | None,
        algorithm: str = "graph_ga",
        oracle: str = "qed",
        oracle_fn=None,
        objective: OptimizationObjective | None = None,
        oracle_registry: dict[str, object] | None = None,
        name: str | None = None,
        max_oracle_calls: int = 1000,
        freq_log: int = 10,
        patience: int = 5,
        seed: int = 0,
        n_jobs: int = 1,
        verbose: bool = False,
        source_smi_file: str | os.PathLike | None = None,
        output_root: str | os.PathLike | None = None,
        algorithm_cls=None,
    ):
        if algorithm not in MOL_OPT_ALGORITHMS:
            known = ", ".join(sorted(MOL_OPT_ALGORITHMS))
            raise ValueError(f"Unknown MolOpt algorithm '{algorithm}'. Known algorithms: {known}")

        self.molopt_repo = Path(molopt_repo).resolve() if molopt_repo else None
        self.algorithm = algorithm
        self.oracle = oracle
        self.oracle_fn = oracle_fn
        self.objective = objective
        self.oracle_registry = oracle_registry or {}
        self.name = name or f"molopt_{algorithm}"
        self.max_oracle_calls = max_oracle_calls
        self.freq_log = freq_log
        self.patience = patience
        self.seed = seed
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.source_smi_file = Path(source_smi_file).resolve() if source_smi_file else None
        self.output_root = Path(output_root).resolve() if output_root else None
        self.algorithm_cls = algorithm_cls
        self._score_cache: dict[tuple[str, str], float] = {}

    def documentation(self) -> str:
        description = MOL_OPT_ALGORITHM_DESCRIPTIONS.get(self.algorithm, "molecular optimizer")
        return (
            f"MolOpt {self.algorithm}: {description}. Uses MADDIE oracles from "
            "exactly one scalar property objective per run; this integration does "
            "not perform simultaneous multi-property optimization. Supports "
            "natural-language goals such as increase QED above 0.85 or decrease "
            "TPSA below 90. Direction applies to the score returned by the "
            "registered evaluator; evaluator-backed toxicity oracles return raw risk."
        )

    def setup_tool(self):
        if self.molopt_repo is not None and not self.molopt_repo.exists():
            raise FileNotFoundError(f"MolOpt repository does not exist: {self.molopt_repo}")
        self._load_algorithm()
        self._load_oracle()

    def availability_error(self) -> str | None:
        """Return an import error message when this optional algorithm is unavailable."""
        try:
            with quiet_external_output(not self.verbose):
                self._load_algorithm()
        except Exception as exc:
            return str(exc)
        return None

    def optimize_request(self, request: OptimizationRequest) -> list[str]:
        if self.output_root is not None:
            self.output_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=self.output_root) as tmpdir:
            work_dir = Path(tmpdir)
            input_smi = self._write_request_smi(work_dir, request)
            output_dir = work_dir / "results"
            with quiet_external_output(not self.verbose):
                optimizer_cls = self._load_algorithm()
                oracle = self._load_oracle(request)

                with molopt_repo_context(self.molopt_repo):
                    optimizer = self._make_optimizer(optimizer_cls, input_smi, output_dir)
                    optimize_kwargs = {
                        "oracle": oracle,
                        "patience": self.patience,
                        "seed": self.seed,
                    }
                    config = self._algorithm_config()
                    if config is not None:
                        optimize_kwargs["config"] = str(config)
                    optimizer.optimize(**optimize_kwargs)

            return self._read_top_smiles(output_dir, request.num_outputs, oracle)

    def evaluate(self, smiles: str, adjustment: str):
        request = OptimizationRequest(
            input_id="evaluation",
            smiles=smiles,
            adjustment=adjustment,
            num_outputs=1,
        )
        oracle = self._load_oracle(request)
        cached_value = self._score_cache.get((oracle.name, smiles))
        if cached_value is not None:
            return oracle.name, cached_value
        return oracle.name, oracle.raw_value(smiles)

    def _write_request_smi(self, work_dir: Path, request: OptimizationRequest) -> Path:
        if self.source_smi_file is not None:
            return self.source_smi_file

        input_smi = work_dir / "input.smi"
        input_smi.write_text(f"{request.smiles}\n")
        return input_smi

    def _load_algorithm(self):
        if self.algorithm_cls is not None:
            return self.algorithm_cls

        module_name, class_name = MOL_OPT_ALGORITHMS[self.algorithm]
        with molopt_repo_context(self.molopt_repo):
            module = importlib.import_module(module_name)
            return getattr(module, class_name)

    def _load_oracle(self, request: OptimizationRequest | None = None):
        registry = self._load_evaluator_registry()
        adjustment = request.adjustment if request is not None else self.oracle
        objective = self.objective or parse_optimization_objective(
            adjustment,
            properties=registry,
        )
        if objective is None:
            objective = parse_optimization_objective(
                adjustment,
                properties=registry,
                default_property=self.oracle,
            )
        if objective.property not in registry:
            available = ", ".join(sorted(registry))
            raise ValueError(
                f"Oracle '{objective.property}' is not registered. Available: {available}"
            )

        return NamedOracle(
            name=objective.property,
            evaluator=registry[objective.property],
            objective=objective,
        )

    def _load_evaluator_registry(self) -> dict[str, object]:
        """Load oracle evaluators from explicit registry (no magic imports)."""
        registry = dict(self.oracle_registry)
        if self.oracle_fn is not None:
            registry[self.oracle] = self.oracle_fn

        # Check if oracle is available
        if self.oracle not in registry:
            available = ", ".join(sorted(registry.keys())) if registry else "(none)"
            raise ImportError(
                f"Oracle '{self.oracle}' not in registry. "
                f"Available oracles: {available}\n"
                f"Pass oracle_registry or oracle_fn to MolOptOptimizerTool to add it."
            )

        return registry

    def _make_optimizer(self, optimizer_cls, input_smi: Path | None, output_dir: Path):
        optimizer_smi_file = MOL_OPT_SMI_FILE_OVERRIDES.get(self.algorithm, input_smi)
        kwargs = {
            "smi_file": None if optimizer_smi_file is None else str(optimizer_smi_file),
            "n_jobs": self.n_jobs,
            "max_oracle_calls": self.max_oracle_calls,
            "freq_log": self.freq_log,
            "output_dir": str(output_dir),
            "log_results": self.verbose,
        }
        try:
            return optimizer_cls(**kwargs)
        except TypeError:
            kwargs.pop("n_jobs")
            return optimizer_cls(**kwargs)

    def _algorithm_config(self) -> Path | None:
        config = MOL_OPT_ALGORITHM_CONFIGS.get(self.algorithm)
        if config is None:
            return None
        if self.molopt_repo is None:
            return Path(config)
        return self.molopt_repo / config

    def _read_top_smiles(
        self,
        output_dir: Path,
        num_outputs: int,
        oracle: NamedOracle,
    ) -> list[str]:
        result_files = sorted(output_dir.glob("results_*.yaml"))
        if not result_files:
            raise FileNotFoundError(f"MolOpt did not write a results YAML in {output_dir}")

        with result_files[-1].open() as handle:
            mol_buffer = yaml.safe_load(handle) or {}

        ranked = sorted(
            mol_buffer.items(),
            key=lambda item: (item[1][0], -item[1][1]),
            reverse=True,
        )
        top_items = ranked[:num_outputs]
        for smiles, score_and_order in top_items:
            raw_value = self._raw_value_from_molopt_score(
                oracle.objective,
                float(score_and_order[0]),
            )
            if raw_value is not None:
                self._score_cache[(oracle.name, smiles)] = raw_value
        return [smiles for smiles, _score_and_order in top_items]

    def _raw_value_from_molopt_score(
        self,
        objective: OptimizationObjective,
        score: float,
    ) -> float | None:
        if objective.direction == "target_range":
            return None
        if objective.direction == "increase":
            return score
        if objective.property == "sascore":
            return 10.0 - score
        if objective.property in TOXICITY_ORACLES:
            return 1.0 - score
        return None


def build_molopt_tools(
    molopt_repo: str | os.PathLike | None,
    algorithms: list[str] | None = None,
    oracle: str = "qed",
    evaluator_root: str | os.PathLike | None = None,
    oracle_registry: dict[str, object] | None = None,
    skip_unavailable: bool = False,
    **kwargs,
) -> list[MolOptOptimizerTool]:
    """Build one lead-optimizer tool per requested MolOpt algorithm."""
    if oracle_registry is None and "oracle_fn" not in kwargs:
        from ..oracles.registry import build_oracle_registry

        try:
            oracle_registry = build_oracle_registry(
                molopt_repo=molopt_repo,
                evaluator_root=evaluator_root,
            )
        except ImportError:
            if skip_unavailable:
                return []
            raise
        except ValueError as exc:
            if skip_unavailable and "No oracles were registered" in str(exc):
                return []
            raise

    selected = algorithms or BENCHMARKED_MOL_OPT_ALGORITHMS
    tools = []
    for algorithm in selected:
        tool = MolOptOptimizerTool(
            molopt_repo=molopt_repo,
            algorithm=algorithm,
            oracle=oracle,
            oracle_registry=oracle_registry,
            **kwargs,
        )
        if skip_unavailable and tool.availability_error():
            continue
        tools.append(tool)
    return tools
