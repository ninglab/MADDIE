from pathlib import Path
from collections.abc import Callable
from dataclasses import replace

from ..core import normalize_smiles_csv, write_optimized_csv
from ..utils.logging import get_logger
from ..schemas import OptimizationBatchResult, OptimizationRequest, OptimizationResult
from ..tools import FakeOptimizerTool, OptimizationTool
from ..validation import canonicalize_smiles, normalize_smiles


logger = get_logger()


class InsufficientValidOutputs(RuntimeError):
    """Raised internally when retry exhaustion still produced partial outputs."""

    def __init__(
        self,
        tool_name: str,
        input_id: str,
        accepted: list[str],
        attempt_limit: int,
        requested: int,
    ):
        self.tool_name = tool_name
        self.input_id = input_id
        self.accepted = accepted
        self.attempt_limit = attempt_limit
        self.requested = requested
        super().__init__(
            f"{tool_name} stopped after {attempt_limit} attempts for {input_id} "
            f"and returned {len(accepted)}/{requested} valid unique molecules."
        )


class DirectOptimizer:
    """Deterministic CSV optimizer that does not require an LLM planner."""

    def __init__(
        self,
        tool: OptimizationTool | None = None,
        num_outputs: int = 5,
        max_attempts: int = 1,
        smiles_validator: Callable[[str], str | None] = canonicalize_smiles,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")

        self.tool = tool or FakeOptimizerTool()
        self.num_outputs = num_outputs
        self.max_attempts = max_attempts
        self.smiles_validator = smiles_validator

    def _collect_valid_outputs(self, request: OptimizationRequest) -> list[str]:
        accepted = []
        seen = set()
        invalid_count = 0
        duplicate_count = 0
        attempt_limit = self.max_attempts

        for attempt in range(1, attempt_limit + 1):
            remaining = request.num_outputs - len(accepted)
            if remaining == 0:
                return accepted

            adjustment = request.adjustment
            if attempt > 1 and self.tool.include_retry_instruction:
                adjustment = self._retry_adjustment(
                    request.adjustment,
                    remaining,
                    accepted,
                    invalid_count,
                    duplicate_count,
                )

            retry_request = replace(
                request,
                adjustment=adjustment,
                num_outputs=max(remaining, self.tool.min_batch_size),
                attempt=attempt,
            )
            logger.info(
                "Starting optimization attempt %s for %s with %s; requesting %s candidate(s).",
                attempt,
                request.input_id,
                self.tool.name,
                retry_request.num_outputs,
            )
            candidates = self.tool.optimize_request(retry_request)
            logger.info(
                "Optimization attempt %s for %s complete; validating %s candidate(s).",
                attempt,
                request.input_id,
                len(candidates),
            )

            for candidate in candidates:
                if isinstance(candidate, str):
                    candidate = normalize_smiles(candidate)
                canonical_smiles = self.smiles_validator(candidate)
                if canonical_smiles is None:
                    invalid_count += 1
                    continue
                if canonical_smiles in seen:
                    duplicate_count += 1
                    continue

                seen.add(canonical_smiles)
                accepted.append(candidate)
                if len(accepted) == request.num_outputs:
                    logger.info(
                        "Validation complete for %s: accepted %s/%s unique molecule(s).",
                        request.input_id,
                        len(accepted),
                        request.num_outputs,
                    )
                    return accepted

            logger.info(
                "Validation progress for %s: accepted %s/%s unique molecule(s); "
                "invalid=%s, duplicates=%s.",
                request.input_id,
                len(accepted),
                request.num_outputs,
                invalid_count,
                duplicate_count,
            )

        raise InsufficientValidOutputs(
            tool_name=self.tool.name,
            input_id=request.input_id,
            accepted=accepted,
            attempt_limit=attempt_limit,
            requested=request.num_outputs,
        )

    @staticmethod
    def _retry_adjustment(
        adjustment: str,
        remaining: int,
        accepted: list[str],
        invalid_count: int,
        duplicate_count: int,
    ) -> str:
        accepted_text = ", ".join(accepted) or "none"
        return (
            f"{adjustment}\n\n"
            f"Retry instruction: Return {remaining} additional valid, unique SMILES. "
            f"Do not repeat these accepted molecules: {accepted_text}. "
            f"Previous attempts contained {invalid_count} invalid and "
            f"{duplicate_count} duplicate candidates."
        )

    def optimize_csv(
        self,
        input_csv: str | Path,
        output_csv: str | Path,
        adjustment: str,
        num_outputs: int | None = None,
    ) -> OptimizationBatchResult:
        input_path = Path(input_csv)
        output_path = Path(output_csv)
        output_count = num_outputs or self.num_outputs

        # CSV rows become one OptimizationRequest per input molecule.
        molecules, input_warnings, skipped_input_rows = normalize_smiles_csv(
            input_path
        )
        logger.info(
            "Loaded %s molecule(s) from %s; skipped %s row(s).",
            len(molecules),
            input_path,
            skipped_input_rows,
        )
        for warning in input_warnings:
            logger.warning("Input cleanup: %s", warning)

        requests = [
            OptimizationRequest(
                input_id=molecule["input_id"],
                smiles=molecule["smiles"],
                adjustment=adjustment,
                num_outputs=output_count,
            )
            for molecule in molecules
        ]

        # Each request can produce multiple optimized SMILES, so the output CSV
        # may have more rows than the input CSV.
        result_rows = []
        failed_input_ids = []
        partial = False
        for request in requests:
            try:
                optimized = self._collect_valid_outputs(request)
            except InsufficientValidOutputs as exc:
                partial = True
                optimized = exc.accepted
                if not optimized:
                    failed_input_ids.append(exc.input_id)
                warning = (
                    f"{exc.tool_name} stopped after {exc.attempt_limit} attempts "
                    f"for {exc.input_id} and returned {len(optimized)}/"
                    f"{exc.requested} valid unique molecules."
                )
                input_warnings.append(warning)
                logger.warning(warning)
            except Exception as exc:
                partial = True
                failed_input_ids.append(request.input_id)
                warning = f"{self.tool.name} failed for {request.input_id}: {exc}"
                input_warnings.append(warning)
                logger.warning(warning)
                write_optimized_csv(output_path, result_rows)
                continue

            try:
                input_evaluation = self.tool.evaluate(
                    request.smiles,
                    request.adjustment,
                )
            except Exception as exc:
                partial = True
                input_evaluation = None
                warning = (
                    f"Could not score input {request.input_id} with "
                    f"{self.tool.name}: {exc}"
                )
                input_warnings.append(warning)
                logger.warning(warning)
            for output_id, optimized_smiles in enumerate(optimized, start=1):
                try:
                    optimized_evaluation = self.tool.evaluate(
                        optimized_smiles,
                        request.adjustment,
                    )
                except Exception as exc:
                    partial = True
                    optimized_evaluation = None
                    warning = (
                        f"Could not score output {output_id} for {request.input_id} "
                        f"with {self.tool.name}: {exc}"
                    )
                    input_warnings.append(warning)
                    logger.warning(warning)
                result_rows.append(
                    OptimizationResult(
                        input_id=request.input_id,
                        input_smiles=request.smiles,
                        optimized_smiles=optimized_smiles,
                        output_id=output_id,
                        adjustment=request.adjustment,
                        tool=self.tool.name,
                        property=(
                            optimized_evaluation[0]
                            if optimized_evaluation
                            else None
                        ),
                        input_property_value=(
                            input_evaluation[1]
                            if input_evaluation
                            else None
                        ),
                        optimized_property_value=(
                            optimized_evaluation[1]
                            if optimized_evaluation
                            else None
                        ),
                    )
                )
            write_optimized_csv(output_path, result_rows)

        write_optimized_csv(output_path, result_rows)
        logger.info("Wrote %s optimization row(s) to %s.", len(result_rows), output_path)

        return OptimizationBatchResult(
            input_csv=str(input_path),
            output_csv=str(output_path),
            molecule_count=len(requests),
            row_count=len(result_rows),
            tool=self.tool.name,
            results=result_rows,
            input_warnings=input_warnings,
            skipped_input_rows=skipped_input_rows,
            failed_input_ids=failed_input_ids,
            partial=partial,
        )
