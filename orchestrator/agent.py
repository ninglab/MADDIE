from abc import ABC
from datetime import datetime
from typing import (
    Type,
    Dict,
    List,
    Any,
    NoReturn,
    Literal,
    Callable,
    Tuple,
    Iterator,
    Protocol,
    Optional,
)
from rich.console import Console
from rich.markup import escape
from pathlib import Path
import re
import json
import os
import sys
import copy
import time
from watchdog.utils.dirsnapshot import DirectorySnapshot, DirectorySnapshotDiff

from orchestrator.memory import BaseMemory
from orchestrator.llm import BaseLLM, ClaudeLLM, OpenAILLM, Message, LLMResponse
from orchestrator.tools import SBDD, OptimizeQED, EvaluateQED, EvaluateSynthesisRoute
from orchestrator.worker import BaseWorker
from orchestrator.automaton import Automaton, DirectoryWatcherAutomaton
import dataclasses
from dataclasses import dataclass


# region logger
class Logger:
    """Lightweight console logger with timestamped, color-coded output.

    Wraps a Rich :class:`Console` to emit leveled log messages (info, error,
    debug) prefixed with an ``MM-DD HH:MM:SS [filename:line]`` tag, where the
    location points at the caller of the log method. Message text is escaped so
    that user-supplied content cannot be interpreted as Rich markup.
    """

    def __init__(self):
        self.console = Console()

    def _print(self, msg: str) -> None:
        ts = datetime.now().strftime("%m-%d %H:%M:%S")
        # Walk two frames up: _print -> info/error/debug -> actual caller
        frame = sys._getframe(2)
        # escape() so the brackets render literally instead of being parsed
        # as Rich markup tags (which would silently drop the location).
        loc = escape(f"[{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}]")
        self.console.print(f"[dim]{ts} {loc}[/dim] {msg}", highlight=False)

    def info(self, msg: str) -> None:
        """Log INFO-level message."""
        self._print(f"[cyan][INFO][/cyan] {escape(msg)}")

    def error(self, msg: str) -> None:
        """Log ERROR-level message."""
        self._print(f"[red][ERROR][/red] {escape(msg)}")

    def warning(self, msg: str) -> None:
        """Log WARNING-level message."""
        self._print(f"[yellow][WARNING][/yellow] {escape(msg)}")

    def debug(self, msg: str) -> None:
        """Log DEBUG-level message."""
        self._print(f"[orange1][DEBUG][/orange1] {escape(msg)}")

    def fail(self, msg: str, err_cls: Type[Exception]) -> NoReturn:
        """Log ERROR-level message, then raise err_cls to abort."""
        self.error(msg)
        raise err_cls(msg)


# endregion


# region provider
@dataclass(frozen=True)
class Provider:
    "Provider of LLM services (e.g. Anthropic, OpenAI, etc.)"

    models: set[str]  # List of model names that this provider supports
    cls: type  # BaseLLM class to instantiate for this provider
    env_var: str  # Environment variable name for the API key


CLAUDE_MODELS = {"claude-sonnet-4-6", "claude-haiku-4-5", "claude-opus-4-7"}
OPENAI_MODELS = {"gpt-5.4-mini"}

PROVIDERS = (
    Provider(CLAUDE_MODELS, ClaudeLLM, "ANTHROPIC_API_KEY"),
    Provider(OPENAI_MODELS, OpenAILLM, "OPENAI_API_KEY"),
)
# endregion


# region dataclass
@dataclass
class WorkerEntry:
    worker: BaseWorker
    profile: Path


@dataclass
class Action:
    tag: str  # closing tag to detect in LLM response, e.g. "</PROCESS>"
    description: str  # description of how to use the Action
    parse: Callable[
        [str], tuple[bool, tuple[Any, ...] | str]
    ]  # returns (True, payload_tuple) on success or (False, correction_str) on failure
    execute: (
        Callable[..., str] | None
    )  # returns output string; if terminal=True, that string is the final answer
    result_role: (
        str | Callable[..., str]
    )  # role for memory entry; callable receives *payload
    result_tags: List[str] | None
    terminal: bool = False  # when True, a non-None execute result ends the loop


@dataclass
class MessageBlock:
    role: str  # one of Orchestrator, User, System, {WorkerName}
    message: Message
    tags: List[str] | None = None


# endregion


# region summarizer & verifier
class Summarizer(Protocol):
    def summarize(self, messages: List[MessageBlock]) -> str: ...

class Verifier(Protocol):
    def verify_csv(self, user_prompt: str, path_to_csv: str) -> tuple[bool, str]: ...

class LLMSummarizer:
    SUMMARY_SYSTEM = (
        "You are MADDIE, the orchestrator of a multi-agent drug discovery system. Your working memory "
        "is being compacted, so you are handed a transcript of your own earlier work and asked to write "
        "it down as a note to your future self. Write in the first person ('I delegated...', 'I "
        "verified...') — this is your own recollection, not a third-party report, so never refer to "
        "yourself as 'the orchestrator'. Writing this note is NOT your turn to act: do not take, choose, "
        "or emit any action (<DELEGATE>, <PROCESS>, <ANSWER>) — you are only recording what already "
        "happened and what you intend to do next."
    )

    SUMMARY_INSTRUCTIONS = (
        "Your task is to create a detailed, first-person summary of the orchestration transcript above, "
        "paying close attention to the user's explicit goal and the delegations you have made and their "
        "results. Write it as your own recollection — use 'I' throughout (e.g. 'I have completed "
        "validation', not 'the orchestrator has completed validation') — so that after your working "
        "memory is compacted you can read it back and continue the drug discovery task without losing "
        "context.\n\n"

        "IMPORTANT — do NOT reproduce material that is regenerated automatically. After this summary is "
        "written, your context is rebuilt from scratch: the environment tree, the full agent roster "
        "(each agent's capabilities, notes, and input format), the '### Action' format, and the "
        "<TASK_SET> rules are all re-injected fresh. Repeating them wastes space. Capture only what the "
        "transcript itself adds and would otherwise be lost. Do not list generic background concepts "
        "(assays, scoring functions, etc.) — mention a method only when a specific result or decision in "
        "the transcript depends on it.\n\n"

        "Writing this note is NOT your turn to act. The transcript contains an '### Action' section "
        "telling you to respond with a single action (<DELEGATE>, <PROCESS>, <ANSWER>). Ignore it: do not "
        "emit any filled-in action block. Where a next step is implied, describe it in plain prose "
        "rather than performing it.\n\n"

        "Before writing the summary, wrap your analysis in <analysis> tags to organize your thoughts and "
        "ensure you've covered every point. In your analysis:\n"
        "1. Chronologically analyze each message and section of the transcript. For each, identify:\n"
        "   - The user's explicit requests and intent\n"
        "   - How you addressed them (which agent you delegated to, with what input, and why)\n"
        "   - Key decisions and assumptions\n"
        "   - Concrete results: molecule IDs, SMILES/structures, metrics/scores, and file paths\n"
        "2. Double-check for technical accuracy and completeness.\n\n"

        "Your summary (written in the first person) should include these sections:\n"
        "1. Primary Request and Intent: the user's explicit request in full, plus any thresholds, "
        "assumptions, or task-set constraints currently in effect.\n"
        "2. Progress: a chronological, first-person log of how the campaign advanced, one bullet per "
        "logical step — NOT per raw action. Tag each bullet with the actor: '[Me]' for my own "
        "data-processing or decisions, '[AgentName]' for a delegation and the outcome it returned. "
        "Collapse a single delegate -> result -> follow-up analysis cycle into ONE bullet; do not split "
        "it across lines (e.g. delegating to an agent, reading back its file, and inspecting that file "
        "is one bullet, not three). Mark which sub-goals are done, in progress, or still pending.\n"
        "3. Artifacts: the result files produced so far, keyed by file path. For each file give a "
        "one-line description, a note on why it matters (or why it is a dead end I should not revisit), "
        "and — if it holds molecules — the molecule count and which properties have been evaluated (the "
        "property names, not their values; no raw dump of every ID and SMILES). Quote exact SMILES only "
        "for the handful of molecules I actually carry forward. Favor the most recent files and be "
        "precise about paths.\n"
        "4. Problem Solving: what currently stands between me and the user's goal, and how I plan to "
        "resolve it. This is broader than errors — it includes scientific blockers (e.g. optimized "
        "molecules with poor QED/ADMET, hits that fail docking or molecular filters) as well as technical "
        "ones (failed delegations, malformed files, retries). For each, note the cause and the intended "
        "fix.\n"
        "5. Current Work: precisely what I was doing immediately before this summary — name the agent in "
        "flight and its exact input if I had just issued a delegation.\n"
        "6. Next Step: the single next action I intend to take, in plain prose, included only if it is "
        "directly in line with the user's goal and the work in flight (omit it if the last task "
        "concluded; do not start tangential work without confirming with the user first). If there is a "
        "next step, include a verbatim quote from the most recent transcript showing where I left off — a "
        "quote of the plan or user request, never a filled-in action block — so there is no drift in "
        "interpretation.\n\n"

        "Here's how your output should be structured:\n"
        "<example>\n"
        "<analysis>\n"
        "[Your thought process, ensuring all points are covered thoroughly and accurately]\n"
        "</analysis>\n"
        "<summary>\n"
        "1. Primary Request and Intent:\n"
        "   [Detailed description]\n"
        "2. Progress:\n"
        "   - [Me] [what I did myself, e.g. prepared or validated a file]\n"
        "   - [AgentName] [what I delegated and the outcome it returned, including my read of the results]\n"
        "   - [Me] [the next thing I did]\n"
        "3. Artifacts:\n"
        "   - [File path 1]\n"
        "      - [One short description of the file]\n"
        "      - [Why this file is important, or why it is a dead end]\n"
        "      - [If it holds molecules: number of molecules, and which properties have been evaluated (names, not values)]\n"
        "      - [Exact SMILES of molecules I carry forward, if any]\n"
        "   - [File path 2]\n"
        "      - [...]\n"
        "4. Problem Solving:\n"
        "   [Description of solved problems and ongoing approaches to solve, if any]\n"
        "5. Current Work:\n"
        "   [Precise description of current work]\n"
        "6. Next Step:\n"
        "   [Next step, with a verbatim quote of where I left off]\n"
        "</summary>\n"
        "</example>\n\n"

        "Provide your summary now, following this structure with precision and thoroughness."
    )

    def __init__(self, llm: BaseLLM, logger: Optional[Logger] = None):
        self.llm = llm
        self.logger = logger or Logger()

    def summarize(self, messages: List[MessageBlock]) -> str:
        transcript = "\n\n".join(f"[{m.role.upper()}]\n{m.message.content}" for m in messages)
        prompt = Message(
            role="user",
            content=(
                f"### TRANSCRIPT\n{transcript}\n\n"
                f"### INSTRUCTIONS\n{self.SUMMARY_INSTRUCTIONS}"
            ),
        )
        print(prompt.content)

        # swap in the minimal guardrail system prompt, then restore (the llm may
        # be shared with the orchestrator)
        prev_system = self.llm.system
        self.llm.update_system(self.SUMMARY_SYSTEM)
        try:
            response = self.llm.get_response([prompt])
        finally:
            self.llm.update_system(prev_system)
        return response.content


class LLMVerifier:
    """Final-answer gate: decide whether a FINAL.csv actually answers the user's
    request. Implements the :class:`Verifier` protocol, mirroring
    :class:`LLMSummarizer`'s system-swap pattern."""

    MAX_ROWS = 200   # cap the CSV excerpt sent to the LLM
    MAX_RETRY = 5    # re-prompts allowed when the response has no parseable <VERDICT>

    VERIFY_SYSTEM = (
        "You are MADDIE's final-answer verifier. You are given the user's request and the contents "
        "of a final results file (FINAL.csv); decide only whether the molecules in the file satisfy "
        "that request. You do not act, plan, or generate molecules — you only judge the file."
    )

    VERIFY_INSTRUCTIONS = (
        "Judge whether the molecules in FINAL.csv answer the user's request: the right target, the "
        "required properties/constraints, and the requested count when one is stated. Be lenient about "
        "formatting and column naming, but strict about substance — an empty, irrelevant, or "
        "constraint-violating file must fail.\n\n"
        "If user requires synthesis route, make sure the value in the synthesis route is actually a synthesis route, not just any string.\n"
        "End your reply with exactly these two tagged lines and nothing after them:\n"
        "<VERDICT>PASS</VERDICT> or <VERDICT>FAIL</VERDICT>\n"
        "<REASON>one sentence explaining the verdict</REASON>"
    )

    _VERDICT_RE = re.compile(r"<VERDICT>\s*(PASS|FAIL)\s*</VERDICT>", re.IGNORECASE)
    _REASON_RE = re.compile(r"<REASON>\s*(.*?)\s*</REASON>", re.IGNORECASE | re.DOTALL)

    def __init__(self, llm: BaseLLM, logger: Optional[Logger] = None):
        self.llm = llm
        self.logger = logger or Logger()

    def verify_csv(self, user_prompt: str, path_to_csv: str) -> tuple[bool, str]:
        # read the file (bail early on missing/empty/unreadable)
        try:
            with open(path_to_csv, "r") as fh:
                lines = fh.read().splitlines()
        except OSError as e:
            self.logger.warning(f"LLMVerifier could not read '{path_to_csv}': {e}")
            return False, "FINAL.csv is missing or empty"
        if not any(line.strip() for line in lines):
            return False, "FINAL.csv is missing or empty"

        # reject SMILES-only files (must have evaluation columns, not just smiles)
        header_cols = [c.strip() for c in lines[0].split(",")]
        if len(header_cols) <= 1:
            return False, "FINAL.csv contains only one column — evaluation columns are missing; re-run the Evaluator and write FINAL.csv from its output"

        _required_eval = {"docking_kcal_mol", "admet_hERG", "admet_ClinTox", "admet_Mutagenicity"}
        if not any(c in header_cols for c in _required_eval):
            return False, "FINAL.csv is missing evaluation columns (docking_kcal_mol, admet_*) — write FINAL.csv from property_pool.csv or synthesizable_pool.csv, not from a SMILES-only source"

        total_rows = max(len(lines) - 1, 0)  # excluding header
        excerpt = "\n".join(lines[: self.MAX_ROWS + 1])
        if len(lines) > self.MAX_ROWS + 1:
            excerpt += f"\n... ({total_rows} data rows total; showing first {self.MAX_ROWS})"

        messages = [
            Message(
                role="user",
                content=(
                    f"### USER REQUEST\n{user_prompt}\n\n"
                    f"### FINAL.csv ({total_rows} data rows)\n{excerpt}\n\n"
                    f"### INSTRUCTIONS\n{self.VERIFY_INSTRUCTIONS}"
                ),
            )
        ]

        # swap in the minimal guardrail system prompt, then restore (the llm may
        # be shared with the orchestrator)
        prev_system = self.llm.system
        self.llm.update_system(self.VERIFY_SYSTEM)
        try:
            # re-prompt up to MAX_RETRY times until the response carries a <VERDICT>
            for attempt in range(self.MAX_RETRY + 1):
                response = self.llm.get_response(messages)
                content = response.content or ""
                verdict_match = self._VERDICT_RE.search(content)
                if verdict_match is not None:
                    reason_match = self._REASON_RE.search(content)
                    reason = reason_match.group(1).strip() if reason_match else content.strip()
                    passed = verdict_match.group(1).upper() == "PASS"
                    return passed, reason

                # no verdict — feed the bad reply back and ask for the exact format
                self.logger.warning(
                    f"LLMVerifier response had no parseable <VERDICT> "
                    f"(attempt {attempt + 1}/{self.MAX_RETRY + 1}); re-prompting."
                )
                messages.append(Message(role="assistant", content=content))
                messages.append(
                    Message(
                        role="user",
                        content=(
                            "Your reply did not contain a valid verdict. Respond again, ending with "
                            "exactly:\n<VERDICT>PASS</VERDICT> or <VERDICT>FAIL</VERDICT>\n"
                            "<REASON>one sentence explaining the verdict</REASON>"
                        ),
                    )
                )
        finally:
            self.llm.update_system(prev_system)

        # retries exhausted: fail-open so a stubborn parser issue can't loop the run forever
        self.logger.warning(
            f"LLMVerifier got no parseable <VERDICT> after {self.MAX_RETRY + 1} attempts; "
            "passing by default."
        )
        return True, "verifier returned no parseable verdict after retries; accepted by default"

# endregion

# region system prompt
ORCHESTRATOR_SYSTEM_PROMPT = """
You are MADDIE, the orchestrator of a multi-agent drug discovery system. You PLAN, DELEGATE, and MONITOR. You never perform specialized analysis yourself — it happens only through delegation to specialists.

Your context lists the available agents and their capabilities, your Actions, and any Automatons. Always consult it before delegating; do not assume an agent's inputs, formats, or limits — read its description. Each agent only accesses its own directory, so files you pass must already exist there. A shared database holds all molecules, targets, and data.

Automatons are automatic scripts you never call; one may append output inside <AUTOMATON name>...</AUTOMATON>. Use it when it appears.

Follow any rules, constraints, or warnings inside <TASK_SET>...</TASK_SET>.

# Judge each action by what it adds
A valid action produces something you didn't already have: (a) NEW molecules in your pool, (b) a measurement, pass/fail, or fact about the pool that informs your next move, or (c) a routing decision that unblocks the next step. Judge an action by what it adds, not by its type.

Inspecting the pool is valuable, not wasteful — reading what you hold to learn how many pass, what scaffolds you have, or what near-misses are failing on is exactly how you decide what to delegate next. Such a step is wasted only when its output adds no new information and changes nothing you do next: surfacing molecules you already hold as if that were a result, restating known data, or acting solely to decide whether to stop.

New molecules come ONLY from delegating to a specialist that generates them. Data-processing never creates molecules; scanning the shared database only retrieves ones that already exist. To generate, you must delegate.

# Workflow
1. Restate the goal.
2. Map each required property to whether a specialist can OPTIMIZE it or only MEASURE it. Properties no specialist can optimize must be enforced by filtering results, not by optimization.
3. Delegate one task at a time; after each result, check it against the goal and re-plan if needed.
4. Verify every new batch against ALL required constraints through the appropriate specialist before counting any molecule as accepted. Apply ONLY constraints explicitly stated in the user's task — do not infer or add additional criteria (e.g. structural filters, drug-likeness rules) unless the user asked for them.
5. Finish by reporting the accepted-pool size and its file path.

# "As many as possible" / use-all-your-actions goals
"Use all your actions" means spend the whole budget growing your pool of verified qualifying molecules — through generating, verifying, AND the analysis that steers them — not taking filler actions until the counter runs out.

Each cycle involves three kinds of work, and all three matter:
- ANALYZE (data-processing): inspect the current pool to learn something you don't already know — how many molecules pass, what scaffolds you hold, whether diversity is saturating, what your near-misses are failing on. This is what tells you what to do next.
- GENERATE (delegation only): the sole source of new molecules.
    - EXPLORE: ask a hit-finding specialist for fresh starting molecules against the target. Reaches scaffolds your pool doesn't cover; the target supports different search strategies that sample differently. Reusable — not a one-time seed.
    - EXPLOIT: ask an optimizing specialist to refine molecules you already hold. Expands around scaffolds already known to work.
- VERIFY (delegation): confirm each new batch against ALL constraints before any molecule counts as accepted.

Run the cycle until the budget is spent:
  a. Analyze the pool to decide the next move.
  b. Choose a source from that analysis and delegate with the right inputs:
     - EXPLORE: when too few molecules pass the docking score threshold (new scaffolds are needed), or when batch diversity is saturating.
     - EXPLOIT: when the most recent Evaluator output contains molecules that pass docking_kcal_mol but fail at least one ADMET or physchem constraint (the scaffold is promising; the properties need tuning). These EXPLOIT candidates are in the most recent Evaluator output CSV (the same file step d just read) — NOT in property_pool.csv. They were excluded from property_pool because they failed ≥1 non-docking constraint; LeadOptimizer exists to fix those failures. If step d printed EXPLOIT count > 0, run LeadOptimizer on those molecules. Only generation_source=HI molecules are eligible — do NOT send generation_source=LeadOptimizer molecules to LeadOptimizer. Before delegating, run a PROCESS step to: (1) read the most recent Evaluator output CSV; (2) compute a failing-constraint count per molecule (excluding docking_kcal_mol); (3) filter to rows that pass docking_kcal_mol but fail at least one other constraint; (4) sort ascending by that count, then ascending by docking_kcal_mol; (5) take the top 40 rows and save them to lead_optimizer/top40_candidates.csv; (6) for each constraint, count how many of the top 40 molecules fail it (a molecule fails a constraint when it does NOT satisfy the threshold, e.g. fails `admet_hERG < 0.96` means its value is `>= 0.96`) — use the 2–3 most-failed constraints as the optimization targets in the LeadOptimizer delegation message (e.g. "decrease DILI, decrease hERG"). Do NOT include constraints that none of the top 40 molecules fail. Also do NOT include any `filter_*` columns (e.g. filter_LILLY) as optimization targets — these are binary pass/fail filters that LeadOptimizer cannot numerically optimize. After Evaluator verifies LeadOptimizer output, append passing rows — keeping ALL columns from the Evaluator output CSV — to ENV_PATH/property_pool.csv with generation_source=`"LeadOptimizer"`, deduplicating by smiles. Do NOT run step d for this Evaluator call — use this append rule instead.
  c. Verify the new batch via Evaluator.
  d. Immediately after every EXPLORE → Evaluator run (i.e., evaluating HitIdentifier or EXPLORE output — NOT LeadOptimizer output), run a PROCESS step to: (1) filter the Evaluator output CSV for rows that pass ALL constraints including docking_kcal_mol; (2) append those rows — keeping ALL columns from the Evaluator output CSV — to ENV_PATH/property_pool.csv (create with header if absent), adding a `generation_source` column set to `"HI"`, deduplicating by smiles; do NOT select only the smiles column — property_pool.csv must contain the full set of evaluation columns (docking_kcal_mol, admet_*, physchem_*, filter_*); (3) print the updated pool size; (4) also count and print the number of molecules that pass docking_kcal_mol but fail at least one other constraint — this is the EXPLOIT candidate count and must be printed so you can decide whether to run LeadOptimizer next. Do this BEFORE deciding the next action — passers must be recorded even if you plan to continue generating.
Then report the pool size and file path.

Rule: after an EXPLOIT → VERIFY cycle, the next GENERATE must be EXPLORE. Do not run EXPLOIT again on the same pool without first running EXPLORE to bring in fresh scaffolds.

Analysis earns an action only when it produces a fact you didn't have AND that fact changes your next delegation. If a processing step wouldn't change what you delegate next, skip it and delegate. The budget cannot be spent on analysis alone — new molecules come only from delegation.

# Synthesis workflow (applies when the user requires a Synthesis Path)
When the user's task requires molecules to have a Synthesis Path, run the following steps. property_pool.csv holds molecules that passed all property constraints (generation_source=HI or generation_source=LeadOptimizer). synthesizable_pool.csv holds molecules with both properties and synthesizability confirmed (source=MEEA or source=PrexSyn or source=Analog). Molecules in synthesizable_pool.csv must never be sent to LeadOptimizer.

**Trigger**: Send property_pool.csv to Synthesizer when ANY condition is met:
1. `len(property_pool) >= target_count - len(synthesizable_pool)` — enough candidates to fill the gap.
2. `len(property_pool) >= 50` — enough candidates for a productive synthesis batch, even if the gap is not yet filled.
3. Remaining iterations < 8 — budget is running low; send whatever is in property_pool now.
Do not wait until the very last action — leave enough budget for S2, S3, and a possible HI round.

**Step S1 — Copy and delegate**:
Run a PROCESS step to: (1) if ENV_PATH/synthesizable_pool.csv exists, filter out from ENV_PATH/property_pool.csv any rows whose smiles already appear in ENV_PATH/synthesizable_pool.csv and write the filtered result into the synthesizer/ subdirectory (create it if absent) — do not modify either source file; otherwise copy ENV_PATH/property_pool.csv into the synthesizer/ subdirectory as-is. Then DELEGATE to Synthesizer with the absolute path of that file. The Synthesizer writes `{input_stem}_meea.csv` (e.g. `property_pool_meea.csv`) into the synthesizer/ subdirectory.
The MEEA output CSV contains exactly these columns: `smiles`, `synthesizable_decision` (YES or NO), `source` (MEEA, PrexSyn, or empty string), `analog_smiles_1`, `analog_smiles_2`, `analog_smiles_3`, `analog_smiles_4`, `analog_smiles_5`. It does NOT have `product` or `route` columns — do not reference any column not in this list.

**Step S2 — YES stream**:
Run a PROCESS step to read synthesizer/property_pool_meea.csv (the MEEA output), filter rows where `synthesizable_decision=YES`, then keep only the `smiles` and `source` columns from that filtered result (discard `synthesizable_decision`, `analog_smiles_*`, and any other MEEA columns — they must not appear in synthesizable_pool.csv). Join those two columns with ENV_PATH/property_pool.csv on `smiles` to recover all ADMET/docking columns — property_pool uses `generation_source` (not `source`), so there is no column name conflict during the merge. Append the merged result to ENV_PATH/synthesizable_pool.csv (create with header if absent, deduplicate by smiles).

**Step S3 — Analog stream**:
From synthesizer/property_pool_meea.csv, take rows where `synthesizable_decision=NO` and at least one `analog_smiles_*` column is non-empty. Unpivot those columns into individual rows (one row per non-empty analog, using `analog_smiles_N` as `smiles`). Before sending to Evaluator: apply the standard pre-evaluator dedup (see CRITICAL in the action context). Write the remaining analogs to an absolute path inside the evaluator/ subdirectory using the standard XXX_ prefix (pick the next available number, e.g. evaluator/002_analogs_to_evaluate.csv), then pass that absolute path to Evaluator. Do NOT run step d for this Evaluator call. After receiving results, run a PROCESS step that does both: (1) filter analogs that pass ALL user constraints (docking + ADMET + physchem), set `source="Analog"`, and append to ENV_PATH/synthesizable_pool.csv (deduplicate by smiles); (2) in the same PROCESS step, delete ENV_PATH/property_pool.csv — this must happen regardless of how many analogs passed.

**Step S4 — Clear and check**:
Run a PROCESS step to delete ENV_PATH/property_pool.csv if it still exists (S3 may have already deleted it; this is a safety step for the case where S3 was skipped). Then check the size of ENV_PATH/synthesizable_pool.csv:
- If `len(synthesizable_pool) >= target_count`: proceed to write FINAL.csv.
- If `len(synthesizable_pool) < target_count`: do NOT write FINAL.csv. Go back to HI to generate more molecules, rebuild ENV_PATH/property_pool.csv, and repeat from the Trigger check. Writing FINAL.csv with fewer than target_count molecules is forbidden at this step.
- If the iteration budget is exhausted: write FINAL.csv from whatever is in ENV_PATH/synthesizable_pool.csv.

**FINAL.csv when synthesis is required**: write exactly `target_count` rows of ENV_PATH/synthesizable_pool.csv (not ENV_PATH/property_pool.csv) using `.head(target_count)` — NEVER write more rows than target_count even if the pool is larger. Apply all user constraints as a final filter. Include the `source` column alongside all ADMET/docking columns. Deduplicate by smiles.

# Hints
- If HitIdentifier returns an error or zero molecules, do NOT write SMILES directly in your response. Retry HitIdentifier with different parameters instead.
- If HitIdentifier fails with a ligand-related error (e.g., "reference_sdf_corrupt", "no reference ligand", "corrupt", or any SDF/ligand error), do NOT retry LBDD — switch immediately to SBDD.
- Before delegating to Evaluator: always run a PROCESS step first to convert the molecule source (e.g., HitIdentifier's successful_molecules.json) into a CSV file with a `smiles` column. Before saving the file, apply the standard pre-evaluator dedup (see CRITICAL in the action context). Save the deduplicated result to an absolute path inside the evaluator/ directory using the standard XXX_ prefix (pick the next available number, e.g. evaluator/000_hit_identifier.csv, evaluator/001_lead_optimizer.csv, evaluator/002_analogs_to_evaluate.csv). Then pass that absolute path to Evaluator. Never pass a JSON file, a directory path, or a relative path to Evaluator.
- Don't hallucinate file contents — verify.
- Be concise with the user and when delegating.
- Carry forward structures/IDs and relevant findings, not raw outputs.
- When writing FINAL.csv and synthesis is NOT required: filter ENV_PATH/property_pool.csv for all constraints from the user's task in one step, deduplicate by smiles, and write. Do not omit any constraint. Do not write FINAL.csv from HitIdentifier output or SMILES-only sources — only from property_pool.csv. FINAL.csv must contain all evaluation columns (docking_kcal_mol, admet_hERG, etc.) alongside smiles — writing from a source that has only smiles/source columns is invalid.
- Molecules already in property_pool.csv are verified; do not delegate them to the Evaluator again.
- When the framework tells you "You have 1 action left": if FINAL.csv does not yet exist, write it now (PROCESS); if FINAL.csv already exists, issue <ANSWER>. Do not delegate when only 1 action remains.
"""
# endregion


# region memory
class WorkingMemory:
    def __init__(
        self,
        logger: Logger | None = None,
        context: Message | str | None = None,
        task_set: Message | str | None = None,
        gating: Callable | None = None,
        threshold: int = 32000,
    ):
        self.episodic_buffer: List[MessageBlock] = []
        self.logger: Logger = logger or Logger()
        self.threshold = threshold

        self.context: Message | None = None
        if context:
            self.update_context(context)

        self.task_set: Message | None = None
        if task_set:
            self.update_task_set(task_set)

        self.gating: Callable | None = None
        if gating:
            self.gating = gating

    def _default_gating(
        self, blocks: List[MessageBlock], tag: str = "mask", recency: int = 8
    ) -> List[MessageBlock]:
        if len(blocks) <= recency:
            return blocks

        result = []
        for i, b in enumerate(blocks):
            if i < len(blocks) - recency and b.tags and tag in b.tags:
                result.append(
                    MessageBlock(
                        role=b.role,
                        message=dataclasses.replace(b.message, content="<MASKED>"),
                        tags=b.tags,
                    )
                )
            else:
                result.append(b)
        return result

    def __repr__(self) -> str:
        lines = []
        for i, b in enumerate(self.state_gated):
            header = (
                f"[{i}] {b.role.upper()} [tags: {b.tags if b.tags is not None else []}]"
            )
            lines.append(f"{header}\n{'-' * len(header)}\n{b.message.content}")
        return "\n\n".join(lines)

    @property
    def state_full(self) -> List[MessageBlock]:
        prefix = (
            [MessageBlock(role="System", message=self.context)]
            if self.context is not None
            else []
        )
        suffix = (
            [MessageBlock(role="User", message=self.task_set)]
            if self.task_set is not None
            else []
        )
        return prefix + self.episodic_buffer + suffix

    @property
    def state_gated(self) -> List[MessageBlock]:
        prefix = (
            [MessageBlock(role="System", message=self.context)]
            if self.context is not None
            else []
        )
        suffix = (
            [MessageBlock(role="User", message=self.task_set)]
            if self.task_set is not None
            else []
        )
        gate = self.gating if self.gating is not None else self._default_gating
        return prefix + gate(self.episodic_buffer) + suffix

    @property
    def snapshot(self) -> Dict[str, Any]:
        def _block_to_dict(b: MessageBlock) -> Dict[str, Any]:
            return {
                "role": b.role,
                "message": {"role": b.message.role, "content": b.message.content},
                "tags": b.tags,
            }

        return {
            "context": self.context.content if self.context is not None else None,
            "task_set": self.task_set.content if self.task_set is not None else None,
            "episodic_buffer": [_block_to_dict(b) for b in self.episodic_buffer],
        }

    @property
    def messages(self) -> List[Message]:
        return [b.message for b in self.state_gated]

    @property
    def full_messages(self) -> List[Message]:
        return [b.message for b in self.state_full]

    def reset_buffer(self):
        self.episodic_buffer = []
        self.logger.debug("Reset 'Orchestrator' Working Memory[Episodic Buffer].")

    def reset(self):
        self.context = None
        self.task_set = None
        self.gating = None
        self.episodic_buffer = []
        self.logger.debug("Reset 'Orchestrator' Working Memory[All].")

    def add(self, block: MessageBlock | List[MessageBlock], debug: bool = True):
        if isinstance(block, list):
            self.episodic_buffer.extend(block)
        elif isinstance(block, MessageBlock):
            self.episodic_buffer.append(block)
        else:
            self.logger.fail(
                f"Invalid instance of block when adding to 'Orchestrator' Working Memory: {block}"
            )
        if debug:
            self.logger.debug(
                f"Add MessageBlock to WorkingMemory. Capacity: {self.current_tokens}/{self.threshold}"
            )

    def add_message(self, role: str, message: Message, tags: List[str] | None = None):
        self.add(MessageBlock(role=role, message=message, tags=tags))

    def update_task_set(self, task_set: str | Message):
        if isinstance(task_set, Message):
            self.task_set = task_set
        elif isinstance(task_set, str):
            self.task_set = Message(role="user", content=task_set)

    def update_context(self, context: str | Message):
        if isinstance(context, Message):
            self.context = context
        elif isinstance(context, str):
            self.context = Message(role="assistant", content=context)
        self.logger.debug(
            f"'Orchestrator' Working Memory[Context] Updated:\n{self.context.content}"
        )

    def get_context(self) -> str | None:
        return self.context.content if self.context else None

    def get_task_set(self) -> str | None:
        return self.task_set.content if self.task_set else None
    
    def get_episodic_buffer(self) -> List[MessageBlock]:
        return copy.deepcopy(self.episodic_buffer)

    @property
    def current_tokens(self) -> int:
        import tiktoken

        enc = tiktoken.get_encoding(
            "cl100k_base"
        )  # works for GPT-4, close enough for Qwen
        return sum(len(enc.encode(m.content)) for m in self.messages)

    def trigger_compact(self) -> bool:
        if self.current_tokens > self.threshold:
            self.logger.debug("'Orchestrator' Working Memory Compactor triggered.")
            return True
        else:
            return False

    def compact(self, context: str | Message | None = None, task_set: str | Message | None = None) -> None:
        """Reset Episodic Buffer + Update Context & Task Set"""
        self.reset_buffer()
        if context:
            self.update_context(context)
        if task_set:
            self.update_task_set(task_set)
        if self.logger:
            self.logger.debug("'Orchestrator' Working Memory compacted.")


class EpisodeMemory:
    def __init__(self, path_to_mem: str):
        self.path_to_mem = path_to_mem
        self._index = 0
        self._state: Dict[int, str] = {}  # idx -> file path

    def add(self, episode: WorkingMemory) -> None:
        os.makedirs(self.path_to_mem, exist_ok=True)
        path = os.path.join(self.path_to_mem, f"episode_{self._index:03d}.json")
        with open(path, "w") as f:
            json.dump(episode.snapshot, f, indent=2)
        self._state[self._index] = path
        self._index += 1

    def __getitem__(self, idx: int) -> List[Dict]:
        if idx not in self._state:
            raise IndexError(f"No episode at index {idx}.")
        with open(self._state[idx], "r") as f:
            return json.load(f)

    def __len__(self) -> int:
        return self._index


class SemanticMemory:
    def __init__(self, path_to_mem: str):
        self.path_to_mem = path_to_mem

    def add(self, docstring: str, path_to_data: str) -> None:
        full_path = os.path.join(self.path_to_mem, path_to_data)
        if not os.path.splitext(full_path)[1]:
            raise ValueError(
                f"path_to_data must be a file path, not a directory: {path_to_data!r}"
            )
        os.makedirs(os.path.dirname(os.path.abspath(full_path)), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(docstring)


# endregion


# region orchestrator
class Orchestrator:
    CTL_FILE_EXTRA_ITER = 5  # extra iterations granted to satisfy a <FILE=...> tag
    TERMINAL_EXTRA_ITER = 1  # extra iteration granted to elicit a terminal action under <STOP=MAX_ITER>

    def __init__(
        self,
        llm: str,
        env_dir: str,
        eps_mem_dir: str,
        sem_mem_dir: str,
        workers: Dict[str, WorkerEntry] | None = None,
        system: str | None = None,
        api_key: str | None = None,
        logger: Logger | None = None,
        llm_args: Literal["default"] | Dict = "default",
        wm_args: Literal["default"] | Dict = "default",
    ):

        ## setup logger
        self.logger = logger or Logger()

        ## setup environment directory
        self.env_dir = os.path.abspath(env_dir)
        os.makedirs(self.env_dir, exist_ok=True)

        ## setup system prompt
        if system is None:
            self.system = ORCHESTRATOR_SYSTEM_PROMPT
        else:
            self.system = system

        ## setup LLM and api_key
        # parse llm_args argument
        self.llm_args: Dict = {}
        if llm_args == "default":
            pass  # provider-specific defaults set after provider resolution
        elif isinstance(llm_args, Dict):
            self.llm_args = llm_args
            self.logger.info(f"'Orchestrator' LLM arguments: {self.llm_args}")
        else:
            self.logger.fail(f"'Orchestrator' invalid llm_args: {llm_args}")

        # check provider
        self.model = llm
        provider = next((p for p in PROVIDERS if llm in p.models), None)
        # Fall back to OpenAI-compatible for unrecognized names (e.g. vLLM / custom models)
        if provider is None:
            provider = next(p for p in PROVIDERS if p.cls is OpenAILLM)
            self.logger.info(
                f"Model '{llm}' not in any known provider set; using OpenAI-compatible backend."
            )
        # default llm_args are provider-specific
        if llm_args == "default":
            if provider.cls is ClaudeLLM:
                DEFAULT_LLM_ARGS = {"cache": "first"}
            else:
                DEFAULT_LLM_ARGS = {}
            self.llm_args = DEFAULT_LLM_ARGS
            self.logger.info(
                f"Use default setting for 'Orchestrator' LLM arguments: {self.llm_args}"
            )
        self.logger.info(f"Initializing 'Orchestrator' with model '{llm}'.")
        self.llm: BaseLLM = provider.cls(
            model=llm,
            api_key=api_key or os.environ.get(provider.env_var),
            system=self.system,
            **(self.llm_args or {}),
        )

        ## setup working memory
        # parse wm_args argument
        self.wm_args: Dict = {}
        if wm_args == "default":
            self.logger.info(
                f"Use default setting for 'Orchestrator' Working Memory arguments: {self.wm_args}"
            )
        elif isinstance(wm_args, Dict):
            self.wm_args = wm_args
            self.logger.info(f"'Orchestrator' Working Memory arguments: {self.wm_args}")
        else:
            self.logger.fail(f"'Orchestrator' invalid wm_args: {wm_args}")
        self._init_working_memory()
        
        # setup working memory summarizer
        self._init_working_memory_summarizer()

        # setup final-answer verifier
        self._init_working_memory_verifier()

        ## setup episodic memory
        self._init_episodic_memory(eps_mem_dir)

        ## setup semantic memory
        self._init_semantic_memory(sem_mem_dir)

        ## setup workers
        if workers is None or len(workers) == 0:
            self.logger.fail("No workers provided.", ValueError)
        else:
            self.logger.info(f"Initializing workers.")
            self._init_workers(workers)

        ## setup automatons
        self._init_automatons()

        ## setup state
        self.state = {
            "llm_responses": [],
            "interactions": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }

        ## baseline snapshot for tracking environment changes between interactions.
        ## independent of DirectoryWatcherAutomaton; advanced on each _record_interaction.
        self._env_snapshot = DirectorySnapshot(self.env_dir)

    # region static methods
    @staticmethod
    def _build_dir_tree(path: str, depth: int = 2, _prefix: str | None = None) -> str:
        lines = []
        if _prefix is None:
            lines.append(os.path.basename(path.rstrip("/")) + "/")
            _prefix = ""
        if depth == 0:
            return "\n".join(lines)
        entries = sorted(os.listdir(path))
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            full_path = os.path.join(path, entry)
            is_dir = os.path.isdir(full_path)
            lines.append(f"{_prefix}{connector}{entry}{'/' if is_dir else ''}")
            if is_dir:
                child_prefix = _prefix + ("    " if is_last else "│   ")
                subtree = Orchestrator._build_dir_tree(
                    full_path, depth - 1, child_prefix
                )
                if subtree:
                    lines.append(subtree)
        return "\n".join(lines)

    # endregion

    # region init methods
    def _init_working_memory(
        self,
        context: Message | str | None = None,
        task_set: Message | str | None = None,
        gating: Callable | None = None,
    ):
        self.working_memory = WorkingMemory(
            logger=self.logger,
            context=context,
            task_set=task_set,
            gating=gating,
            **(self.wm_args or {}),
        )

    def _init_episodic_memory(self, path_to_mem: str):
        self.episodic_memory = EpisodeMemory(path_to_mem)

    def _init_semantic_memory(self, path_to_mem: str):
        self.semantic_memory = SemanticMemory(path_to_mem)

    def _init_workers(self, workers: Dict[str, WorkerEntry]):
        self.workers: Dict[str, WorkerEntry] = workers

    def _init_working_memory_summarizer(self):
        self.working_memory_summarizer: Summarizer = LLMSummarizer(self.llm)

    def _init_working_memory_verifier(self):
        self.verifier: Verifier = LLMVerifier(self.llm, logger=self.logger)

    def _init_automatons(self):
        self.automatons: List[Automaton] = []
        #add DirectoryWatcherAutomaton
        self.automatons.append(DirectoryWatcherAutomaton(path=self.env_dir))

    # endregion

    # region actions
    def process(self, script: str, max_len: int = 2000) -> str:
        import subprocess
        import tempfile

        # Prepend env path so scripts can use ENV_PATH without hardcoding it
        preamble = f"ENV_PATH = {self.env_dir!r}\n\n"
        full_script = preamble + script

        # write to the system temp dir (not env_dir) so the directory watcher
        # does not report the temp script as a spurious file change
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(full_script)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=self.env_dir,
            )
            output = result.stdout

            # truncate output if len(output) > max_len
            if len(output) > max_len:
                output = output[:max_len] + f"(...>{max_len} chars)"
            if result.returncode != 0:
                output += f"\n[stderr]\n{result.stderr}"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: script timed out after 60 seconds."
        finally:
            os.remove(tmp_path)

    def _call_worker(self, worker_name: str, prompt: str) -> str:
        if worker_name not in self.workers:
            self.logger.fail(
                f"_call_worker failed — unknown agent '{worker_name}'. Available: {list(self.workers.keys())}.",
                ValueError,
            )

        worker = self.workers[worker_name].worker
        try:
            self.logger.info(f"Calling '{worker_name}'.")
            worker_response = worker.run(prompt)
            self.logger.debug(f"['{worker_name}' RESPONSE]\n{worker_response}")
            return worker_response
        except Exception as e:
            self.logger.error(f"Agent '{worker_name}' crashed: {e}")
            return f"Agent '{worker_name}' encountered an error: {e}."

    # endregion

    # region parse_action
    def _parse_process(self, content: str) -> tuple[bool, tuple[Any, ...] | str]:
        m = re.search(r"<PROCESS>([\w\W]+)</PROCESS>", content, re.DOTALL)
        block = m.group(1).strip() if m else None
        if not block:
            self.logger.debug("'Orchestrator' PROCESS block parsing failed.")
            return (
                False,
                "Your PROCESS block is not in a valid format. Please strictly follow the specified output format.",
            )
        m = re.search(r"```python([\w\W]+?)```", block, re.DOTALL)
        script = m.group(1).strip() if m else None
        if not script:
            # Fallback: LLM placed the code block after </PROCESS> — grab it anyway
            close_pos = content.find("</PROCESS>")
            after_close = content[close_pos + len("</PROCESS>"):] if close_pos >= 0 else ""
            m2 = re.search(r"```python([\w\W]+?)```", after_close, re.DOTALL)
            if m2:
                self.logger.debug(
                    "'Orchestrator' PROCESS code was outside tags — extracted from after </PROCESS>."
                )
                script = m2.group(1).strip()
            else:
                self.logger.debug(
                    "'Orchestrator' PROCESS block does not contain a Python script."
                )
                return (
                    False,
                    "Your PROCESS block does not contain a Python script. The ```python ... ``` code block MUST be placed INSIDE the <PROCESS> and </PROCESS> tags — not after the closing </PROCESS> tag. Check the ### Action context for more details.",
                )
        # self.logger.debug(f"[PARSED PROCESS SCRIPT]\n{script}")
        return True, (script,)

    def _parse_delegate(self, content: str) -> tuple[bool, tuple[Any, ...] | str]:
        m = re.search(r"<DELEGATE (.*?)>", content)
        worker_name = m.group(1).strip() if m else None
        m = re.search(r"<DELEGATE.*?>([\w\W]+)</DELEGATE>", content, re.DOTALL)
        worker_prompt = m.group(1).strip() if m else None
        if not worker_name or not worker_prompt:
            self.logger.debug("'Orchestrator' DELEGATE block parsing failed.")
            return (
                False,
                "Your DELEGATE block is not in a valid format. Please strictly follow the specified output format.",
            )
        if worker_name not in self.workers:
            self.logger.debug(
                f"'Orchestrator' DELEGATE references unknown agent '{worker_name}'."
            )
            return (
                False,
                f"Agent '{worker_name}' does not exist. Available agents: {list(self.workers.keys())}. Please use one of these.",
            )
        # self.logger.debug(f"[PARSED DELEGATE] agent='{worker_name}'\n{worker_prompt}")
        return True, (worker_name, worker_prompt)

    def _parse_question(self, content: str) -> tuple[bool, tuple[Any, ...] | str]:
        m = re.search(r"<QUESTION (.*?)>", content)
        worker_name = m.group(1).strip() if m else None
        m = re.search(r"<QUESTION.*?>([\w\W]+)</QUESTION>", content, re.DOTALL)
        worker_prompt = m.group(1).strip() if m else None
        if not worker_name or not worker_prompt:
            self.logger.debug("'Orchestrator' QUESTION block parsing failed.")
            return (
                False,
                "Your QUESTION block is not in a valid format. Please strictly follow the specified output format.",
            )
        if worker_name not in self.workers:
            self.logger.debug(
                f"'Orchestrator' QUESTION references unknown agent '{worker_name}'."
            )
            return (
                False,
                f"Agent '{worker_name}' does not exist. Available agents: {list(self.workers.keys())}. Please use one of these.",
            )
        # self.logger.debug(f"[PARSED QUESTION] agent='{worker_name}'\n{worker_prompt}")
        return True, (worker_name, worker_prompt)

    def _parse_answer(self, content: str) -> tuple[bool, tuple[Any, ...] | str]:
        m = re.search(r"<ANSWER>([\w\W]+?)</ANSWER>", content, re.DOTALL)
        answer = m.group(1).strip() if m else content
        # self.logger.debug(f"[PARSED ANSWER]\n{answer}")
        return True, (answer,)

    def _parse_summary(self, content: str) -> tuple[bool, tuple[Any, ...] | str]:
        m = re.search(r"<SUMMARY>([\w\W]+?)</SUMMARY>", content, re.DOTALL)
        answer = m.group(1).strip() if m else content
        # self.logger.debug(f"[PARSED SUMMARY]\n{answer}")
        return True, (answer,)

    # endregion

    # region make_action
    def _make_process_action(self) -> Action:
        return Action(
            tag="</PROCESS>",
            description=(
                "- Data processing: read, inspect, transform, and write data — verify files, convert "
                "formats between agents, merge and deduplicate results, maintain your pool, and inspect "
                "what you hold to decide what to do next (how many pass, what scaffolds you have, what "
                "near-misses fail on). Inspecting the pool this way is valid and useful. Judge a PROCESS "
                "by what it adds: it should write/update a file or produce a fact that informs your next "
                "move. It is wasted only when it adds nothing new — surfacing molecules you already hold "
                "as a 'result', restating known data, or acting just to decide whether to stop. Keep "
                "output small: counts and heads, never whole files.\n"
                "\n"
                "<PROCESS>\n"
                "Objective: [what this produces and how it informs your next move. Concise.]\n"
                "```python\n"
                "import os, shutil\n"
                "import pandas as pd\n"
                "# include necessary imports\n"
                "# your script here\n"
                "```\n"
                "</PROCESS>\n"
                "CRITICAL: The ```python ... ``` code block MUST appear INSIDE the <PROCESS> and </PROCESS> tags. Do NOT place the code after the closing </PROCESS> tag.\n"
                "CRITICAL: Every PROCESS script MUST begin with `import os, shutil` and `import pandas as pd` — each script runs in an isolated environment with no persistent state.\n"
                "CRITICAL: ENV_PATH already points to the env/ directory. Write paths as `os.path.join(ENV_PATH, 'synthesizable_pool.csv')` — NEVER as `ENV_PATH + '/env/synthesizable_pool.csv'` (that creates a wrong env/env/ path).\n"
                "CRITICAL: When appending rows to a CSV that may not exist yet, always use a proper if-else so every variable is defined before use:\n"
                "  if os.path.exists(path):\n"
                "      existing = pd.read_csv(path)\n"
                "      combined = pd.concat([existing, new_rows]).drop_duplicates(subset='smiles')\n"
                "  else:\n"
                "      combined = new_rows.drop_duplicates(subset='smiles')\n"
                "  combined.to_csv(path, index=False)\n"
                "  print(len(combined))  # always defined — never a NameError\n"
                "NEVER define a variable only inside the `if` branch and then use it after the block — this causes NameError when the file does not yet exist.\n"
                "CRITICAL: When appending rows that need a new column label (e.g. generation_source='HI', source='Analog'), assign the column to new_rows BEFORE the concat — never assign to combined after concat, as that overwrites the column value for all existing rows too. Correct pattern: `new_rows['generation_source'] = 'HI'` then concat. Wrong pattern: concat first, then `combined['generation_source'] = 'HI'`.\n"
                "CRITICAL: Standard pre-evaluator dedup — apply this to every molecule DataFrame before writing it to the evaluator/ directory (covers HI output, LeadOptimizer output, and analogs). Two layers: (1) drop internal duplicates, (2) remove smiles already in property_pool.csv or synthesizable_pool.csv:\n"
                "  df = df.drop_duplicates(subset='smiles')\n"
                "  _seen = set()\n"
                "  for _pool in ['property_pool.csv', 'synthesizable_pool.csv']:\n"
                "      _p = os.path.join(ENV_PATH, _pool)\n"
                "      if os.path.exists(_p):\n"
                "          _seen |= set(pd.read_csv(_p)['smiles'])\n"
                "  df = df[~df['smiles'].isin(_seen)]\n"
                "\n"
            ),
            parse=self._parse_process,
            execute=self.process,
            result_role="System",
            result_tags=["mask"],
        )

    def _make_delegate_action(self) -> Action:
        return Action(
            tag="</DELEGATE>",
            description=(
                "- Delegate: issue a task to a specialist agent. Delegation is the ONLY way to obtain "
                "new molecules or specialized analysis. After each delegation, record what was produced and where (file paths), "
                "and carry forward only the relevant structures/IDs — not raw outputs.\n"
                "\n"
                "<DELEGATE AgentName>\n"
                "[prompt to AgentName.]\n"
                "[the agent only accesses its own directory — ensure the files you reference exist there.]\n"
                "</DELEGATE>\n"
                "\n"
            ),
            parse=self._parse_delegate,
            execute=self._call_worker,
            result_role=lambda worker_name, _prompt: worker_name,
            result_tags=None,
        )

    def _make_question_action(self) -> Action:
        return Action(
            tag="</QUESTION>",
            description=(
                "- Question: to ask a specific agent a single, focused question.\n"
                "\n"
                "<QUESTION AgentName>\n"
                "[replace AgentName with actual names of the agent]\n"
                "[Your question to the agent.]\n"
                "</QUESTION>\n"
                "\n"
            ),
            parse=self._parse_question,
            execute=self._call_worker,
            result_role=lambda worker_name, _prompt: worker_name,
            result_tags=None,
        )

    def _make_answer_action(self) -> Action:
        return Action(
            tag="</ANSWER>",
            description=(
                "- Final answer: to issue the final answer to the user.\n"
                "\n"
            "<ANSWER>\n"
            "[Issue only when the goal is met or the action budget is spent. Report the accepted-pool size and file path.]\n"
            "</ANSWER>\n"
            ),
            parse=self._parse_answer,
            execute=None,
            result_role="User",
            result_tags=None,
            terminal=True,
        )

    def _make_summary_action(self) -> Action:
        return Action(
            tag="</SUMMARY>",
            description=(
                "- Final summary: to issue the final summary of agent to the user.\n"
                "\n"
                "<SUMMARY>\n"
                "[Your summary to the user.]\n"
                "</SUMMARY>\n"
                "\n"
                "Use this format when generating the summary:\n\n"
                "# NAME - the agent's name. If none is given, set to ANONYMOUS.\n"
                "# DESCRIPTION - a concise description of what the agent does and what it is useful for. "
                "Frame it in terms of tasks you could delegate to it, not its internal machinery, tools, or models.\n"
                "# NOTES - bullet points that affect HOW YOU DELEGATE — e.g., behaviors to be aware of, special values the agent "
                "reports in its RESPONSE TEXT, how it interprets certain phrasing, what it does by default when you omit "
                "something, ambiguities you should resolve in your prompt. Frame every note around WHAT TO ASK FOR or WHAT TO "
                "SUPPLY, not around which internal tool runs or how that tool is fed. Every note must be actionable purely "
                "through the prompt-in / response-text-out exchange shown in the CLI harness above. Do NOT include: which "
                "specific tool/model/library the agent uses internally or what inputs that tool expects; Python keyword "
                "arguments or code-level parameter names (express requests as prose, not as `kwarg=value`); setup, "
                "infrastructure, file-access, runtime, or discarded-metadata notes — none of those are available to you.\n"
                "# PROMPT FORMAT - what to put in the single free-text prompt, and what must instead be staged in the "
                "working/environment directory. Since the prompt is one free-text field, all task instructions, targets, "
                "baselines, requested outputs, and formatting requests go inside it as prose; data files go in the environment "
                "directory. Only include conventions the agent actually requires.\n"
                "# PROMPT EXAMPLE - a few realistic example prompts. Each example is ONLY the text you would pass as the prompt "
                "argument — free-text prose, with no 'Task config:' lines, no structured parameters, and no Python kwargs, since "
                "the harness provides no such arguments. Refer to input data as residing in the working/environment directory.\n"
                "# CONSTRAINTS - behavioral or task-level limits that affect what you can ask of the agent — input formats, "
                "scope of tasks it will refuse or cannot do, output limits, and semantic limits (i.e. preconditions on what you "
                "must supply for a given task to be possible). State these in terms of what YOU must supply or avoid, not in "
                "terms of tool internals. Do NOT include deployment, dependency, runtime, or infrastructure constraints. If "
                "there are no delegation-relevant constraints, write 'None specified.'\n"
            ),
            parse=self._parse_summary,
            execute=None,
            result_role="User",
            result_tags=None,
            terminal=True,
        )

    # endregion

    def get_state(self) -> Dict[str, Any]:
        return copy.deepcopy(self.state)

    def _diff_environment(self) -> Dict[str, List[str]] | None:
        """File-level changes in env_dir since the last call, as
        {created/modified/deleted/moved: [paths relative to env_dir]}. Advances
        the baseline. Returns None when nothing changed. Independent of any
        automaton."""
        try:
            snapshot = DirectorySnapshot(self.env_dir)
        except OSError:
            return None  # transient walk error; keep baseline, retry next time
        diff = DirectorySnapshotDiff(self._env_snapshot, snapshot)
        self._env_snapshot = snapshot

        rel = lambda p: os.path.relpath(p, self.env_dir)
        change = {
            "created": sorted(rel(p) for p in diff.files_created),
            "modified": sorted(rel(p) for p in diff.files_modified),
            "deleted": sorted(rel(p) for p in diff.files_deleted),
            "moved": sorted(f"{rel(src)} -> {rel(dest)}" for src, dest in diff.files_moved),
        }
        if not any(change.values()):
            return None
        return change

    def _record_interaction(
        self,
        response: str,
        action_tag: str | None = None,
        parse: bool = False,
        payload: List | None = None,
        result: str | None = None,
        time: float | None = None,
    ) -> None:
        """Append one orchestrator interaction (LLM response + outcome) to
        state['interactions']."""
        self.state["interactions"].append(
            {
                "timestamp": datetime.now().strftime("%m-%d %H:%M:%S"),
                "response": response,
                "action": action_tag,
                "parse": parse,
                "payload": payload,
                "result": result,
                "time": time,
                "environment_change": self._diff_environment(),
            }
        )

    def get_llm_response(
        self, messages: List[Message], log_id: str | None = None, **kwargs
    ) -> LLMResponse:
        response = self.llm.get_response(messages, **kwargs)

        llm_response = {
            "id": log_id if log_id else "None",
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cache_read_tokens": response.cache_read_tokens,
            "cache_write_tokens": response.cache_write_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "response": response.content,
            # "thinking": response.thinking
        }

        self.state["llm_responses"].append(llm_response)
        self.state["input_tokens"] += response.input_tokens
        self.state["output_tokens"] += response.output_tokens
        self.state["cache_read_tokens"] += response.cache_read_tokens
        self.state["cache_write_tokens"] += response.cache_write_tokens

        return response

    def get_environment(self) -> str:
        FORMAT = (
            "## Path to the environment:\n{env_dir}\n\n"
            "## Data in the environment:\n{env_files}\n\n"
        )

        return FORMAT.format(
            env_dir=self.env_dir, env_files=self._build_dir_tree(self.env_dir)
        )

    def _default_create_context(self) -> str:
        CONTEXT = (
            "### Environment:\n{env}\n\n"
            "### Available Agents:\n\n{agent_list}\n\n"
            "### Automatons:\n\n{automaton_list}\n\n"
        )

        agent_list = "\n\n".join(
            f"<{name}>\n{entry.profile.read_text()}\n</{name}>"
            for name, entry in self.workers.items()
        )

        automaton_list = (
            "\n".join(f"{a.name}: {a.description}" for a in self.automatons)
            if self.automatons
            else "None."
        )

        return CONTEXT.format(
            env=self.get_environment(),
            agent_list=agent_list,
            automaton_list=automaton_list,
        )

    def run(self, prompt: str, max_iterations: int = 5, action_log_path: str | None = None) -> str | None:
        ## Use Default Context
        context_generator = self._default_create_context

        task_set = (
            "<TASK_SET>\n"
            # "RULE: Always reiterate the user's task in your thinking.\n"
            "RULE: Provide FULL PATH when delegating.\n"
            "RULE: Always follow output format in ## Action.\n"
            "RULE: Think step-by-step before taking an action.\n"
            "RULE: File convention standard when creating or copying files: `XXX_[name].ext`, where XXX is a number starting from 000. Example: `003_hit_identifier.csv`\n"
            "RULE: Check agent directory after every agent's response. Some agents returns error or reaches limit, but they still store some results in the directory.\n"
            # "SUGGESTION: Never present more than 5 SMILES in ANSWER, refer them to a file instead.\n"
            "WARNING: If user provide property requirements, focus only on those properties when delegating.\n"
            "WARNING: REMEMBER TO COPY FILE! Each agent only work with files in their respective directories.\n"
            "WARNING: Verify results from each agent by checking the content of the files yourself. They may hallucinate.\n"
            "</TASK_SET>"
        )

        actions = [
            self._make_delegate_action(),
            self._make_process_action(),
            self._make_answer_action(),
        ]

        return self._run_procedure(
            prompt=prompt,
            context_generator=context_generator,
            task_set=task_set,
            actions=actions,
            max_iterations=max_iterations,
            action_log_path=action_log_path,
        )

    # region stream
    # def stream(
    #     self,
    #     prompt: str,
    #     max_iterations: int = 5,
    #     reset_working_memory: bool = False,
    # ) -> Iterator[dict]:
    #     """Generator version of run(). Mirrors _run_procedure's behaviour but
    #     yields event dicts for each orchestrator action so a caller can render
    #     progress incrementally."""
    #     ## assertions
    #     assert max_iterations > 0, "max_iterations must be > 0"

    #     ## Use Default Context (same generator run() uses)
    #     context_generator = self._default_create_context

    #     task_set = (
    #         "<TASK_SET>\n"
    #         "RULE: Always reiterate the user's task in your thinking.\n"
    #         "RULE: Provide FULL PATH when delegating, except HitIdentifier which can only work with 'X.pdb' file.\n"
    #         "RULE: Always follow output format in ## Action.\n"
    #         "RULE: Think step-by-step before taking an action.\n"
    #         # "SUGGESTION: Never present more than 5 SMILES in ANSWER, refer them to a file instead.\n"
    #         "WARNING: REMEMBER TO COPY FILE! Each agent only work with files in their respective directories.\n"
    #         "WARNING: Verify results from each agent by checking the content of the files yourself. They may hallucinate.\n"
    #         "</TASK_SET>"
    #     )

    #     actions = [
    #         self._make_delegate_action(),
    #         self._make_process_action(),
    #         self._make_answer_action(),
    #     ]

    #     ## initialize working memory
    #     # init context (generate context + append actions)
    #     context = context_generator()
    #     action_section = (
    #         "### Action\n"
    #         "At the end of your reasoning process, your output must be structured in one of the following formats, depending on what you think is the best action.\n"
    #         "Make sure you only take ONE ACTION at a time, and that your output strictly adheres to the format for that action.\n"
    #         "\n" + "".join(a.description for a in actions)
    #     )
    #     full_context = f"{context}\n{action_section}"
    #     self._init_working_memory(context=full_context, task_set=task_set)

    #     ## add user's or task prompt to working memory
    #     self.working_memory.add_message("User", Message(role="user", content=prompt))

    #     self.logger.info(f"'Orchestrator' stream started.")
    #     self.logger.debug(f"[USER MESSAGE]\n{prompt}")

    #     yield {"type": "start"}

    #     just_compacted = False
    #     for i in range(max_iterations):
    #         self.logger.info(f"'Orchestrator' iteration {i + 1}/{max_iterations}.")
    #         yield {"type": "iteration", "current": i + 1, "max": max_iterations}

    #         # update task set to keep count of how many actions left
    #         self.working_memory.update_task_set(
    #             f"{task_set}\n\nYou have {max_iterations - i} actions left."
    #         )

    #         self.logger.info(f"Calling 'Orchestrator'.")

    #         # run LLM on current working memory
    #         response = self.get_llm_response(
    #             self.working_memory.messages, f"stream_call_{i:03d}"
    #         )
    #         self.working_memory.add_message(
    #             "Orchestrator", Message(role="assistant", content=response.content)
    #         )
    #         self.logger.debug(f"['Orchestrator' RESPONSE]\n{response.content}")

    #         # CHECKPOINT: ensure exactly one action AND response format is valid
    #         matched = [a for a in actions if a.tag in response.content]
    #         if len(matched) > 1:
    #             self.logger.debug(
    #                 f"'Orchestrator' returned multiple actions {[a.tag for a in matched]}, sending correction prompt."
    #             )
    #             _message = Message(
    #                 role="user",
    #                 content="You took more than one action. You must take exactly ONE action at a time. Please respond again with only one action.",
    #             )
    #             self.working_memory.add_message("System", _message, tags=["mask"])
    #             continue
    #         if len(matched) == 0:
    #             self.logger.debug(
    #                 "'Orchestrator' response format invalid, sending correction prompt."
    #             )
    #             _message = Message(
    #                 role="user",
    #                 content="Your response is not in a valid format. Please strictly follow the output format in ## Action.",
    #             )
    #             self.working_memory.add_message("System", _message, tags=["mask"])
    #             continue

    #         # parse action
    #         action = matched[0]
    #         ok, payload = action.parse(response.content)
    #         if not ok:
    #             self.logger.debug(
    #                 f"'Orchestrator' action '{action.tag}' parse failed, sending correction prompt."
    #             )
    #             assert isinstance(payload, str)
    #             _message = Message(role="user", content=payload)
    #             self.working_memory.add_message("System", _message)
    #             continue

    #         yield {"type": "action", "tag": action.tag}

    #         if action.terminal:
    #             self.logger.info(
    #                 f"'Orchestrator' reached terminal after {i + 1} iteration(s)."
    #             )
    #             assert isinstance(payload[0], str)
    #             self.working_memory.add_message(
    #                 "Orchestrator", Message(role="assistant", content=payload[0])
    #             )
    #             ## add snapshot of memory (episode) to episodic memory
    #             self.episodic_memory.add(self.working_memory)
    #             ## reset working memory
    #             if reset_working_memory:
    #                 self.working_memory.reset()
    #             yield {"type": "answer", "content": payload[0]}
    #             return

    #         # execute action
    #         self.logger.info(f"Running action '{action.tag}'.")
    #         assert isinstance(payload, tuple)
    #         assert action.execute is not None

    #         if "DELEGATE" in action.tag:
    #             self.logger.info(f"Delegating to '{payload[0]}'.")
    #             yield {"type": "delegate", "worker": payload[0], "prompt": payload[1]}
    #         elif "PROCESS" in action.tag:
    #             self.logger.info(f"Running PROCESS script.")
    #             yield {"type": "process", "script": payload[0]}

    #         result = action.execute(*payload)
    #         yield {"type": "action_result", "content": result}
    #         result = "<ACTION OUTPUT>\n" + result + "\n</ACTION OUTPUT>"
    #         self.logger.debug(f"[ACTION OUTPUT]\n{result}")

    #         # add to working memory
    #         _message = Message(role="user", content=result)
    #         role = (
    #             action.result_role(*payload)
    #             if callable(action.result_role)
    #             else action.result_role
    #         )
    #         self.working_memory.add_message(role, _message, tags=action.result_tags)

    #         # running compactualization if triggered
    #         just_compacted = False
    #         if self.working_memory.trigger_compact():
    #             self.compact_working_memory(context_generator, action_section)
    #             just_compacted = True
    #             yield {"type": "compact"}

    #     ## add snapshot of memory (episode) to episodic memory only if not just compacted
    #     if not just_compacted:
    #         self.episodic_memory.add(self.working_memory)
    #     ## reset working memory
    #     if reset_working_memory:
    #         self.working_memory.reset()

    #     self.logger.error(
    #         f"Max iterations ({max_iterations}) reached without reaching terminal."
    #     )
    #     yield {
    #         "type": "error",
    #         "content": f"Max iterations ({max_iterations}) reached without a final answer.",
    #     }

    # endregion

    def _finalize_run(self, answer: str, reset_working_memory: bool) -> str:
        """Record a terminal answer, snapshot the episode, optionally reset, and
        return it. The single exit door for a successful :meth:`_run_procedure`."""
        self.working_memory.add_message(
            "Orchestrator", Message(role="assistant", content=answer)
        )
        self.episodic_memory.add(self.working_memory)
        if reset_working_memory:
            self.working_memory.reset()
        return answer

    def _run_procedure(
        self,
        prompt: str,
        context_generator: Callable[[], str],
        task_set: str,
        actions: List[Action],
        max_iterations: int,
        reset_working_memory: bool = False,
        action_log_path: str | None = None,
    ) -> str | None:
        ## assertions
        assert max_iterations > 0, "max_iterations must be > 0"

        ## action log helper
        import json as _json
        from datetime import datetime as _dt

        def _log_action(iter_num: int, action_tag: str, payload: tuple, exec_time: float, response_content: str) -> None:
            if action_log_path is None:
                return
            tag_map = {"</DELEGATE>": "DELEGATE", "</PROCESS>": "PROCESS", "</ANSWER>": "ANSWER"}
            action_type = tag_map.get(action_tag, action_tag.strip("</>"))
            if action_type == "DELEGATE":
                detail = f"{payload[0]}: {str(payload[1])[:100]}"
            elif action_type == "PROCESS":
                m = re.search(r"Objective:\s*(.+)", response_content)
                detail = m.group(1).strip()[:120] if m else str(payload[0])[:100]
            else:
                detail = str(payload[0])[:100]
            entry = {"iter": iter_num, "ts": _dt.now().strftime("%H:%M:%S"), "action": action_type,
                     "detail": detail, "time_s": exec_time}
            if action_type == "DELEGATE":
                entry["agent"] = payload[0]
            with open(action_log_path, "a") as _f:
                _f.write(_json.dumps(entry) + "\n")

        ## parse user prompt - extract control tags and strip them from the prompt
        self.logger.debug(f"[USER MESSAGE]\n{prompt}")
        prompt, control = self._parse_control_tag(prompt)
        if control:
            self.logger.info(f"'Orchestrator' control tags: {control}")

        # <STOP=MAX_ITER>: terminal actions do NOT end the run early; the loop
        # keeps going until all iterations are exhausted, returning the most recent
        # terminal answer at the end.
        force_max_iter = control.get("stop") == "MAX_ITER"
        last_terminal_answer: str | None = None

        # <FILE=...>: the run may only terminate once this file exists and is
        # non-empty in the environment. Grant extra iterations to produce it.
        file_target = control.get("file")  # e.g. "FINAL.csv" or None
        file_abs = os.path.join(self.env_dir, file_target) if file_target else None

        def _file_ok() -> bool:
            return bool(file_abs) and os.path.isfile(file_abs) and os.path.getsize(file_abs) > 0

        # <POOL=N>: stop early when synthesizable_pool.csv reaches N qualified molecules.
        _pool_raw = control.get("pool")
        pool_target: int | None = _pool_raw if isinstance(_pool_raw, int) and _pool_raw > 0 else None

        def _pool_size() -> int:
            pool_csv = os.path.join(self.env_dir, "synthesizable_pool.csv")
            if not os.path.isfile(pool_csv):
                return 0
            try:
                with open(pool_csv) as _f:
                    return max(0, sum(1 for _ in _f) - 1)
            except Exception:
                return 0

        def _write_final_from_pool(n: int | None) -> None:
            pool_csv = os.path.join(self.env_dir, "synthesizable_pool.csv")
            dest = file_abs if file_abs else os.path.join(self.env_dir, "FINAL.csv")
            if not os.path.isfile(pool_csv):
                self.logger.warning("<POOL> synthesizable_pool.csv not found; skipping FINAL.csv write.")
                return
            try:
                import csv as _csv
                with open(pool_csv, newline="") as _in:
                    _rows = list(_csv.reader(_in))
                _header = _rows[:1]
                _data = _rows[1:] if n is None else _rows[1 : n + 1]
                with open(dest, "w", newline="") as _out:
                    _csv.writer(_out).writerows(_header + _data)
                self.logger.info(f"<POOL> wrote {len(_data)} molecule(s) to '{dest}'.")
            except Exception as _e:
                self.logger.warning(f"<POOL> failed to write FINAL.csv: {_e}")

        _pool_csv_abs = os.path.join(self.env_dir, "synthesizable_pool.csv")

        # make the <STOP=MAX_ITER> requirement visible to the orchestrator every iteration
        if force_max_iter:
            task_set = (
                f"{task_set}\n\nREQUIRED: Do NOT give your final answer until you have "
                f"used all your actions. Keep working — verify results, refine your "
                f"analysis, or run more delegations — and only finish when no actions remain."
            )

        # make the file requirement visible to the orchestrator every iteration
        if file_target:
            task_set = (
                f"{task_set}\n\nREQUIRED: Before your final answer, save the final "
                f"molecule set to '{file_target}' in the environment (full path: "
                f"{file_abs}). You cannot finish until that file exists."
            )

        if pool_target is not None:
            task_set = (
                f"{task_set}\n\nFor this run, target_count = {pool_target}. "
                f"Use this value in the Synthesis workflow Trigger condition 1, Step S4, and FINAL.csv row count."
            )

        ## initialize working memory
        # init context (generate context + append actions)
        context = context_generator()
        action_section = (
            "### Action\n"
            "At the end of your reasoning process, your output must be structured in one of the following formats, depending on what you think is the best action.\n"
            "CRITICAL: You must take EXACTLY ONE ACTION per response. Never combine a <PROCESS> and a <DELEGATE> (or any two actions) in the same response — even if they feel like a natural sequence. Finish one action, wait for its result, then take the next action in your following response.\n"
            "\n" + "".join(a.description for a in actions)
        )
        full_context = f"{context}\n{action_section}"
        self._init_working_memory(context=full_context, task_set=task_set)

        ## add user's or task prompt to working memory
        self.working_memory.add_message("User", Message(role="user", content=prompt))

        self.logger.info(f"'Orchestrator' run started.")

        ## init loop
        # start automatons (created in __init__; start() builds a fresh observer)
        for automaton in self.automatons:
            automaton.start()
        try:
            just_compacted = False

            def run_turn(idx: int, budget: int, log_id: str) -> str | None:
                """One orchestrator turn. Returns the answer to finalize when a
                terminal action is accepted, else None to keep looping."""
                nonlocal last_terminal_answer, just_compacted

                ## update task set
                # update based on how many actions are left
                self.working_memory.update_task_set(
                    f"{task_set}\n\nYou have {budget - idx} actions left."
                )

                self.logger.info(f"Calling 'Orchestrator'.")

                # run LLM on current working memory
                response = self.get_llm_response(
                    self.working_memory.messages, log_id
                )
                self.working_memory.add_message(
                    "Orchestrator", Message(role="assistant", content=response.content)
                )
                self.logger.debug(f"['Orchestrator' RESPONSE]\n{response.content}")

                # CHECKPOINT: ensure exactly one action AND response format is valid
                matched = [a for a in actions if a.tag in response.content]
                if len(matched) > 1:
                    self.logger.debug(
                        f"'Orchestrator' returned multiple actions {[a.tag for a in matched]}, sending correction prompt."
                    )
                    self._record_interaction(response.content)
                    _message = Message(
                        role="user",
                        content="You took more than one action. You must take exactly ONE action at a time. Please respond again with only one action.",
                    )
                    self.working_memory.add_message("System", _message, tags=["mask"])
                    if self.working_memory.trigger_compact():
                        self.compact_working_memory(context_generator, action_section)
                        just_compacted = True
                    _log_action(idx + 1, "MULTIPLE_ACTIONS", (f"multiple actions: {[a.tag for a in matched]}",), 0.0, response.content)
                    return None
                if len(matched) == 0:
                    self.logger.debug(
                        "'Orchestrator' response format invalid, sending correction prompt."
                    )
                    self._record_interaction(response.content)
                    _message = Message(
                        role="user",
                        content="Your response is not in a valid format. Please strictly follow the output format in ## Action.",
                    )
                    self.working_memory.add_message("System", _message, tags=["mask"])
                    if self.working_memory.trigger_compact():
                        self.compact_working_memory(context_generator, action_section)
                        just_compacted = True
                    _log_action(idx + 1, "FORMAT_INVALID", ("no valid action tag",), 0.0, response.content)
                    return None

                # Parse action
                action = matched[0]
                ok, payload = action.parse(response.content)
                if not ok:
                    self.logger.debug(
                        f"'Orchestrator' action '{action.tag}' parse failed, sending correction prompt."
                    )
                    self._record_interaction(response.content, action.tag, parse=False)
                    assert isinstance(payload, str)
                    _log_action(idx + 1, "PARSE_FAIL", (payload[:120],), 0.0, response.content)
                    _message = Message(role="user", content=payload)
                    self.working_memory.add_message("System", _message)
                    if self.working_memory.trigger_compact():
                        self.compact_working_memory(context_generator, action_section)
                        just_compacted = True
                    return None

                if action.terminal:
                    assert isinstance(payload[0], str)
                    self._record_interaction(
                        response.content,
                        action.tag,
                        parse=True,
                        payload=list(payload),
                        result=payload[0],
                    )
                    # <FILE=...> gate: refuse to terminate until the required file
                    # exists and is non-empty; bank the answer and nudge instead.
                    if file_target and not _file_ok():
                        self.logger.info(
                            f"'Orchestrator' terminal blocked: required file "
                            f"'{file_target}' is missing or empty."
                        )
                        last_terminal_answer = payload[0]
                        self.working_memory.add_message(
                            "System",
                            Message(
                                role="user",
                                content=(
                                    f"You attempted to finish, but the required output file "
                                    f"'{file_target}' does not exist yet (or is empty). Save the "
                                    f"final molecule set to it (full path: {file_abs}) before "
                                    f"giving your final answer. You have "
                                    f"{budget - idx - 1} action(s) left."
                                ),
                            ),
                            tags=["mask"],
                        )
                        _log_action(idx + 1, "TERMINAL_BLOCKED", (f"file_missing: {file_target}",), 0.0, response.content)
                        return None

                    # <FILE=...> verification gate: the file exists, but an LLM must
                    # confirm it actually answers the user's prompt before we finish.
                    # On rejection, bank the answer and nudge with the reason instead.
                    if file_target and _file_ok():
                        ok, reason = self.verifier.verify_csv(prompt, file_abs)
                        if not ok:
                            self.logger.info(
                                f"'Orchestrator' terminal blocked: verifier rejected "
                                f"'{file_target}': {reason}"
                            )
                            last_terminal_answer = payload[0]
                            self.working_memory.add_message(
                                "System",
                                Message(
                                    role="user",
                                    content=(
                                        f"You attempted to finish, but a review of '{file_target}' "
                                        f"found it does not yet answer the user's request. Reason: "
                                        f"{reason}\nRefine your results and update the file (full path: "
                                        f"{file_abs}) before giving your final answer. You have "
                                        f"{budget - idx - 1} action(s) left."
                                    ),
                                ),
                                tags=["mask"],
                            )
                            _log_action(idx + 1, "TERMINAL_BLOCKED", (f"verifier_rejected: {reason[:80]}",), 0.0, response.content)
                            return None
                        self.logger.info(
                            f"'Orchestrator' verifier accepted '{file_target}': {reason}"
                        )

                    # <STOP=MAX_ITER> override: don't end on a terminal action; bank the
                    # answer, nudge the model, and keep going until the budget is spent.
                    if force_max_iter and idx < budget - 1:
                        self.logger.info(
                            f"'Orchestrator' produced a terminal action at iteration {idx + 1}, "
                            f"but <STOP=MAX_ITER> is set; banking it and continuing."
                        )
                        last_terminal_answer = payload[0]
                        self.working_memory.add_message(
                            "System",
                            Message(
                                role="user",
                                content=(
                                    f"You still have {budget - idx - 1} action(s) left "
                                    "and must not stop yet. Keep working — verify your results, "
                                    "refine your analysis, or run additional delegations — and "
                                    "only give a final answer when no actions remain."
                                ),
                            ),
                            tags=["mask"],
                        )
                        _log_action(idx + 1, "TERMINAL_BLOCKED", ("STOP=MAX_ITER",), 0.0, response.content)
                        return None

                    self.logger.info("'Orchestrator' reached terminal.")
                    _log_action(i + 1, action.tag, payload, 0.0, response.content)
                    return payload[0]

                # execute action (timed)
                self.logger.info(f"Running action '{action.tag}'.")
                assert isinstance(payload, tuple)
                assert action.execute is not None
                _t0 = time.perf_counter()
                result = action.execute(*payload)
                exec_time = round(time.perf_counter() - _t0, 3)
                _log_action(i + 1, action.tag, payload, exec_time, response.content)
                result = "<ACTION OUTPUT>\n" + result + "\n</ACTION OUTPUT>"
                self.logger.debug(f"[ACTION OUTPUT]\n{result}")

                # append output of any triggered automaton to the action result
                for automaton in self.automatons:
                    if automaton.trigger:
                        output = automaton.consume()
                        if output:
                            result += f"\n\n<AUTOMATON {automaton.name}>\n{output}\n</AUTOMATON>"
                            self.logger.debug(f"[AUTOMATON {automaton.name}]\n{output}")

                # record interaction (result includes any automaton output)
                self._record_interaction(
                    response.content,
                    action.tag,
                    parse=True,
                    payload=list(payload),
                    result=result,
                    time=exec_time,
                )

                # add to working memory
                _message = Message(role="user", content=result)
                role = (
                    action.result_role(*payload)
                    if callable(action.result_role)
                    else action.result_role
                )
                self.working_memory.add_message(role, _message, tags=action.result_tags)

                # running compactualization if triggered
                just_compacted = False
                if self.working_memory.trigger_compact():
                    self.compact_working_memory(context_generator, action_section)
                    just_compacted = True

                return None

            # main loop — the agent's real budget is max_iterations
            for i in range(max_iterations):
                self.logger.info(f"'Orchestrator' iteration {i + 1}/{max_iterations}.")
                answer = run_turn(i, max_iterations, f"run_call_{i:03d}")
                if answer is not None:
                    return self._finalize_run(answer, reset_working_memory)
                if pool_target is not None:
                    current = _pool_size()
                    if current >= pool_target:
                        self.logger.info(
                            f"<POOL={pool_target}> reached ({current} rows in synthesizable_pool.csv); "
                            f"writing FINAL.csv and stopping."
                        )
                        _write_final_from_pool(pool_target)
                        last_terminal_answer = (
                            f"Pool target of {pool_target} molecules reached "
                            f"({current} rows); FINAL.csv written from accepted pool."
                        )
                        break

            # <POOL=N> fallback: if pool target was set but loop exhausted, write FINAL.csv from pool
            if pool_target is not None and not _file_ok():
                self.logger.info(
                    f"Main loop ended with <POOL={pool_target}>; writing FINAL.csv from pool ({_pool_size()} rows)."
                )
                _write_final_from_pool(None)

            # grace loop — extra budget to produce the required <FILE=...> and then
            # to elicit a brand-new final answer once it exists.
            if file_target and not _file_ok():
                self.logger.info(
                    f"Base budget spent; granting up to {self.CTL_FILE_EXTRA_ITER} "
                    f"extra iteration(s) to produce '{file_target}'."
                )
                for j in range(self.CTL_FILE_EXTRA_ITER):
                    self.logger.info(
                        f"'Orchestrator' extra iteration {j + 1}/{self.CTL_FILE_EXTRA_ITER}."
                    )
                    if _file_ok():
                        nudge = (
                            f"The required file '{file_target}' now exists. Give your final "
                            f"answer now."
                        )
                    else:
                        nudge = (
                            f"Your {max_iterations} actions are used up, but the required "
                            f"file '{file_target}' is still missing or empty. You have "
                            f"{self.CTL_FILE_EXTRA_ITER - j} extra action(s) to save the final "
                            f"molecule set to it (full path: {file_abs}), then give your final answer. "
                            "Make sure the final set contains both SMILES and the computed properties with their appropriate names, like 'QED', 'Synthesis Path', etc.\n"
                            f"User's Prompt: {prompt}"
                        )
                    self.working_memory.add_message(
                        "System", Message(role="user", content=nudge), tags=["mask"]
                    )
                    answer = run_turn(j, self.CTL_FILE_EXTRA_ITER, f"file_call_{j:03d}")
                    if answer is not None:
                        return self._finalize_run(answer, reset_working_memory)

            ## <FILE=...>: budget spent without the required file — warn, then fall
            ## through to return the banked answer (best-effort).
            if file_target and not _file_ok():
                self.logger.error(
                    f"Required output file '{file_target}' was never produced "
                    f"(checked: {file_abs}); returning best-effort answer."
                )

            ## <STOP=MAX_ITER> terminal grace: the budget is spent but the model never
            ## produced a terminal action (it spent its last action on other work, e.g.
            ## a PROCESS). Grant a small extra budget used solely to elicit one.
            if force_max_iter and last_terminal_answer is None:
                self.logger.info(
                    f"Base budget spent under <STOP=MAX_ITER> without a terminal action; "
                    f"granting up to {self.TERMINAL_EXTRA_ITER} extra iteration(s) to elicit one."
                )
                for j in range(self.TERMINAL_EXTRA_ITER):
                    self.logger.info(
                        f"'Orchestrator' terminal iteration {j + 1}/{self.TERMINAL_EXTRA_ITER}."
                    )
                    nudge = (
                        "Your actions are all used up. Stop working now and respond with your "
                        "final terminal action to conclude the run."
                    )
                    self.working_memory.add_message(
                        "System", Message(role="user", content=nudge), tags=["mask"]
                    )
                    answer = run_turn(j, self.TERMINAL_EXTRA_ITER, f"terminal_call_{j:03d}")
                    if answer is not None:
                        return self._finalize_run(answer, reset_working_memory)

            ## <STOP=MAX_ITER>: return the answer we banked earlier now the budget is spent.
            if last_terminal_answer is not None:
                self.logger.info(
                    f"Max iterations ({max_iterations}) reached; returning banked terminal "
                    "answer (<STOP=MAX_ITER>)."
                )
                return self._finalize_run(last_terminal_answer, reset_working_memory)

            ## no terminal answer: snapshot the episode (unless a compaction just did) and report.
            if not just_compacted:
                self.episodic_memory.add(self.working_memory)
            if reset_working_memory:
                self.working_memory.reset()
            self.logger.error(
                f"Max iterations ({max_iterations}) reached without reaching terminal."
            )
        finally:
            # stop automatons
            for automaton in getattr(self, "automatons", []):
                try:
                    automaton.stop()
                except Exception as e:
                    self.logger.fail(
                        f"Failed to stop automaton '{automaton.name}': {e}",
                        RuntimeError,
                    )

    def _parse_control_tag(self, prompt: str) -> tuple[str, Dict]:
        """Extract ``<KEY=VALUE>`` control tags from a user prompt.

        Scans for tags of the form ``<KEY=VALUE>``, strips them out of the
        prompt, and returns ``(cleaned_prompt, control_params)``:

        - ``My name is MADDIE <STOP=MAX_ITER>`` -> ``('My name is MADDIE', {'stop': 'MAX_ITER'})``
        - ``Hello who are you <PLAN=TRUE>``     -> ``('Hello who are you', {'plan': True})``

        Values are coerced: ``TRUE``/``FALSE`` (case-insensitive) become booleans,
        integer-looking values become ``int``, everything else is a stripped
        string with its original case preserved. On a duplicate key the last
        occurrence wins.
        """
        # KEY: letters/digits/underscore; VALUE: anything up to the closing '>'.
        _CONTROL_TAG_RE = re.compile(r"<\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^>]*?)\s*>")

        control_params: Dict = {}
        for match in _CONTROL_TAG_RE.finditer(prompt):
            key = match.group(1).strip().lower()
            raw = match.group(2).strip()

            value: Any
            if raw.lower() in ("true", "false"):
                value = raw.lower() == "true"
            elif re.fullmatch(r"[+-]?\d+", raw):
                value = int(raw)
            else:
                value = raw

            control_params[key] = value
            self.logger.debug(f"'Orchestrator' parsed control tag: {key}={value!r}")

        # remove the matched tags, then tidy up the whitespace they left behind
        cleaned = _CONTROL_TAG_RE.sub("", prompt)
        cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()

        return cleaned, control_params

    def compact_working_memory(self, context_generator: Callable[[], str], action_section: str, summarizer: Optional[Summarizer] = None) -> None:
        # add existing environment to WM
        self.working_memory.add_message(
            "System",
            Message(
                role="user",
                content=f"Current Environment:\n{self.get_environment()}",
            ),
        )
        # add WM to EM
        self.episodic_memory.add(self.working_memory)
        
        ## Compaction algorithm
        self.logger.debug("'Orchestrator' Working Memory Compaction initiated.")
        
        # run summarizer
        _summarizer = summarizer or self.working_memory_summarizer
        summary = _summarizer.summarize(self.working_memory.get_episodic_buffer())
        
        # create new context (re-append the action section, as in _init_working_memory)
        context = context_generator()
        full_context = f"{context}\n{action_section}\n\n### PREVIOUS SUMMARY\n\n{summary}"
        self.working_memory.compact(context=full_context)

# endregion

if __name__ == "__main__":
    import os

    api_key = os.getenv("ANTHROPIC_API_KEY")

    # orchestrator = Orchestrator(
    #     llm="claude-sonnet-4-6",
    #     env_dir="/path/to/MADDIE/env",
    #     api_key=api_key,
    # )

    # response = orchestrator.run("Who are you?")
    # print(response)
