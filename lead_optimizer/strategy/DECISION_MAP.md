# LeadOptimizer Phase 2 Decision Map

This note defines the bounded decision space for LeadOptimizer's optional
reasoning layer. The goal is to make LO more adaptive without turning it into a
broad autonomous agent loop. The LLM may recommend high-level strategy, but
execution remains owned by `LeadOptimizerAgent`, `DirectOptimizer`, and
`AdaptiveOptimizerRunner`.

## Design Principles

- Keep deterministic planning and adaptive fallback as the baseline.
- Treat benchmark packets as imperfect evidence, not ground truth.
- Use the LLM for bounded strategy recommendations, not direct tool execution.
- Validate every LLM action before execution.
- Prefer clear partial/failure reporting over unbounded retries.
- Preserve raw LLM responses and structured decision traces in metadata.

## Initial Planning Actions

These actions happen before the first optimizer run.

| Action | Meaning | Required Validation |
| --- | --- | --- |
| `optimize_csv` | Select one compatible tool and run CSV optimization. | Tool exists, is not excluded, supports requested mode, input file resolved, objective parsed. |
| `use_user_requested_tool` | Honor an explicit tool request. | Requested tool exists and is compatible with objective/request mode. |
| `override_baseline_tool` | Choose a non-top deterministic/benchmark tool with rationale. | Tool exists, is compatible, and rationale cites objective, tool docs, prior evidence, or user preference. |
| `ask_clarification` | Stop before execution because the request is ambiguous. | Missing/unclear property, file, target, or conflicting user constraints. |
| `reject_unsupported` | Stop before execution because the request is impossible with configured tools. | Example: simultaneous multi-property optimization while excluding all multi-property tools. |

## Post-Run Actions

These actions happen after an optimizer run has produced measured status.

| Action | Meaning | Required Validation |
| --- | --- | --- |
| `stop_success` | Report final output because all scored outputs met the property target. | `final_target_status == "met"`. |
| `report_partial` | Report current best output because only some scored outputs met the target. | `final_target_status == "partial"` and no useful validated action remains or budget is exhausted. |
| `report_validated_only` | Report valid unique SMILES when property scoring is unavailable. | Rows exist, but numeric target status is unavailable. |
| `report_failure` | Report failure details and any partial/intermediate artifacts. | No optimized rows or all attempts failed. |
| `retry_same_tool` | Re-run the same tool with a meaningful variation. | Tool declares retry support, budget remains, and retry strategy changes something material. |
| `switch_tool` | Try another compatible tool. | Tool exists, is not excluded, has not already failed/tried for this request, and budget remains. |
| `ask_clarification` | Ask user to relax/change target, allow tools, or spend more compute. | No safe useful action remains or target appears unrealistic. |

## Target Status Contract

`final_target_status` is authoritative.

- `met`: all scored optimized outputs met the requested property target.
- `partial`: some, but not all, scored optimized outputs met the target.
- `none_met`: scored outputs exist, but none met the target.
- `validated_smiles_only`: outputs are valid unique SMILES, but target scoring is unavailable.
- `failed`: no usable optimized output was produced.

`requested_outputs_per_input` is per input molecule. If there are two input
molecules and the user requests five candidates, the expected total output count
is ten. Do not treat five target hits as full success unless the explicit output
selection policy says that is sufficient.

## Retry Strategy Options

`retry_same_tool` should only be allowed when a retry can change the result.
The policy records this as `legal_actions.retry_rule` with:

- `allowed`: whether retry is currently legal
- `reason`: why retry is allowed or blocked
- `requires_material_variation`: whether the retry must change seed, prompt,
  instruction schedule, budget, or another meaningful setting

| Tool Family | Useful Retry Variations | Notes |
| --- | --- | --- |
| GeLLMO-C | Instruction template index, sampling, temperature, prompt phrasing, requested batch size. | Useful for validity/diversity recovery and chemistry-plausible analog generation. |
| MolOpt | Seed change, oracle budget increase, patience/frequency settings when exposed. | Seed retry is useful for stochastic algorithms; budget increase is useful when the run was capped below benchmark guidance. |
| Deterministic/simple tools | Usually none. | Prefer `switch_tool`, `report_partial`, or `ask_clarification`. |

Do not retry for structural failures such as missing files, unsupported
objectives, impossible tool constraints, evaluator import failures, or repeated
runtime/model errors unless the retry strategy directly addresses the cause.

## Switch Strategy Options

`switch_tool` is preferred when:

- Another compatible tool has not been tried.
- The current tool produced `none_met` or weak `partial` target hits.
- The current tool cannot score the target but another tool can.
- The user allowed alternatives and the budget permits another run.

The policy records this as `legal_actions.switch_rule` with:

- `allowed`: whether switch is currently legal
- `reason`: why switch is allowed or blocked
- `candidate_tools`: untried non-excluded tools before target-specific filtering
- `legal_tools`: tools that remain legal after policy filtering
- `requires_property_scoring_tool`: whether the current numeric target requires
  the next tool to provide property scoring

For MolOpt, switching among algorithms may be more useful than repeating the
same algorithm when the only retry variation is minor. For GeLLMO-C, switching
to MolOpt can be useful when a numeric single-property target must be scored.
If the user explicitly requested GeLLMO-C and it returned the requested number
of valid unique SMILES, the safer default is to report `validated_smiles_only`
rather than silently switch tools. If GeLLMO-C returns partial valid output and
has retry budget, retrying with a fresh instruction/sampling schedule is legal.

## Metadata To Preserve

Each decision should keep enough information for audit and orchestrator use:

- phase: `pre_run` or `post_run`
- action type and selected tool
- deterministic/baseline recommendation
- LLM recommendation and raw response
- validated policy action
- allowed actions
- excluded tools
- target hits and scored outputs
- retry/switch strategy if executed
- output CSV and debug/intermediate CSVs
- reason for stopping

## Bounded Reasoning Limits

The LLM reasoning layer is advisory and bounded. It may recommend the next
action, but it cannot execute tools directly and cannot exceed configured
limits.

LeadOptimizer is configured with `max_tool_actions`, the total number of
optimizer/tool executions allowed for one `agent.run(...)` call. The initial
optimizer run counts as one tool action; retry/switch executions consume the
remaining tool-action budget. LLM planning, next-action advice, and final
response summarization are tracked separately and do not count as tool actions.

The run metadata records `reasoning_limits` with:

- `reasoning_mode`: `bounded_llm` or `disabled`
- `max_decision_rounds`: maximum post-run LLM advisor calls
- `max_tool_actions`: maximum total optimizer/tool executions
- `max_followup_tool_actions`: maximum retry/switch executions after the first run
- `decision_rounds_used`: advisor calls actually made
- `tool_actions_executed`: total optimizer/tool executions, including the initial run
- `followup_tool_actions_executed`: retry/switch executions actually made
- `stopped_by`: `policy_stop`, `decision_round_limit`,
  `tool_action_limit`, `disabled`, or `no_decision_trace`

If a limit stops further automatic exploration before the property target is
fully met, the response summary may include a brief bounded-reasoning note so
the user/orchestrator knows the run stopped by policy rather than because the
LLM chose to continue indefinitely.

## Deferred Decisions

These are intentionally later-phase items:

- Combining target hits from multiple tools into one final output set.
- Multi-objective MolOpt scoring beyond existing single-scalar oracles.
- Chemistry-plausibility grading beyond validity/dedup/filter checks.
- Automatic target relaxation.
- Long multi-turn autonomous planning like Synthesizer or Orchestrator.
