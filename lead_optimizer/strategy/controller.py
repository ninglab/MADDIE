from collections.abc import Callable


EXECUTABLE_POLICY_ACTIONS = {"retry_same_tool", "switch_tool"}


class DecisionController:
    """Bounded controller loop for policy-approved agent actions.

    The controller does not know how to run chemistry tools. It asks for a
    decision, validates whether the policy action is executable, and delegates
    execution to the caller-provided callback.
    """

    def __init__(
        self,
        advisor,
        max_decision_rounds: int = 1,
        execute_next_actions: bool = False,
        max_agent_steps: int = 1,
        output_selection_policy: str = "best_attempt",
    ):
        self.advisor = advisor
        self.max_decision_rounds = max(0, max_decision_rounds)
        self.execute_next_actions = execute_next_actions
        self.max_agent_steps = max(0, max_agent_steps)
        self.output_selection_policy = output_selection_policy

    def run(
        self,
        context_builder: Callable[[], dict],
        action_executor: Callable[[dict, int], dict],
    ) -> list[dict]:
        trace = []
        executed_steps = 0

        for step in range(1, self.max_decision_rounds + 1):
            context = context_builder()
            remaining_tool_steps = max(0, self.max_agent_steps - executed_steps)
            context.update({
                "execution_enabled": self.execute_next_actions,
                "remaining_tool_steps": remaining_tool_steps,
                "remaining_strategy_steps": max(
                    0,
                    self.max_decision_rounds - step + 1,
                ),
            })
            decision = self.advisor.recommend(context)
            decision.update({
                "step": step,
                "phase": "post_run",
                "executed": False,
                "execution_enabled": self.execute_next_actions,
                "max_agent_steps": self.max_agent_steps,
                "remaining_tool_steps": remaining_tool_steps,
                "output_selection_policy": self.output_selection_policy,
            })

            policy_action = decision.get("policy_action")
            can_execute = (
                self.execute_next_actions
                and executed_steps < self.max_agent_steps
                and policy_action in EXECUTABLE_POLICY_ACTIONS
            )
            if can_execute:
                outcome = action_executor(decision, step)
                decision["executed"] = bool(outcome.get("executed"))
                decision["execution_result"] = outcome
                if decision["executed"]:
                    executed_steps += 1
                    trace.append(decision)
                    continue

            trace.append(decision)
            break

        return trace
