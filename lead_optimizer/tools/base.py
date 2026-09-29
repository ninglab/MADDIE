from ..schemas import OptimizationRequest


class OptimizationTool:
    """Common interface for anything that can optimize one molecule request."""

    name = "optimization_tool"
    min_batch_size = 1
    include_retry_instruction = True
    supports_multi_property = False
    supports_adaptive_fallback = False
    supports_agent_retry = False
    supports_property_scoring = False

    def setup_tool(self):
        pass

    def documentation(self) -> str:
        return ""

    def optimize_request(self, request: OptimizationRequest) -> list[str]:
        raise NotImplementedError

    def evaluate(self, smiles: str, adjustment: str):
        """Return (property_name, value) when the tool has an evaluator."""
        return None

    def optimize(self, smiles: str, adjustment: str, num_outputs: int) -> list[str]:
        request = OptimizationRequest(
            input_id="manual",
            smiles=smiles,
            adjustment=adjustment,
            num_outputs=num_outputs,
        )
        return self.optimize_request(request)


class FakeOptimizerTool(OptimizationTool):
    """Tiny deterministic tool for local tests when GeLLMO-C is unavailable."""

    name = "fake_optimizer"

    def documentation(self) -> str:
        return "Fake local optimization tool for testing agent IO."

    def optimize_request(self, request: OptimizationRequest) -> list[str]:
        suffixes = ["C", "N", "O", "Cl", "F"]
        return [request.smiles + suffix for suffix in suffixes[:request.num_outputs]]


class GeLLMOOptimizerTool(OptimizationTool):
    """Adapter that lets DirectOptimizer call a configured GeLLMO-C model."""

    name = "gellmo_c"
    min_batch_size = 5
    include_retry_instruction = False
    supports_multi_property = True
    supports_agent_retry = True

    def __init__(
        self,
        gellmo_tool=None,
        retry_temperatures: tuple[float, ...] = (0.7, 0.8, 0.9),
        retry_attempt_offset: int = 0,
        sampling_temperature_override: float | None = None,
    ):
        if not retry_temperatures:
            raise ValueError("retry_temperatures must not be empty.")
        self.gellmo_tool = gellmo_tool
        self.retry_temperatures = retry_temperatures
        self.retry_attempt_offset = retry_attempt_offset
        self.sampling_temperature_override = sampling_temperature_override

    def documentation(self) -> str:
        return (
            "GeLLMO-C performs instruction-guided molecular edits. In this tool "
            "chest, use it for simultaneous multi-property optimization in one "
            "request and for requests emphasizing chemically plausible close "
            "analogs. Local smoke tests currently favor GeLLMO-C for hERG, "
            "ClinTox, mutagenicity, and carcinogenicity; DILI remains under "
            "testing. For benchmarked physicochemical single-property threshold "
            "optimization without a chemistry-reasonableness emphasis, use "
            "MolOpt evidence to choose an algorithm."
        )

    def setup_tool(self):
        if self.gellmo_tool is None:
            raise RuntimeError("GeLLMO-C tool is not configured.")

    def optimize_request(self, request: OptimizationRequest) -> list[str]:
        if self.gellmo_tool is None:
            raise RuntimeError("GeLLMO-C tool is not configured.")

        optimize_kwargs = dict(
            request=request.to_gellmo_text(),
            num_return_sequences=request.num_outputs,
        )
        prompter = getattr(self.gellmo_tool, "prompter", None)
        instructions = getattr(prompter, "instructions", None)
        instruction_count = len(instructions) if instructions else 1
        effective_attempt = request.attempt + self.retry_attempt_offset
        optimize_kwargs["instr_idx"] = (effective_attempt - 1) % instruction_count

        if self.sampling_temperature_override is not None:
            optimize_kwargs.update(
                do_sample=True,
                temperature=self.sampling_temperature_override,
            )
        elif effective_attempt > instruction_count:
            temperature_index = (
                effective_attempt - instruction_count - 1
            ) % len(self.retry_temperatures)
            optimize_kwargs.update(
                do_sample=True,
                temperature=self.retry_temperatures[temperature_index],
            )

        result = self.gellmo_tool.optimize(**optimize_kwargs)
        if isinstance(result, str):
            return [result]
        return list(result)


class ToolChest:
    """
    Tool registry for optimization tools.

    Stores and retrieves optimization tools by name. Provides inspection methods
    for planners and user code.

    Usage:
        chest = ToolChest()
        chest.add("gellmo_c", gellmo_tool)
        chest.add("molopt_screening", molopt_tool)

        # List available tools
        print(chest.available())

        # Get a tool
        tool = chest.get("gellmo_c")

        # Get hints for tool selection (for planners)
        hints = chest.tool_hints()  # {name → short description}
    """

    def __init__(self, tools: list[OptimizationTool] | None = None):
        self.tools = {}
        for tool in tools or []:
            self.add(tool.name, tool)

    def add(self, name: str, tool: OptimizationTool) -> None:
        """Add a tool to the chest."""
        tool.name = name  # Override tool name if needed
        self.tools[name] = tool

    def register(self, tool: OptimizationTool) -> None:
        """Backward-compatible alias for adding a tool by its own name."""
        self.add(tool.name, tool)

    def get(self, name: str) -> OptimizationTool:
        """Get a tool by name."""
        return self.tools[name]

    def available(self) -> list[str]:
        """List available tool names."""
        return list(self.tools.keys())

    def names(self) -> list[str]:
        """Backward-compatible alias for available()."""
        return self.available()

    def multi_property_tools(self) -> list[str]:
        """List tools that support simultaneous multi-property optimization."""
        return [
            name
            for name, tool in self.tools.items()
            if getattr(tool, "supports_multi_property", False)
        ]

    def property_scoring_tools(self) -> list[str]:
        """List tools that can numerically score optimization properties."""
        return [
            name
            for name, tool in self.tools.items()
            if (
                getattr(tool, "supports_property_scoring", False)
                or type(tool).evaluate is not OptimizationTool.evaluate
            )
        ]

    def tool_hints(self) -> dict[str, str]:
        """
        Return dict of tool_name → short description for planner.

        Used by SmartPlanner to understand what tools do and choose appropriately.
        """
        hints = {}
        for name, tool in self.tools.items():
            doc = tool.documentation()
            # Extract first line or first sentence
            first_line = doc.split("\n")[0] if doc else ""
            hints[name] = first_line[:100]  # Keep it short
        return hints

    def documentation(self) -> str:
        """Return formatted documentation of all tools."""
        lines = []
        for name, tool in self.tools.items():
            lines.append(f"- {name}: {tool.documentation()}")
        return "\n".join(lines)


ToolRegistry = ToolChest
