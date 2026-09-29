from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from flask import Flask, request, render_template_string

from synthesizer.agent.agent import Agent
from synthesizer.llm.openai_compatible import OpenAILLM
from synthesizer.tools.MEEA import MEEA
from synthesizer.tools.code import Code


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ENV_DIR = ROOT_DIR / "tmp" / "synthesizer-chat"
DEFAULT_MEEA_IMAGE = "/path/to/synthesizer-assets/meea.sif"
DEFAULT_PREXSYN_IMAGE = "/path/to/synthesizer-assets/prexsyn.sif"
DEFAULT_LARGEFILES = {
    "eMoleculesBB.txt": "/path/to/synthesizer-assets/eMoleculesBB.txt",
    "EnamineREAL.txt": "/path/to/synthesizer-assets/EnamineREAL.txt",
    "policy_model.ckpt": "/path/to/synthesizer-assets/policy_model.ckpt",
    "template_rules.dat": "/path/to/synthesizer-assets/template_rules.dat",
    "value_pc.pt": "/path/to/synthesizer-assets/value_pc.pt",
}
DEFAULT_PROMPT = (
    "Can you quickly find a synthesis path for the molecules in smiles.csv in 15 minutes? "
    "If a molecule is not synthesizable, please sample and return at least one synthesizable "
    "analog with 0.85 Tanimoto similarity to the original molecule "
    "(Morgan Fingerprint, radius 2, 2048 bit)."
)

HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Synthesizer Chat</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4efe6;
      --panel: #fffdf9;
      --ink: #1f2933;
      --muted: #5f6c7b;
      --line: #d8cfc2;
      --accent: #0f766e;
      --accent-2: #d97706;
      --error: #b91c1c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      background:
        radial-gradient(circle at top left, rgba(217, 119, 6, 0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(15, 118, 110, 0.16), transparent 35%),
        linear-gradient(180deg, #f7f3eb 0%, var(--bg) 100%);
      color: var(--ink);
    }
    .wrap {
      max-width: 1100px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: clamp(2rem, 4vw, 3.4rem);
      line-height: 1;
      letter-spacing: -0.04em;
    }
    .lede {
      max-width: 780px;
      margin: 0 0 24px;
      color: var(--muted);
      font-size: 1.05rem;
    }
    .grid {
      display: grid;
      gap: 20px;
      grid-template-columns: 1.1fr 0.9fr;
      align-items: start;
    }
    .card {
      background: color-mix(in srgb, var(--panel) 92%, white);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 12px 30px rgba(31, 41, 51, 0.06);
    }
    label {
      display: block;
      margin: 0 0 8px;
      font-weight: 700;
      font-size: 0.95rem;
    }
    .field { margin-bottom: 16px; }
    textarea, input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 14px;
      font: inherit;
      color: var(--ink);
      background: #fff;
    }
    textarea { min-height: 180px; resize: vertical; }
    .row {
      display: grid;
      gap: 14px;
      grid-template-columns: 1fr 160px;
    }
    button {
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      color: white;
      background: linear-gradient(135deg, var(--accent), #155e75);
    }
    .note, .status {
      font-size: 0.95rem;
      color: var(--muted);
      white-space: pre-wrap;
      word-break: break-word;
    }
    .error {
      color: var(--error);
      font-weight: 700;
      white-space: pre-wrap;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 0.9rem;
      line-height: 1.45;
    }
    .pill {
      display: inline-block;
      margin: 0 8px 8px 0;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(15, 118, 110, 0.08);
      color: var(--accent);
      font-size: 0.85rem;
      font-weight: 700;
    }
    @media (max-width: 860px) {
      .grid, .row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Synthesizer Chat</h1>
    <p class="lede">Quick browser UI for exercising the Synthesizer agent with the same prompt-and-environment flow shown in the README.</p>
    <div class="grid">
      <form method="post" class="card">
        <div class="field">
          <label for="prompt">Prompt</label>
          <textarea id="prompt" name="prompt">{{ prompt }}</textarea>
        </div>
        <div class="row">
          <div class="field">
            <label for="environment_path">Environment Directory</label>
            <input id="environment_path" name="environment_path" value="{{ environment_path }}">
          </div>
          <div class="field">
            <label for="turn_limit">Turn Limit</label>
            <input id="turn_limit" name="turn_limit" type="number" min="1" max="30" value="{{ turn_limit }}">
          </div>
        </div>
        <button type="submit">Run Synthesizer</button>
      </form>
      <div class="card">
        <div class="field">
          <label>Session Defaults</label>
          <div class="pill">MEEA image: {{ config.meea_image }}</div>
          <div class="pill">PrexSyn image: {{ config.prexsyn_image }}</div>
          <div class="pill">LLM model: {{ config.llm_model }}</div>
        </div>
        <div class="field">
          <label>Large Files</label>
          <pre>{{ config.largefiles }}</pre>
        </div>
      </div>
    </div>

    {% if error %}
    <div class="card" style="margin-top: 20px;">
      <label>Run Error</label>
      <div class="error">{{ error }}</div>
    </div>
    {% endif %}

    {% if response %}
    <div class="card" style="margin-top: 20px;">
      <label>Agent Response</label>
      <pre>{{ response }}</pre>
    </div>
    {% endif %}

    {% if metadata %}
    <div class="card" style="margin-top: 20px;">
      <label>Metadata</label>
      <pre>{{ metadata }}</pre>
    </div>
    {% endif %}

    <div class="card" style="margin-top: 20px;">
      <label>How To Use</label>
      <div class="status">Point the environment directory at a writable folder that contains files referenced in your prompt, such as <code>smiles.csv</code>. The agent will be told to read from and write results back into that same directory.</div>
    </div>
  </div>
</body>
</html>
"""

_agent_lock = threading.Lock()
_cached_agent: Agent | None = None


def _default_config() -> dict[str, object]:
    largefiles = dict(DEFAULT_LARGEFILES)
    return {
        "meea_image": os.environ.get("MEEA_IMAGE", DEFAULT_MEEA_IMAGE),
        "prexsyn_image": os.environ.get("PREXSYN_IMAGE", DEFAULT_PREXSYN_IMAGE),
        "llm_model": os.environ.get("LLM_MODELNAME", "gpt-5.4"),
        "largefiles": largefiles,
    }


def _build_agent() -> Agent:
    config = _default_config()
    os.environ.setdefault("MEEA_IMAGE", str(config["meea_image"]))
    os.environ.setdefault("PREXSYN_IMAGE", str(config["prexsyn_image"]))
    llm = OpenAILLM()
    tools = [
        MEEA(largefiles=config["largefiles"]),
        Code(),
    ]
    return Agent(tools, llm=llm)


def _get_agent() -> Agent:
    global _cached_agent
    with _agent_lock:
        if _cached_agent is None:
            _cached_agent = _build_agent()
        return _cached_agent


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def index():
        prompt = request.form.get("prompt", DEFAULT_PROMPT)
        environment_path = request.form.get("environment_path", str(DEFAULT_ENV_DIR))
        turn_limit_raw = request.form.get("turn_limit", "10")
        response_text = ""
        metadata_text = ""
        error_text = ""

        if request.method == "POST":
            try:
                turn_limit = max(1, int(turn_limit_raw))
                env_path = Path(environment_path).expanduser().resolve()
                env_path.mkdir(parents=True, exist_ok=True)
                agent = _get_agent()
                response, metadata = agent(prompt, str(env_path), turn_limit=turn_limit)
                response_text = response
                metadata_text = json.dumps(metadata, indent=2, default=str)
            except Exception as exc:
                error_text = f"{exc.__class__.__name__}: {exc}"

        return render_template_string(
            HTML,
            config={
                "meea_image": _default_config()["meea_image"],
                "prexsyn_image": _default_config()["prexsyn_image"],
                "llm_model": _default_config()["llm_model"],
                "largefiles": json.dumps(_default_config()["largefiles"], indent=2),
            },
            prompt=prompt,
            environment_path=environment_path,
            turn_limit=turn_limit_raw,
            response=response_text,
            metadata=metadata_text,
            error=error_text,
        )

    return app


if __name__ == "__main__":
    app = create_app()
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host=host, port=port, debug=False)
