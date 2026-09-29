from abc import ABC, abstractmethod

class Automaton(ABC):
    """A background observer that, when its trigger condition is met,
    contributes a string to the action result inside _run_procedure.

    Lifecycle: start() once before the run loop, stop() once after.
    Each iteration the loop checks `.trigger`; if True it calls `.consume()`
    and appends the returned string to `result`.
    """

    name: str          # short identifier, used in the <AUTOMATON name> tag
    description: str    # one-line summary shown to the orchestrator in its context

    def start(self) -> None:
        """Begin background monitoring. Default: no-op for synchronous automatons."""

    def stop(self) -> None:
        """Tear down background monitoring. Default: no-op."""

    @property
    @abstractmethod
    def trigger(self) -> bool:
        """True if there is new output to emit since the last consume()."""

    @abstractmethod
    def consume(self) -> str:
        """Return accumulated output as a string and reset internal state."""

import os
import csv
import json
from collections import Counter
from watchdog.utils.dirsnapshot import DirectorySnapshot, DirectorySnapshotDiff


class DirectoryWatcherAutomaton(Automaton):
    name = "DirectoryWatcher"
    description = (
        "Reports files created, modified, moved, or deleted in the environment "
        "since your last action, as a compact directory tree."
    )

    # bounding so a large burst can't overwhelm the LLM
    MAX_FILES_PER_DIR = 5  # collapse a directory listing past this many files
    MIN_SIMILAR_DIRS = 3   # collapse sibling dirs sharing a structure past this many
    MAX_LINES = 20          # global cap on rendered tree lines
    MAX_PEEK_BYTES = 5_000_000  # skip content peek above this size
    _PEEK_COLS = 6          # max column/key names to show in a peek
    _EVENT_PRIORITY = {"created": 0, "deleted": 1, "moved": 2, "modified": 3}

    def __init__(self, path: str):
        self.path = path
        self._pending: list[tuple[str, str]] = []
        self._baseline: DirectorySnapshot | None = None

    def start(self) -> None:
        # take a baseline snapshot; changes are detected synchronously against it
        # at trigger/consume time, so a fast write can never slip through a poll
        # window. No background thread.
        self._pending = []
        self._baseline = DirectorySnapshot(self.path)

    def stop(self) -> None:
        self._baseline = None
        self._pending = []

    def _scan(self) -> None:
        """Diff the directory against the baseline, queue the changes, and advance
        the baseline. Synchronous — captures changes regardless of timing."""
        if self._baseline is None:
            return
        try:
            snapshot = DirectorySnapshot(self.path)
        except OSError:
            return  # transient walk error; baseline holds, retry next scan
        diff = DirectorySnapshotDiff(self._baseline, snapshot)
        self._baseline = snapshot
        for p in diff.files_created:
            self._pending.append(("created", self._to_str(p)))
        for p in diff.files_modified:
            self._pending.append(("modified", self._to_str(p)))
        for _src, dest in diff.files_moved:
            self._pending.append(("moved", self._to_str(dest)))
        for p in diff.files_deleted:
            self._pending.append(("deleted", self._to_str(p)))

    @staticmethod
    def _to_str(p) -> str:
        """Coerce a snapshot path (typed bytes | str) to str."""
        if isinstance(p, str):
            return p
        if isinstance(p, (bytes, bytearray, memoryview)):
            return bytes(p).decode()
        return str(p)

    @property
    def trigger(self) -> bool:
        # scans the directory (one walk) and caches changes for consume()
        self._scan()
        return bool(self._pending)

    # region formatting
    def _relpath(self, abs_path: str) -> str:
        """Path relative to the watched root (drops the long shared prefix)."""
        return os.path.relpath(abs_path, self.path)

    def _dedupe(self, events: list[tuple[str, str]]) -> dict[str, str]:
        """Collapse to one event per relative path, keeping the most significant
        (created > deleted > moved > modified). Preserves first-seen order."""
        result: dict[str, str] = {}
        for event_type, abs_path in events:
            rel = self._relpath(abs_path)
            prev = result.get(rel)
            if prev is None or (
                self._EVENT_PRIORITY.get(event_type, 99)
                < self._EVENT_PRIORITY.get(prev, 99)
            ):
                result[rel] = event_type
        return result

    @staticmethod
    def _summarize_files(files: list[tuple[str, str, str]]) -> str:
        """One-line summary for a collapsed directory: counts by event (only if
        mixed) plus the top extensions, e.g. '(48 files [created]: .out×24, .json×24)'."""
        n = len(files)
        events = Counter(event for _, event, _ in files)
        exts = Counter(os.path.splitext(name)[1] or "(none)" for name, _, _ in files)

        if len(events) == 1:
            ev_part = f"[{next(iter(events))}]"
        else:
            ev_part = ", ".join(f"{c} {ev}" for ev, c in events.most_common())

        top = exts.most_common(3)
        ext_part = ", ".join(f"{ext}×{c}" for ext, c in top)
        remaining = len(exts) - len(top)
        if remaining > 0:
            ext_part += f", +{remaining} more"
        return f"({n} files {ev_part}: {ext_part})"

    @classmethod
    def _fmt_names(cls, names: list[str]) -> str:
        """Comma-joined names, capped at _PEEK_COLS with a '+N' overflow tag."""
        shown = ", ".join(names[: cls._PEEK_COLS])
        extra = len(names) - cls._PEEK_COLS
        return f"{shown}, +{extra}" if extra > 0 else shown

    def _peek(self, rel: str, event: str) -> str:
        """Short content summary for a file leaf (csv/tsv/json). Returns '' when
        there is nothing useful to add or anything goes wrong (e.g. a file still
        being written). Never raises."""
        if event == "deleted":
            return ""  # file is gone, nothing to read
        abs_path = os.path.join(self.path, rel)
        ext = os.path.splitext(rel)[1].lower()
        try:
            if os.path.getsize(abs_path) > self.MAX_PEEK_BYTES:
                return ""
            if ext in (".csv", ".tsv"):
                return self._peek_csv(abs_path, "\t" if ext == ".tsv" else ",")
            if ext == ".json":
                return self._peek_json(abs_path)
        except Exception:
            return ""
        return ""

    def _peek_csv(self, abs_path: str, delimiter: str) -> str:
        with open(abs_path, newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)
            try:
                header = next(reader)
            except StopIteration:
                return "empty"
            rows = sum(1 for _ in reader)  # data rows (header excluded)
        return f"{rows} rows × {len(header)} cols: {self._fmt_names(header)}"

    def _peek_json(self, abs_path: str) -> str:
        with open(abs_path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            keys = list(data.keys())
            return f"object, {len(keys)} keys: {self._fmt_names(keys)}"
        if isinstance(data, list):
            n = len(data)
            if n and isinstance(data[0], dict):
                return f"array[{n}], item keys: {self._fmt_names(list(data[0].keys()))}"
            return f"array[{n}]"
        return type(data).__name__

    @classmethod
    def _signature(cls, node: dict):
        """Hashable structural fingerprint of a dir node: its (file name, event)
        set plus the (name, signature) of each child dir. Sibling dirs with the
        same signature are 'identical' and can be collapsed to one representative."""
        files = frozenset((name, ev) for name, ev, _ in node["files"])
        dirs = frozenset(
            (name, cls._signature(child)) for name, child in node["dirs"].items()
        )
        return files, dirs

    def _build_tree(self, path_to_event: dict[str, str]) -> str:
        """Render a compact directory tree from {rel_path: event}, collapsing
        single-child dir chains and busy directories, capped at MAX_LINES."""
        # nested structure: node = {"dirs": {name: node}, "files": [(name, event, rel)]}
        root: dict = {"dirs": {}, "files": []}
        for rel, event in path_to_event.items():
            parts = rel.split(os.sep)
            node = root
            for part in parts[:-1]:
                node = node["dirs"].setdefault(part, {"dirs": {}, "files": []})
            node["files"].append((parts[-1], event, rel))

        lines: list[str] = []

        def collapse_chain(name: str, child: dict) -> tuple[str, dict]:
            # collapse single-child dir chains: a/b/c/
            display = name
            while not child["files"] and len(child["dirs"]) == 1:
                only = next(iter(child["dirs"]))
                display = f"{display}/{only}"
                child = child["dirs"][only]
            return display, child

        def render_body(child: dict, child_prefix: str) -> None:
            child_files = child["files"]
            if len(child_files) > self.MAX_FILES_PER_DIR:
                lines.append(f"{child_prefix}{self._summarize_files(child_files)}")
                # still recurse into subdirectories of the collapsed dir
                render({"dirs": child["dirs"], "files": []}, child_prefix)
            else:
                render(child, child_prefix)

        def render(node: dict, prefix: str) -> None:
            # group sibling dirs by structural signature, preserving first-seen order
            entries = sorted(node["dirs"].keys())
            sigs = {n: self._signature(node["dirs"][n]) for n in entries}
            groups: dict = {}
            for n in entries:
                groups.setdefault(sigs[n], []).append(n)

            # flatten into an ordered list of renderables so connectors are simple.
            # files come first (higher print priority), then directories.
            renderables: list = []
            for name, event, rel in sorted(node["files"]):
                renderables.append(("file", name, event, rel))
            seen: set = set()
            for n in entries:
                sig = sigs[n]
                if sig in seen:
                    continue
                seen.add(sig)
                group = groups[sig]
                if len(group) >= self.MIN_SIMILAR_DIRS:
                    renderables.append(("repdir", group[0], len(group)))
                    renderables.append(("more", group[0], len(group) - 1))
                else:
                    renderables.extend(("dir", name, 0) for name in group)

            last = len(renderables) - 1
            for idx, item in enumerate(renderables):
                is_last = idx == last
                connector = "└── " if is_last else "├── "
                child_prefix = prefix + ("    " if is_last else "│   ")
                kind = item[0]
                if kind == "file":
                    _, name, event, rel = item
                    peek = self._peek(rel, event)
                    suffix = f"  {peek}" if peek else ""
                    lines.append(f"{prefix}{connector}{name} [{event}]{suffix}")
                elif kind == "more":
                    _, example, extra = item
                    lines.append(f"{prefix}{connector}… {extra} more like {example}/")
                else:  # "dir" or "repdir"
                    name = item[1]
                    display, child = collapse_chain(name, node["dirs"][name])
                    tag = f"   (1 of {item[2]} similar dirs)" if kind == "repdir" else ""
                    lines.append(f"{prefix}{connector}{display}/{tag}")
                    render_body(child, child_prefix)

        render(root, "")

        if len(lines) > self.MAX_LINES:
            hidden = len(lines) - self.MAX_LINES
            lines = lines[: self.MAX_LINES]
            lines.append(f"... and {hidden} more change(s)")
        return "\n".join(lines)

    def consume(self) -> str:
        # drain changes queued by the most recent trigger; call trigger first
        # (the orchestrator does: `if automaton.trigger: automaton.consume()`).
        events = self._pending
        self._pending = []
        if not events:
            return ""
        deduped = self._dedupe(events)
        root = os.path.basename(self.path.rstrip("/"))
        header = f"{len(deduped)} change(s) in {root}/"
        return header + "\n" + self._build_tree(deduped)

    # endregion


if __name__ == "__main__":
    import time
    path_to_dir = "/path/to/MADDIE/test_envs/temp"
    dir_watcher = DirectoryWatcherAutomaton(path_to_dir)
    dir_watcher.start()
    try:
        while True:
            if dir_watcher.trigger:
                print(dir_watcher.consume())
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        dir_watcher.stop()
