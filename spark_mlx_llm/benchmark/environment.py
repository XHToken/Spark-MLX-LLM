"""Read-only macOS idle checks and interference checks for active benchmarks.

macmon's *active* ratios measure busy time. Its similarly named usage fields
are frequency-scaled ratios and must not be used for the idle gate. Missing or
unrecognised measurements fail closed. Sampling never stops another process.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "BenchmarkEnvironment",
    "IdleMonitor",
    "active_problems",
    "collect_snapshot",
]


def _command(arguments: list[str]) -> str:
    return subprocess.run(
        arguments,
        capture_output=True,
        text=True,
        check=True,
        timeout=8,
        env={**os.environ, "LC_ALL": "C"},
    ).stdout


def parse_macmon(output: str) -> dict[str, Any]:
    """Validate the installed macmon JSON schema; never guess ratio units."""
    rows = [json.loads(line) for line in output.splitlines() if line.strip()]
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError("macmon must return exactly one JSON object")
    row = rows[0]
    values = {}
    for source, target in (
        ("cpu_active_ratio", "cpu_percent"),
        ("gpu_active_ratio", "gpu_percent"),
    ):
        value = row.get(source)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"macmon {source} is missing or not a finite 0–1 ratio")
        values[target] = value * 100
    return {
        **values,
        "macmon_timestamp": row.get("timestamp"),
        "memory": row.get("memory"),
        "temperatures": row.get("temp"),
        "gpu_freq_mhz": row.get("gpu_freq_mhz"),
        "cpu_freq_mhz": row.get("pcpu_freq_mhz"),
        "gpu_power": row.get("gpu_power"),
        "fans": row.get("fans"),
    }


def parse_vm_stat(output: str) -> tuple[int, int]:
    page_size = re.search(r"page size of (\d+) bytes", output)
    swapouts = re.search(r"^Swapouts:\s*(\d+)\.?\s*$", output, re.MULTILINE)
    if not page_size or not swapouts or int(page_size[1]) <= 0:
        raise ValueError("vm_stat does not contain page size and Swapouts")
    return int(swapouts[1]), int(page_size[1])


def parse_processes(output: str) -> list[dict[str, Any]]:
    processes = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.split(None, 3)
        if len(fields) != 4:
            raise ValueError("unrecognised ps output")
        pid, ppid, cpu = int(fields[0]), int(fields[1]), float(fields[2])
        if pid <= 0 or ppid < 0 or not math.isfinite(cpu) or cpu < 0:
            raise ValueError("invalid ps process metrics")
        processes.append(
            {"pid": pid, "ppid": ppid, "cpu_percent": cpu, "command": fields[3]}
        )
    if not processes:
        raise ValueError("ps returned no processes")
    return processes


def process_tree_pids(processes: list[dict[str, Any]], roots: set[int]) -> set[int]:
    """Return roots and their descendants, even when ps is not parent-sorted."""
    excluded = set(roots)
    while True:
        children = {row["pid"] for row in processes if row["ppid"] in excluded}
        if children <= excluded:
            return excluded
        excluded.update(children)


def _module_kind(module: str) -> str | None:
    module = module.lower()
    if module in {"llamacpp_spark.benchmark", "llamacpp_spark._benchmark_worker"}:
        return "llama inference/benchmark"
    if module in {
        "mlx_spark.bench",
        "spark_mlx_llm.commands.generate",
        "spark_mlx_llm.commands.chat",
        "spark_mlx_llm.commands.server",
        "spark_mlx_llm.benchmark.bench",
        "spark_mlx_llm.benchmark.profile_metal",
    } or module.startswith(("mlx_lm.", "mlx_vlm.", "mlx_spark.bench.")):
        return "MLX inference/training/benchmark"
    return None


def _entry_kind(
    entry: str, arguments: Iterator[str], launch_paths: list[str]
) -> str | None:
    """Classify an executable/script; its data arguments are not launch evidence."""
    name = Path(entry).name.lower()
    if name in {"llama-bench", "llama-cli", "llama-server", "_benchmark-worker"}:
        return "llama inference/benchmark"
    if name == "watch_models.py":
        return "benchmark queue"
    if name in {"llamacpp-spark", "llamacpp_spark"}:
        task = next(arguments, None)
        if task == "_benchmark-worker":
            return "llama inference/benchmark"
        if task in {"benchmark", "watch"}:
            return "llama benchmark queue"
        return None
    if name in {"spark-mlx-generate", "spark-mlx-chat", "spark-mlx-server"}:
        return "MLX inference/training/benchmark"
    # MLX also installs console entry points such as mlx_lm.generate.
    module_kind = _module_kind(name)
    if module_kind:
        return module_kind
    tasks = {
        "generate",
        "train",
        "finetune",
        "lora",
        "server",
        "benchmark",
        "bench",
        "watch",
        "_benchmark-worker",
    }
    script_tasks = {
        "generate.py",
        "train.py",
        "finetune.py",
        "lora.py",
        "server.py",
        "benchmark.py",
        "watch.py",
        "watch.sh",
    }
    if name in {"mlx-spark", "mlx_spark", "mlx-lm", "mlx_lm", "mlx-vlm", "mlx_vlm"}:
        return (
            "MLX inference/training/benchmark"
            if next(arguments, None) in tasks
            else None
        )
    paths = [*launch_paths, entry]
    if any("mlx" in path.lower() for path in paths) and (
        name in tasks | script_tasks
        or re.match(r"(?:bench|benchmark|train|infer|generate)[_.-]", name)
    ):
        return "MLX inference/training/benchmark"
    if name in {"watch.sh", "run.sh"} and any(
        "llamacpp_spark" in path.lower() for path in paths
    ):
        return "llama benchmark queue"
    return None


def process_kind(command: str) -> str | None:
    """Recognise workload entry points, even at zero current CPU usage.

    ps command text is not lossless argv: quoting and argument boundaries can be
    missing. Read only a recognisable launch prefix, without interpreting shell
    -c/-lc source (including ps's literal \\012 newlines) or Python inline code.
    A workload launched by that source is detected as its own child process.
    """
    lexer = shlex.shlex(command, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    arguments = iter(lexer)
    try:
        entry = next(arguments)
        # Unwrap only supported launchers/options; never search later arguments
        # for something that happens to look like an executable or module.
        for _ in range(8):
            name = Path(entry).name.lower()
            if name == "env":
                entry = next(arguments)
                while True:
                    if entry == "--":
                        entry = next(arguments)
                        break
                    if entry in {"-i", "--ignore-environment"} or re.match(
                        r"^[A-Za-z_][A-Za-z0-9_]*=", entry
                    ):
                        entry = next(arguments)
                    elif entry in {"-u", "--unset", "-C", "--chdir"}:
                        next(arguments)
                        entry = next(arguments)
                    elif entry.startswith(("--unset=", "--chdir=")):
                        entry = next(arguments)
                    elif entry.startswith("-"):
                        return None
                    else:
                        break
                continue
            if name == "uv":
                if next(arguments) != "run":
                    return None
                entry = next(arguments)
                flags = {
                    "--locked",
                    "--frozen",
                    "--no-sync",
                    "--offline",
                    "--no-project",
                    "--isolated",
                    "--active",
                }
                values = {
                    "--directory",
                    "--project",
                    "--python",
                    "--package",
                    "--with",
                    "--with-editable",
                    "--with-requirements",
                }
                while entry.startswith("-"):
                    if entry == "--":
                        entry = next(arguments)
                        break
                    if entry in values:
                        next(arguments)
                    elif entry not in flags and entry.partition("=")[0] not in values:
                        return None
                    entry = next(arguments)
                continue
            if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?w?", name):
                script = next(arguments)
                while script.startswith("-"):
                    if script == "--":
                        script = next(arguments)
                        break
                    if script in {"-", "-c"} or script.startswith("-c"):
                        return None
                    if script == "-m":
                        return _module_kind(next(arguments))
                    if script.startswith("-m"):
                        return _module_kind(script[2:])
                    if script in {"-W", "-X", "--check-hash-based-pycs"}:
                        next(arguments)
                    elif not (
                        re.fullmatch(r"-[bBdEhiIOPqRsSuvVx]+", script)
                        or script.startswith(("-W", "-X", "--check-hash-based-pycs="))
                    ):
                        return None
                    script = next(arguments)
                return _entry_kind(script, arguments, [entry])
            if name in {"sh", "bash", "zsh", "dash", "ksh", "fish"}:
                script = next(arguments)
                while script.startswith("-"):
                    if script == "--":
                        script = next(arguments)
                        break
                    # In particular, -lc introduces opaque source, not a script.
                    if script in {"-", "--command"} or (
                        not script.startswith("--")
                        and any(flag in script[1:] for flag in "cs")
                    ):
                        return None
                    if script not in {
                        "--noprofile",
                        "--norc",
                        "--login",
                    } and not re.fullmatch(r"-[efilnuvx]+", script):
                        return None
                    script = next(arguments)
                return _entry_kind(script, arguments, [entry])
            return _entry_kind(entry, arguments, [])
    except (StopIteration, ValueError):
        # Truncated/ambiguous launch prefixes must not fall back to scanning the
        # rest of the command, which can contain arbitrary quoted source text.
        return None
    return None


def _thermal_state() -> int:
    """Read current NSProcessInfo.thermalState without importing a GUI package."""
    if sys.platform != "darwin":
        raise RuntimeError("thermal monitoring requires macOS")
    ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
    objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    message_object = ctypes.CFUNCTYPE(
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    )(("objc_msgSend", objc))
    message_integer = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(
        ("objc_msgSend", objc)
    )
    process_info = message_object(
        objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo")
    )
    if not process_info:
        raise RuntimeError("NSProcessInfo is unavailable")
    return message_integer(process_info, objc.sel_registerName(b"thermalState"))


def parse_thermal(
    output: str, state: int, cpu_count: int | None = None
) -> dict[str, Any]:
    states = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}
    if state not in states:
        raise ValueError("unknown NSProcessInfo thermal state")
    fields = {
        name.lower(): int(value)
        for name, value in re.findall(
            r"(CPU_Speed_Limit|CPU_Scheduler_Limit|CPU_Available_CPUs|Thermal_Level|Thermal_Warning_Level|Performance_Warning_Level)\s*=\s*(\d+)",
            output,
            re.IGNORECASE,
        )
    }
    if not fields and "No thermal warning level has been recorded" not in output:
        raise ValueError("unrecognised pmset thermal output")
    limited = state >= 2
    limited |= any(
        fields.get(name, 100) < 100
        for name in ("cpu_speed_limit", "cpu_scheduler_limit")
    )
    limited |= any(
        fields.get(name, 0) > 0
        for name in (
            "thermal_level",
            "thermal_warning_level",
            "performance_warning_level",
        )
    )
    if cpu_count is not None and "cpu_available_cpus" in fields:
        limited |= fields["cpu_available_cpus"] < cpu_count
    return {"state": states[state], "limited": bool(limited), "limits": fields}


def collect_snapshot() -> dict[str, Any]:
    """Collect independently so a failed sensor cannot hide other diagnostics."""
    snapshot: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cpu_percent": None,
        "gpu_percent": None,
        "memory_pressure": "unknown",
        "swapouts": None,
        "page_size": None,
        "thermal": None,
        "processes": [],
        "collection_errors": [],
    }

    def collect(name, callback):
        try:
            snapshot.update(callback())
        except (
            OSError,
            RuntimeError,
            ValueError,
            KeyError,
            TypeError,
            subprocess.SubprocessError,
        ) as error:
            snapshot["collection_errors"].append(f"{name}: {error}")

    def macmon():
        binary = shutil.which("macmon")
        if not binary:
            raise RuntimeError("macmon is not installed")
        return parse_macmon(
            _command([binary, "pipe", "--samples", "1", "--interval", "1000"])
        )

    def pressure():
        value = int(
            _command(
                ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"]
            ).strip()
        )
        if value not in (1, 2, 4):
            raise ValueError(f"unknown memory pressure level {value}")
        return {"memory_pressure": {1: "normal", 2: "warning", 4: "critical"}[value]}

    def virtual_memory():
        output = _command(["/usr/bin/vm_stat"])
        swapouts, page_size = parse_vm_stat(output)
        match = re.search(r"^Swapins:\s*(\d+)\.?\s*$", output, re.MULTILINE)
        if match is None:
            raise ValueError("vm_stat does not contain Swapins")
        return {"swapouts": swapouts, "swapins": int(match[1]), "page_size": page_size}

    collect("macmon", macmon)
    collect("memory pressure", pressure)
    collect("swapouts", virtual_memory)
    collect(
        "thermal",
        lambda: {
            "thermal": parse_thermal(
                _command(["/usr/bin/pmset", "-g", "therm"]),
                _thermal_state(),
                os.cpu_count(),
            )
        },
    )
    collect(
        "processes",
        lambda: {
            "processes": parse_processes(
                _command(["/bin/ps", "-axo", "pid=,ppid=,pcpu=,command="])
            )
        },
    )
    return snapshot


def active_problems(
    snapshot: dict[str, Any], *, allowed_pids: set[int] | None = None
) -> list[str]:
    """Problems during our own workload; deliberately ignore total CPU/GPU use.

    Swap-outs interrupt active work after two increasing samples or >=64 MiB
    in one sample. Idle admission is stricter and requires zero new swap-outs.
    Extra allowed PIDs and their descendants can be excluded after sampling.
    """
    reasons = list(snapshot.get("collection_errors", []))
    if snapshot.get("memory_pressure") != "normal":
        reasons.append(
            f"memory pressure is {snapshot.get('memory_pressure', 'unknown')}"
        )
    thermal = snapshot.get("thermal")
    if not isinstance(thermal, dict) or not isinstance(thermal.get("limited"), bool):
        reasons.append("thermal status is unavailable")
    elif thermal["limited"]:
        reasons.append(f"thermal/performance restriction: {thermal['state']}")
    delta = snapshot.get("swapouts_delta")
    if delta is None:
        reasons.append("swap-out baseline is unavailable")
    elif (
        snapshot.get("swapout_streak", 0) >= 2
        or snapshot.get("swapout_bytes_delta", 0) >= 64 * 1024 * 1024
    ):
        reasons.append(f"sustained or large swap-out increase: {delta} pages")
    parents = snapshot.get("process_parents", {})
    rows = [{"pid": int(pid), "ppid": parent} for pid, parent in parents.items()]
    allowed = process_tree_pids(rows, set(allowed_pids or ()))
    for process in snapshot.get("competing_processes", []):
        if process["pid"] not in allowed:
            reasons.append(f"competing {process['kind']}: PID {process['pid']}")
    return list(dict.fromkeys(reasons))


class IdleMonitor:
    def __init__(self, cpu_threshold: float = 15.0, gpu_threshold: float = 10.0):
        for value in (cpu_threshold, gpu_threshold):
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 < value <= 100
            ):
                raise ValueError(
                    "idle thresholds must be finite percentages in (0, 100]"
                )
        self.cpu_threshold = cpu_threshold
        self.gpu_threshold = gpu_threshold
        self._previous_swapouts: int | None = None
        self._swapout_streak = 0
        self._high_cpu_streaks: dict[int, int] = {}

    def sample(self, exclude_pids: set[int] | None = None) -> dict[str, Any]:
        snapshot = collect_snapshot()
        processes = snapshot.pop("processes")
        excluded = process_tree_pids(processes, {os.getpid(), *(exclude_pids or ())})
        snapshot["excluded_pids"] = sorted(excluded)
        snapshot["process_parents"] = {
            str(row["pid"]): row["ppid"] for row in processes
        }
        competitors, cpu_streaks = [], {}
        for process in processes:
            if process["pid"] in excluded:
                continue
            kind = process_kind(process["command"])
            if process["cpu_percent"] > 100:
                streak = self._high_cpu_streaks.get(process["pid"], 0) + 1
                cpu_streaks[process["pid"]] = streak
                if streak >= 2:
                    kind = kind or "high-CPU process"
            if kind:
                competitors.append({**process, "kind": kind})
        self._high_cpu_streaks = cpu_streaks
        snapshot["competing_processes"] = competitors
        swapouts = snapshot["swapouts"]
        previous = self._previous_swapouts
        delta = (
            swapouts - previous
            if swapouts is not None and previous is not None and swapouts >= previous
            else None
        )
        self._previous_swapouts = swapouts
        self._swapout_streak = (
            self._swapout_streak + 1 if delta is not None and delta > 0 else 0
        )
        snapshot.update(
            swapouts_delta=delta,
            swapout_streak=self._swapout_streak,
            swapout_bytes_delta=delta * snapshot["page_size"]
            if delta is not None
            else None,
            cpu_threshold=self.cpu_threshold,
            gpu_threshold=self.gpu_threshold,
        )
        reasons = active_problems(snapshot)
        if delta is not None and delta > 0:
            reasons.append(f"swap-outs increased by {delta} pages")
        for field, threshold in (
            ("cpu_percent", self.cpu_threshold),
            ("gpu_percent", self.gpu_threshold),
        ):
            value = snapshot[field]
            if value is None:
                reasons.append(f"{field} is unavailable")
            elif value >= threshold:
                reasons.append(f"{field} {value:.2f}% is not below {threshold:g}%")
        snapshot["reasons"] = list(dict.fromkeys(reasons))
        snapshot["idle"] = not snapshot["reasons"]
        return snapshot


class BenchmarkEnvironment:
    def __init__(self, report, *, timeout, monitor=None):
        self.report = report
        self.timeout = timeout
        self.monitor = monitor if monitor is not None else IdleMonitor()

    def sample(self, label):
        sample = self.monitor.sample()
        sample["label"] = label
        self.report.append("environment.jsonl", sample)
        return sample

    def wait_idle(self, label):
        deadline = time.monotonic() + self.timeout
        consecutive = 0
        last_notice = 0
        while True:
            state = self.sample(label)
            consecutive = consecutive + 1 if state["idle"] else 0
            if consecutive >= 2:
                return state
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Idle check timed out: " + "; ".join(state["reasons"])
                )
            if time.monotonic() - last_notice >= 15:
                reason = (
                    "; ".join(state["reasons"]) or "confirming a second idle sample"
                )
                print("Waiting for idle GPU/CPU: " + reason, flush=True)
                last_notice = time.monotonic()
            time.sleep(1)
