"""CLI of the FIM benchmark.

Examples:
  # the server is already running on the port
  python -m fim_bench --model Qwen2.5-Coder-3B-nf4-sym-g-1-se-awq

  # let the bench start and stop the server itself (uses _bench_model.bat)
  python -m fim_bench --model Mellum-4b-dpo-all-nf4-sym-g-1-se-awq --launch

  # several models in a row, report into a file
  python -m fim_bench --model A --model B --launch --report reports/fim.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from fim_bench.client import get_model_ids, wait_ready
from fim_bench.models import MODELS, model_info
from fim_bench.runner import format_report, run_bench

REPO = Path(__file__).resolve().parent.parent
LAUNCHER = Path(__file__).resolve().parent / "_bench_model.bat"
LOGS = REPO / "logs"


def launch_server(model: str, port: int, extra_args: list[str] | None = None) -> subprocess.Popen:
    """Start the model server through the launcher bat, return its handle."""
    LOGS.mkdir(exist_ok=True)
    log_path = LOGS / f"bench-{model}.log"
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        cmd = [str(LAUNCHER), model, *(extra_args or [])]
        print(f"[bench]   command: {' '.join(cmd)}")
        return subprocess.Popen(
            [str(LAUNCHER), model, *(extra_args or [])],
            cwd=str(REPO),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )


def stop_server(handle: subprocess.Popen, port: int) -> None:
    """Kill the server process tree of a launcher handle, verify the port is freed."""
    subprocess.run(["taskkill", "/PID", str(handle.pid), "/T", "/F"],
                   capture_output=True, check=False)
    time.sleep(2.0)
    if get_model_ids(port, timeout_s=2.0):
        print(f"[bench] WARNING: port {port} is still serving after the stop - a stale server survives, "
              f"kill it manually (Get-Process python* | Stop-Process -Force)")


def append_results_log(results_log: Path, model: str, label: str, results) -> None:
    """Append case results of one run into the shared JSONL log."""
    from datetime import datetime, timezone

    results_log.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with results_log.open("a", encoding="utf-8") as log_file:
        for res in results:
            entry = {
                "timestamp": stamp,
                "model": model,
                "label": label,
                "case": res.case,
                "passed": res.passed,
                "seconds": round(res.seconds, 2),
                "expected": res.expected,
                "actual": res.output,
            }
            if res.error:
                entry["error"] = res.error
            log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="fim_bench", description="FIM model comparison benchmark")
    parser.add_argument("--model", action="append", required=True,
                        help="model id (dir name in models/); repeatable")
    parser.add_argument("--port", type=int, default=8887, help="agent server port (default 8887)")
    parser.add_argument("--launch", action="store_true",
                        help=f"start/stop the server per model via {LAUNCHER.name}")
    parser.add_argument("--server-arg", action="append", default=[],
                        help="extra agent argument passed to the launcher (repeatable), "
                             "e.g. --server-arg=--npu_prefill_hint --server-arg=DYNAMIC")
    parser.add_argument("--ready-timeout", type=float, default=900.0,
                        help="seconds to wait for the server to become ready")
    parser.add_argument("--report", type=Path, default=None,
                        help="optional path of the markdown report file")
    parser.add_argument("--label", type=str, default="default",
                        help="run label (e.g. prefill mode) stored in the shared results log")
    parser.add_argument("--warmup", action="store_true",
                        help="send a throwaway request before the measured cases")
    parser.add_argument("--results-log", type=Path, default=REPO / "reports" / "fim-results.jsonl",
                        help="shared JSONL log of all case results for the summary table")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    blocks: list[str] = []
    for model in args.model:
        info = model_info(model)
        handle: subprocess.Popen | None = None
        if args.launch:
            extra = " ".join(args.server_arg) if args.server_arg else "-"
            print(f"[bench] starting server for {model}")
            print(f"[bench]   port={args.port} label={args.label} launcher={LAUNCHER.name}")
            print(f"[bench]   server args: {extra}")
            print(f"[bench]   log: {LOGS / ('bench-' + model + '.log')}")
            handle = launch_server(model, args.port, args.server_arg)
            if not wait_ready(args.port, timeout_s=args.ready_timeout):
                print(f"[bench] server for {model} did not become ready, skipping")
                stop_server(handle, args.port)
                blocks.append(f"=== {model} ===\nSERVER NOT READY")
                continue
            served = get_model_ids(args.port)
            if info.name not in served:
                # The port is answered by a stale server of another model:
                # the fresh instance could not bind and died (WinError 10048).
                print(f"[bench] ERROR: port {args.port} serves {served}, expected '{info.name}'. "
                      f"A stale server is running - kill it and rerun.")
                stop_server(handle, args.port)
                blocks.append(f"=== {model} ===\nWRONG SERVER ON PORT: {served}")
                continue
        try:
            results = run_bench(info, args.port, warmup=args.warmup)
        finally:
            if handle:
                stop_server(handle, args.port)
                time.sleep(3.0)
        block = format_report(info, results)
        print(block)
        blocks.append(block)
        append_results_log(args.results_log, info.name, args.label, results)
        print(f"[bench] results appended to {args.results_log}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
        print(f"[bench] report written to {args.report}")
    known = ", ".join(sorted(MODELS))
    print(f"[bench] known models: {known}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
