"""The single command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence, TextIO

from .artifacts import ArtifactError, RunArtifacts
from .config import ConfigError, load_config
from .data import DataError
from .engine import (
    EngineError,
    doctor,
    evaluate_suite,
    freeze_protocol,
    generate_report,
    mark_interrupted,
    plan_case,
    prepare,
    run_synthetic_suite,
    smoke,
    train_dynamics,
    train_outcome,
)
from .planner import PlannerError
from .outcome import OutcomeError
from .policy import PolicyError
from .world import WorldModelError


class _Tee:
    """Minimal text stream that mirrors CLI output to terminal and a log file."""

    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return False


@contextmanager
def _training_log(
    config: dict, run_name: str, operation: str
) -> Iterator[Path]:
    log_root = Path(config["paths"]["project_root"]) / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{run_name}.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as handle:
        with redirect_stdout(_Tee(sys.stdout, handle)), redirect_stderr(
            _Tee(sys.stderr, handle)
        ):
            started = datetime.now(timezone.utc).isoformat()
            print(
                f"[{started}] cloop {operation} started; log={log_path}",
                file=sys.stderr,
            )
            try:
                yield log_path
            except BaseException as exc:
                failed = datetime.now(timezone.utc).isoformat()
                print(
                    f"[{failed}] cloop {operation} failed: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                raise
            else:
                finished = datetime.now(timezone.utc).isoformat()
                print(
                    f"[{finished}] cloop {operation} completed",
                    file=sys.stderr,
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cloop",
        description="RRT ensemble world-model and closed-loop research evaluation",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="strict method configuration YAML")
    parser.add_argument("--paths", default=None, help="server paths YAML")
    parser.add_argument("--run", default=None, help="single-component run name")
    parser.add_argument("--protocol", choices=("main_v1", "legacy_stage1"), default=None)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:N")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="read-only source-path and provenance checks")
    sub.add_parser("prepare", help="audit/import data, freeze split, and fit train-only preprocessing")

    train = sub.add_parser("train", help="train dynamics or outcome models")
    train.add_argument("--suite", choices=("dynamics", "outcome"), required=True)
    train.add_argument("--variants", nargs="+", choices=("baseline", "rrt", "ensemble", "rrt_ensemble"))
    train.add_argument("--seeds", nargs="+", type=int)
    train.add_argument("--resume", action="store_true", help="resume the matching complete-epoch last.pt")
    train.add_argument("--force-task", action="store_true", help="replace only selected model/metric tasks")

    evaluate = sub.add_parser("evaluate", help="evaluate factual forecasting, outcomes, or observed replay")
    evaluate.add_argument("--suite", choices=("dynamics", "outcome", "replay", "all"), required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), required=True)
    evaluate.add_argument("--variants", nargs="+", choices=("baseline", "rrt", "ensemble", "rrt_ensemble"))
    evaluate.add_argument("--seeds", nargs="+", type=int)

    synthetic = sub.add_parser("synthetic", help="train and evaluate the independent toy closed-loop environment")
    synthetic.add_argument("--seeds", nargs="+", type=int)

    sub.add_parser("freeze-protocol", help="freeze validation-selected parameters before test evaluation")
    sub.add_parser("report", help="render one report.md from existing flat artifacts")

    plan = sub.add_parser("plan", help="plan from a current visible, normalized state JSON")
    plan.add_argument("--case", required=True)
    plan.add_argument("--seed", required=True, type=int)

    smoke_parser = sub.add_parser("smoke", help="offline CPU smoke with in-memory non-medical data")
    smoke_parser.add_argument("--device", default=None, help="accepted here for the documented `smoke --device cpu` form")
    return parser


def _artifacts(config: dict, run_name: str | None) -> tuple[RunArtifacts, str]:
    selected = run_name or config["artifacts"]["default_run"]
    return RunArtifacts(config["paths"]["output_root"], selected), selected


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    artifacts: RunArtifacts | None = None
    try:
        config = load_config(args.config, args.paths, protocol=args.protocol, device=args.device)
        artifacts, run_name = _artifacts(config, args.run)
        if args.command == "doctor":
            result = doctor(config, artifacts, run_name)
            print(json.dumps(result, indent=2, sort_keys=True))
            if not result["ok"]:
                raise EngineError("doctor checks failed")
        elif args.command == "prepare":
            result = prepare(config, artifacts, run_name)
            print(json.dumps({"run": run_name, "cohort": result["cohort"], "audit": result["audit"]}, indent=2))
        elif args.command == "train":
            seeds = args.seeds or config["training"]["seeds"]
            operation = (
                f"train/{args.suite} seeds={list(seeds)} "
                f"resume={args.resume} force_task={args.force_task}"
            )
            with _training_log(config, run_name, operation):
                if args.suite == "dynamics":
                    variants = args.variants or config["training"]["variants"]
                    train_dynamics(
                        config,
                        artifacts,
                        variants,
                        seeds,
                        resume=args.resume,
                        force_task=args.force_task,
                    )
                else:
                    if args.variants:
                        parser.error("--variants is only valid for --suite dynamics")
                    train_outcome(
                        config,
                        artifacts,
                        seeds,
                        resume=args.resume,
                        force_task=args.force_task,
                    )
        elif args.command == "evaluate":
            evaluate_suite(
                config, artifacts, args.suite, args.split, seeds=args.seeds, variants=args.variants
            )
        elif args.command == "synthetic":
            seeds = args.seeds or config["training"]["seeds"]
            with _training_log(config, run_name, f"synthetic seeds={list(seeds)}"):
                run_synthetic_suite(config, artifacts, run_name, seeds)
        elif args.command == "freeze-protocol":
            freeze_protocol(config, artifacts)
        elif args.command == "report":
            print(generate_report(artifacts))
        elif args.command == "plan":
            print(json.dumps(plan_case(config, artifacts, args.case, args.seed), indent=2, sort_keys=True))
        elif args.command == "smoke":
            print(json.dumps(smoke(config), indent=2, sort_keys=True))
        else:  # pragma: no cover
            parser.error(f"unknown command {args.command}")
    except KeyboardInterrupt:
        if artifacts is not None:
            mark_interrupted(artifacts)
        print("cloop: interrupted at an epoch-safe boundary when available", file=sys.stderr)
        raise SystemExit(130)
    except (
        ArtifactError,
        ConfigError,
        DataError,
        EngineError,
        OutcomeError,
        PlannerError,
        PolicyError,
        WorldModelError,
        OSError,
        ValueError,
    ) as exc:
        print(f"cloop: error: {exc}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
