#!/usr/bin/env python3
"""jevm: Jev Motion, the decision helper for HyperFrames videos.

    python3 jevm.py <command> ...

It plans bounded decisions (caption highlights, stat claims, focus crops, frame QA) with Jev or the local Decider and writes
them to `decisions.json` / `decisions.md`. It never edits a composition, never runs `npx` and replaces no HyperFrames check.
"""
import sys

MIN_PYTHON = (3, 15)


def check_python(version_info=None):
    """A one-line message when `version_info` (default: this interpreter) is below `MIN_PYTHON`, else None."""
    found = tuple(version_info if version_info is not None else sys.version_info)[:3]
    if found < MIN_PYTHON:
        return (f"jevm needs Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer; this is Python "
                f"{'.'.join(str(n) for n in found)}")
    return None


if check_python() is not None:
    sys.stderr.write(check_python() + "\n")
    sys.exit(2)

import argparse
import importlib
import tempfile
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from core import decisions_io, runner  # noqa: E402
from jevlib.client import make_client  # noqa: E402
from kits import kit  # noqa: E402
from modules.universal import qa  # noqa: E402
from spec import build as spec_build  # noqa: E402
from spec import validate as spec_validate  # noqa: E402

def usd(text):
    value = float(text)
    if value < 0:
        raise argparse.ArgumentTypeError("a budget cannot be negative")
    return value


def _ids(text):
    return [part.strip() for part in text.split(",") if part.strip()]


def _kit_flag(parser, text="the kit folder (default: the spec's kit, else $JEVM_KIT_DIR)"):
    parser.add_argument("--kit-dir", metavar="PATH", help=text)


def _spend_flags(parser):
    parser.add_argument("--dry-run", action="store_true", help="print the estimate per backend and stop; send nothing")
    parser.add_argument("--no-jev", action="store_true",
                        help="send nothing to TypeSafe: Decider sets still run, the rest go to the agent lane")
    parser.add_argument("--no-images", action="store_true",
                        help="send no image to either backend (those rows go to the agent lane)")
    parser.add_argument("--budget", type=usd, metavar="USD",
                        help="Jev dollars this run may spend (default: spec.decide.budget_usd); over it, exit 3")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="jevm", description="Jev Motion: plan bounded video decisions with Jev or the local Decider and write "
        "decisions.json / decisions.md. HyperFrames only: jevm never edits a composition and never runs npx.",
        epilog="Run `jevm <command> --help` for one command. Paths with a space need quotes.")
    commands = parser.add_subparsers(dest="command", metavar="command", title="commands")

    kit_p = commands.add_parser("kit", help="build or check a kit manifest",
                                description="Build or check a kit's manifest.json. A kit is a folder you own: --kit-dir, "
                                "else $JEVM_KIT_DIR. See examples/kit.")
    kit_sub = kit_p.add_subparsers(dest="kit_command", metavar="build|check", required=True)
    for name, text in (("build", "stamp <kit-dir>/manifest.json: source hashes, fingerprint, build time (with --adapter, "
                                 "build it from the kit's files first)"),
                       ("check", "validate the manifest against the schema and say whether a source changed (exit 1 if "
                                 "invalid or stale)")):
        p = kit_sub.add_parser(name, help=text, description=text)
        _kit_flag(p, "the kit folder (default: $JEVM_KIT_DIR)")
        if name == "build":
            p.add_argument("--adapter", metavar="FILE.py",
                           help="a Python file with SOURCE_FILES and build(kit_dir, use_when) -> manifest")

    spec_p = commands.add_parser("spec", help="build or validate a video spec", description="Build or validate a video spec.")
    spec_sub = spec_p.add_subparsers(dest="spec_command", metavar="build|validate", required=True)
    sb = spec_sub.add_parser("build", help="words JSON + assets + a format id -> spec.json",
                             description="Words JSON + asset list + a format id -> spec.json.")
    sb.add_argument("--project", required=True, metavar="DIR", help="the HyperFrames project folder")
    sb.add_argument("--format", required=True, metavar="F", help="format id from formats/formats.json: short or long")
    sb.add_argument("--words", required=True, metavar="PATH", help="words JSON")
    sb.add_argument("--assets", metavar="PATH", help="asset list (JSON)")
    _kit_flag(sb, "the kit folder (default: the kit of an existing spec at the output path, else $JEVM_KIT_DIR)")
    sb.add_argument("-o", "--out", metavar="PATH", help="where to write spec.json (default: <project>/work/jevm/spec.json)")
    sv = spec_sub.add_parser("validate", help="check a spec against the schema and its manifest",
                             description="Check a spec against the schema and its manifest (exit 1 if invalid).")
    sv.add_argument("spec", metavar="spec.json")
    _kit_flag(sv, "check the manifest hash against this kit folder instead of the spec's")

    plan_p = commands.add_parser("plan", help="run CAP-02, CAP-03, STY-06 and QA-05 into decisions.json",
                                 description="Run the modules (caption highlights, stat claims) and merge their rows "
                                 "into <project>/work/jevm/decisions.json. Exit 0; 2 bad input; 3 over budget.")
    plan_p.add_argument("spec", metavar="spec.json")
    plan_p.add_argument("--stage", metavar="S", help="run one module: captions or claims")
    plan_p.add_argument("--only", metavar="ID,ID", type=_ids, help="run the modules that decide these ids (CAP-02, STY-06, ...)")
    _kit_flag(plan_p)
    _spend_flags(plan_p)

    crop_p = commands.add_parser("crop", help="propose a focus crop (AST-01) for one beat",
                                 description="Propose where a window's focus should sit on a frame. Writes an AST-01 row "
                                 "to decisions.json and crops/<beat>.jpg (the frame with the proposed box over it).")
    crop_p.add_argument("spec", metavar="spec.json")
    crop_p.add_argument("--beat", required=True, metavar="B", help="beat id; the row is AST-01:<B>")
    crop_p.add_argument("--asset", required=True, metavar="A", help="asset id from the spec (a video needs --t)")
    crop_p.add_argument("--t", type=float, metavar="S", help="second of the clip to cut the frame at (videos)")
    crop_p.add_argument("--want", required=True, metavar="TEXT", help="the whole region the shot needs to show")
    crop_p.add_argument("--body", metavar="WxH", help="window body in px (default: the kit's; long-form needs it)")
    _kit_flag(crop_p)
    _spend_flags(crop_p)

    qa_p = commands.add_parser("qa", help="triage a render frame by frame (QA-01)",
                               description="Cut a frame every --every seconds from a render, ask the QA set about each and "
                               "write work/jevm/qa/<render-stem>/{qa.json,qa.md,flagged.jpg}. Flags are advisory: exit 0.")
    qa_p.add_argument("render", metavar="RENDER.mp4")
    qa_p.add_argument("--spec", required=True, metavar="spec.json")
    qa_p.add_argument("--every", type=float, default=1.5, metavar="S", help="seconds between frames (default: 1.5)")
    _kit_flag(qa_p)
    _spend_flags(qa_p)

    resolve_p = commands.add_parser("resolve", help="accept or override decision rows",
                                    description="Mark rows of a decisions.json (or qa.json) accepted, or overridden "
                                    "with --value. Exit 0; 2 unknown id.")
    resolve_p.add_argument("decisions", metavar="decisions.json")
    resolve_p.add_argument("ids", metavar="ID[,ID...]", help="row ids, comma-separated")
    how = resolve_p.add_mutually_exclusive_group(required=True)
    how.add_argument("--accept", action="store_true", help="accept each row's own proposed value")
    how.add_argument("--value", metavar="V", help="override with V (JSON when it parses, else plain text)")
    resolve_p.add_argument("--by", required=True, choices=("agent", "human"), help="who decided")
    resolve_p.add_argument("--note", default="", metavar="TEXT", help="why")

    eval_p = commands.add_parser("eval", help="score a question set or module against its frozen eval set",
                                 description="Run frozen eval sets and write eval/<date>-<HHMMSS>-<backend>-<scope>.md. Run it once "
                                 "per backend and the second report carries the parity table. Exit 0 all gates pass; 1 a gate failed or "
                                 "could not be measured; 2 bad input; 3 over budget.")
    eval_p.add_argument("--module", metavar="M", help="qa, captions, assets or claims (default: all)")
    eval_p.add_argument("--qset", metavar="Q", help="qa_frame.short, highlight, crop_box or stat_check")
    eval_p.add_argument("--backend", choices=("jev", "local"), help="force one backend (default: jev; the gates are Jev numbers)")
    eval_p.add_argument("--model", metavar="M", help="replace the pinned Jev model for the sets it runs")
    eval_p.add_argument("--offline", action="store_true",
                        help="replay the frozen synthetic answers through fake backends: no network, no key, nothing spent")
    eval_p.add_argument("--out", type=Path, metavar="DIR", help="folder for the .md and .json (default: eval/)")
    _spend_flags(eval_p)
    return parser


def _run_eval(args):
    """`jevm eval`: hand the flags to the eval harness, imported only now so the other commands never need it. The exit
    code is the harness's own, unchanged: 0 all gates pass, 1 a gate failed or could not be measured, 2 bad input, 3 over
    budget. Every client built here is forced to `--backend`, or to `jev` without it: the gates are Jev numbers, and an
    unforced client would send the local-first sets (stat_check, presence) to Decider."""
    try:
        harness = importlib.import_module("evals.harness")
    except ModuleNotFoundError as exc:
        if exc.name not in ("evals", "evals.harness"):
            raise
        print(f"eval: the eval harness is not installed ({exc})", file=sys.stderr)
        return 2
    kwargs = {"qset": args.qset, "module": args.module, "backend": args.backend, "model": args.model,
              "dry_run": args.dry_run}
    if args.budget is not None:
        kwargs["budget"] = args.budget
    if args.out is not None:
        kwargs["out_dir"] = args.out
    force = args.backend or "jev"
    if args.offline:
        if args.no_jev or args.no_images:
            print("eval: --offline replays frozen answers; --no-jev and --no-images do not apply to it", file=sys.stderr)
            return 2
        with tempfile.TemporaryDirectory(prefix="jevm-eval-") as cache_dir:
            client = harness.offline_client(cache_dir=cache_dir, force_backend=force)
            return harness.run_eval(**kwargs, client=client, offline=True)
    if args.no_jev or args.no_images:
        kwargs["client"] = make_client(None, no_jev=args.no_jev, no_images=args.no_images, force_backend=force)
    return harness.run_eval(**kwargs)


def _run_kit(args):
    if args.kit_command == "build":
        return kit.cli_build(args.kit_dir, args.adapter)
    return kit.cli_check(args.kit_dir)


def _run_spec(args):
    if args.spec_command == "validate":
        return spec_validate.cli_validate(args.spec, args.kit_dir)
    return spec_build.cli_build(args.project, args.format, args.words, args.assets, args.kit_dir, args.out)


def _run_plan(args):
    return runner.plan(args.spec, stage=args.stage, only=args.only, dry_run=args.dry_run, no_jev=args.no_jev,
                       no_images=args.no_images, budget=args.budget, kit_dir=args.kit_dir)


def _run_crop(args):
    return runner.crop(args.spec, beat=args.beat, asset=args.asset, t=args.t, want=args.want, body=args.body,
                       dry_run=args.dry_run, no_jev=args.no_jev, no_images=args.no_images, budget=args.budget,
                       kit_dir=args.kit_dir)


def _run_qa(args):
    return qa.cli_qa(args.render, args.spec, every=args.every, dry_run=args.dry_run, no_jev=args.no_jev,
                     no_images=args.no_images, budget=args.budget, kit_dir=args.kit_dir)


def _run_resolve(args):
    return decisions_io.cli_resolve(args.decisions, args.ids, args.accept, args.value, args.by, args.note)


HANDLERS = {"kit": _run_kit, "spec": _run_spec, "plan": _run_plan, "crop": _run_crop, "qa": _run_qa,
            "resolve": _run_resolve, "eval": _run_eval}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    why = check_python()
    if why is not None:
        print(why, file=sys.stderr)
        return 2
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:                  # argparse exits for --help (0) and for a bad command line (2)
        return exc.code if isinstance(exc.code, int) else 2
    if args.command is None:
        parser.print_help()
        return 2
    return HANDLERS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
