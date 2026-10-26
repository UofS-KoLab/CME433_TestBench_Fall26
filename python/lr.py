import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lrbench import MODELS, Design, evaluate, exact_design, synthesize, table
from lrbench.bench import ensure_exact, ensure_exact_synth, hardware_table
from lrbench.config import ROOT, Model


def main():
    parser = argparse.ArgumentParser(prog="lr")
    parser.add_argument("command", choices=["check", "eval", "synth", "all"])
    parser.add_argument("files", nargs="*", help="SystemVerilog files, one of them defining approx_mult")
    parser.add_argument("--model", choices=MODELS + ["both"], default="both")
    parser.add_argument("--name", help="name of the build folder (default: first file name)")
    args, rest = parser.parse_known_args()
    if any(r.startswith("-") for r in rest):
        parser.error("unrecognized arguments: " + " ".join(rest))
    args.files += rest
    models = MODELS if args.model == "both" else [args.model]

    if args.command == "check":
        ensure_exact([Model(m) for m in models])
        print(table(exact_design(), models))
        return

    print("this can take a few minutes")
    design = Design(args.files or [os.path.join(ROOT, "src", "approx_mult.sv")], args.name)
    if args.command in ("eval", "all"):
        evaluate(design, models)
    if args.command in ("synth", "all"):
        synthesize(design)
        ensure_exact_synth()
    if args.command == "synth":
        print(hardware_table(design))
        return
    print(table(design, models))
    if "denoiser" in models:
        print("\nbest and worst images: %s/denoiser/best.png and worst.png" % design.dir)


try:
    main()
except (RuntimeError, ValueError) as e:
    sys.exit("error: %s" % e)
