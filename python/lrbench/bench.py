import fcntl
import hashlib
import json
import os
import re
import shutil

import numpy as np

from . import metrics, plots, runner
from .config import BUILD, DC_ENV, MODELS, QUESTA_ENV, ROOT, Model

SRC = os.path.join(ROOT, "src")
TB = os.path.join(ROOT, "testbench")


def read_text(path):
    return open(path, encoding="utf-8", errors="replace").read()


def digest(paths):
    h = hashlib.md5()
    for p in paths:
        h.update(open(p, "rb").read())
    return h.hexdigest()


class Design:
    def __init__(self, files, name=None, baseline=False):
        self.files = [os.path.abspath(f) for f in files]
        for f in self.files:
            if not os.path.isfile(f):
                raise ValueError("no such file: " + f)
        self.name = name or os.path.splitext(os.path.basename(self.files[0]))[0]
        self.dir = os.path.join(BUILD, self.name)
        self.stamp = digest(self.files)
        if baseline:
            return
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.name) or self.name == "exact":
            raise ValueError("invalid design name '%s' (letters, digits, - and _ only, not 'exact')" % self.name)
        if not re.search(r"module\s+approx_mult\b", "".join(read_text(f) for f in self.files)):
            raise ValueError("no module named approx_mult in " + ", ".join(self.files))

    def out(self, model, mode):
        return os.path.join(self.dir, model, mode)


def exact_design():
    return Design([os.path.join(TB, "ref_mult.sv")], "exact", baseline=True)


def compile_design(design):
    sources = design.files + [os.path.join(SRC, "exact_mult.sv"), os.path.join(SRC, "mult16via8.sv"),
                              os.path.join(TB, "tb_nn.sv"), os.path.join(TB, "tb_nmed.sv")]
    cmd = ("vlog -work work %s; vopt -work work tb_nn -o tb_opt; vopt -work work tb_nmed -o nmed_opt"
           % " ".join(runner.quote(f) for f in sources))
    shutil.rmtree(os.path.join(design.dir, "work"), ignore_errors=True)
    log = os.path.join(design.dir, "compile.log")
    runner.run(cmd, QUESTA_ENV, design.dir, log)
    for line in open(log, errors="replace"):
        if "vopt-2241" in line:
            print("warning:", line.strip())


def start_tb(design, model, mode, exact_dir=None):
    out = design.out(model.name, mode)
    os.makedirs(out, exist_ok=True)
    args = "+MODEL=%s +OUT=%s +N=%d +SAT=%d +LAST=%d" % (runner.quote(model.dir), runner.quote(out), model.inputs,
                                                          model.saturate, int(mode == "last"))
    if exact_dir:
        args += " +EXACT=" + runner.quote(exact_dir)
    log = os.path.join(out, "sim.log")
    cmd = "vsim -c tb_opt %s -do 'run -all; quit -f'" % args
    return log, runner.start(cmd, QUESTA_ENV, design.dir, log)


def start_nmed(design):
    log = os.path.join(design.dir, "nmed.log")
    cmd = "vsim -c nmed_opt +OUT=%s -do 'run -all; quit -f'" % runner.quote(design.dir)
    return log, runner.start(cmd, QUESTA_ENV, design.dir, log)


def simulate(design, models):
    compile_design(design)
    jobs = [start_tb(design, m, "all") for m in models]
    exact = exact_design()
    if design.name != "exact":
        jobs += [start_tb(design, m, "last", exact.out(m.name, "all")) for m in models]
    jobs.append(start_nmed(design))
    runner.wait(jobs)
    if design.name == "exact":
        runner.wait([start_tb(design, m, "last", design.out(m.name, "all")) for m in models])


def exact_stamp(model):
    paths = [os.path.join(TB, "tb_nn.sv"), os.path.join(TB, "ref_mult.sv"), os.path.join(SRC, "exact_mult.sv"),
             os.path.join(SRC, "mult16via8.sv"), os.path.join(model.dir, "model.json")]
    for folder in ("layers", "inputs"):
        paths += [os.path.join(model.dir, folder, f) for f in sorted(os.listdir(os.path.join(model.dir, folder)))]
    return digest(paths)


def ensure_exact(models):
    os.makedirs(BUILD, exist_ok=True)
    with open(os.path.join(BUILD, "exact.lock"), "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
        except OSError:
            pass
        build_exact(models)


def build_exact(models):
    exact = exact_design()
    todo = []
    for m in models:
        stamp = os.path.join(exact.dir, m.name + ".stamp")
        done = os.path.exists(os.path.join(exact.out(m.name, "last"), "stats.txt"))
        if not (done and os.path.exists(stamp) and open(stamp).read() == exact_stamp(m)):
            todo.append(m)
    if todo:
        simulate(exact, todo)
        for m in todo:
            open(os.path.join(exact.dir, m.name + ".stamp"), "w").write(exact_stamp(m))


def task_metrics(design, model):
    ref = exact_design().out(model.name, "all")
    folder_all, folder_last = design.out(model.name, "all"), design.out(model.name, "last")
    out = {"nmed_l0": metrics.nmed(folder_all, ref, 0),
           "nmed_l1": metrics.nmed(folder_all, ref, 1),
           "nmed_l2": metrics.nmed(folder_all, ref, 2),
           "nmed_last": metrics.nmed(folder_last, ref, 2)}
    for mode, folder in (("all", folder_all), ("last", folder_last)):
        if model.task == "classify":
            out["acc_" + mode] = metrics.accuracy(folder, model.inputs, model.file("labels"))
        else:
            clean = np.load(model.file("clean"))
            imgs = metrics.images(folder, model.inputs, model.output_zero_point(), model.cfg["scale"])
            out["psnr_" + mode] = float(metrics.psnr(imgs, clean).mean())
            out["ssim_" + mode] = float(metrics.ssim(imgs, clean).mean())
    return out


def strips(design):
    model = Model("denoiser")
    zp, scale = model.output_zero_point(), model.cfg["scale"]
    clean, noisy = np.load(model.file("clean")), np.load(model.file("noisy"))
    mine = metrics.images(design.out("denoiser", "all"), model.inputs, zp, scale)
    ref = metrics.images(exact_design().out("denoiser", "all"), model.inputs, zp, scale)
    score, ref_score = metrics.psnr(mine, clean), metrics.psnr(ref, clean)
    order = np.argsort(score)
    columns = [("Clean", clean, None), ("Noisy", noisy, None), ("Exact", ref, ref_score),
               (design.name, mine, score)]
    folder = os.path.join(design.dir, "denoiser")
    plots.strip(os.path.join(folder, "best.png"), "best denoised samples", order[::-1][:8], columns)
    plots.strip(os.path.join(folder, "worst.png"), "worst denoised samples", order[:8], columns)


def read_hardware(design):
    path = os.path.join(design.dir, "hardware.json")
    if not os.path.exists(path):
        return {}
    data = json.load(open(path))
    return data if data.get("source") == design.stamp else {}


def summary(design, models):
    result = {"name": design.name}
    result["hardware"] = read_hardware(design)
    result["hardware"].update(metrics.nmed_all_inputs(design.dir))
    for m in models:
        result[m.name] = task_metrics(design, m)
    with open(os.path.join(design.dir, "summary.json"), "w") as f:
        json.dump(result, f, indent=1)
    return result


def evaluate(design, model_names=MODELS):
    models = [Model(n) for n in model_names]
    ensure_exact(models)
    simulate(design, models)
    result = summary(design, models)
    if "denoiser" in model_names:
        strips(design)
    return result


def synthesize(design):
    work = os.path.join(design.dir, "synth")
    rtl = os.path.join(work, "rtl")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(rtl)
    os.makedirs(os.path.join(work, "dc_files"))
    for f in design.files:
        shutil.copy(f, rtl)
    for f in (".synopsys_dc.setup", "syn_genreports.tcl"):
        shutil.copy(os.path.join(ROOT, "design_compiler", f), work)
    setup = ("set mult_name approx_mult;analyze -library work -format sverilog -autoread -recursive {./rtl};"
             "set root_dir ./dc_files/;set clk_period 0.2;set design approx_mult;set toggle_rate 0.01")
    log = os.path.join(work, "dc.log")
    runner.run('dc_shell-t -f ./syn_genreports.tcl -x "%s"' % setup, DC_ENV, work, log)
    reports = os.path.join(work, "dc_files")
    try:
        area = re.search(r"Total cell area:\s+([\d.]+)", open(reports + "/area_0.2_approx_mult.txt").read())
        delay = re.search(r"data arrival time\s+([\d.]+)", open(reports + "/timing_0.2_approx_mult.txt").read())
        power = open(reports + "/power_0.2_approx_mult_TR0.01.txt").read()
        if not (area and delay):
            raise ValueError
    except (IOError, ValueError):
        raise RuntimeError("synthesis failed, see %s\n%s" % (log, "\n".join(runner.problems(open(log).read()))))
    scale = {"pW": 1e-6, "nW": 1e-3, "uW": 1.0, "mW": 1e3, "W": 1e6}
    total = 0.0
    for key in ("Total Dynamic Power", "Cell Leakage Power"):
        m = re.search(key + r"\s*=\s*([\d.eE+-]+)\s*(\w+)", power)
        total += float(m.group(1)) * scale[m.group(2)]
    hardware = {"area_um2": float(area.group(1)), "delay_ns": float(delay.group(1)), "power_uw": total,
                "source": design.stamp}
    with open(os.path.join(design.dir, "hardware.json"), "w") as f:
        json.dump(hardware, f)
    return hardware


def ensure_exact_synth():
    exact = exact_design()
    if not read_hardware(exact):
        synthesize(exact)


def hardware_table(design):
    rows = {"exact": read_hardware(exact_design()), design.name: read_hardware(design)}
    keys = ["area_um2", "power_uw", "delay_ns"]
    return format_rows(["Hardware", "Area um2", "Power uW", "Delay ns"], rows_for(rows, keys, ["%.1f", "%.1f", "%.2f"]))


def format_rows(header, rows):
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    lines = ["  ".join(str(v).ljust(w) for v, w in zip(r, widths)) for r in [header] + rows]
    return "\n".join([lines[0], "-" * len(lines[0])] + lines[1:])


def rows_for(data, keys, fmt):
    return [[name] + [fmt[i] % data[name][k] if k in data[name] else "-" for i, k in enumerate(keys)]
            for name in data]


def table(design, model_names=MODELS):
    models = [Model(n) for n in model_names]
    results = {"exact": summary(exact_design(), models), design.name: summary(design, models)}
    blocks = []
    hardware = {n: results[n]["hardware"] for n in results}
    keys = ["area_um2", "power_uw", "delay_ns", "nmed_all_inputs", "med", "max_error"]
    header = ["Hardware", "Area um2", "Power uW", "Delay ns", "NMED all inputs", "MED", "Max err"]
    blocks.append(format_rows(header, rows_for(hardware, keys, ["%.1f", "%.1f", "%.2f", "%.2e", "%.2f", "%d"])))
    for m in models:
        data = {n: results[n][m.name] for n in results}
        if m.task == "classify":
            keys = ["acc_all", "nmed_l0", "nmed_l1", "nmed_l2", "acc_last", "nmed_last"]
            header = ["Classifier", "Acc all", "NMED L0", "NMED L1", "NMED L2", "Acc last", "NMED last"]
            fmt = ["%.1f"] + ["%.2e"] * 3 + ["%.1f", "%.2e"]
        else:
            keys = ["psnr_all", "ssim_all", "nmed_l0", "nmed_l1", "nmed_l2", "psnr_last", "ssim_last", "nmed_last"]
            header = ["Denoiser", "PSNR all", "SSIM all", "NMED L0", "NMED L1", "NMED L2", "PSNR last",
                      "SSIM last", "NMED last"]
            fmt = ["%.2f", "%.3f"] + ["%.2e"] * 3 + ["%.2f", "%.3f", "%.2e"]
        blocks.append(format_rows(header, rows_for(data, keys, fmt)))
    return "\n\n".join(blocks)
