import os
import shlex
import shutil
import subprocess

quote = shlex.quote


def command(cmd, env_script):
    tool = cmd.split()[0]
    if shutil.which(tool) is None and os.path.exists(env_script):
        return ["csh", "-c", "source %s; %s" % (env_script, cmd)]
    return ["sh", "-c", cmd]


def start(cmd, env_script, cwd, log):
    os.makedirs(cwd, exist_ok=True)
    return subprocess.Popen(command(cmd, env_script), cwd=cwd, stdout=open(log, "w"),
                            stderr=subprocess.STDOUT)


def run(cmd, env_script, cwd, log):
    wait([(log, start(cmd, env_script, cwd, log))])


def problems(text):
    keys = ("** Error", "** Fatal", "Error:", "not found")
    lines = [l for l in text.splitlines() if any(k in l for k in keys) and "PROC-3" not in l]
    return lines[:6]


def wait(jobs, timeout=3600):
    for log, proc in jobs:
        try:
            proc.wait(timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise RuntimeError("timed out, see " + log)
    for log, proc in jobs:
        text = open(log, errors="replace").read()
        found = problems(text)
        if proc.returncode != 0 or any("** Error" in l or "** Fatal" in l for l in found):
            hint = ""
            if any("not found" in l for l in found):
                hint = "\nThe tools were not found, run this on a USASK ENGR Lab machine."
            raise RuntimeError("failed, see %s\n%s%s" % (log, "\n".join(found), hint))
