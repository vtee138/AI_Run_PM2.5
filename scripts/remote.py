"""Drive the benchmark on the RTX 3090 box over SSH. Reads connection settings from .env.

  python scripts/remote.py check                 # ssh works? nvidia-smi, python, disk
  python scripts/remote.py push                  # copy code (src, scripts, configs, tests, pyproject) to REMOTE_DIR
  python scripts/remote.py setup                 # venv + CUDA torch + deps + kaggle creds + build parquet + tests
  python scripts/remote.py run  "<shell cmd>"    # run in REMOTE_DIR inside the venv, foreground
  python scripts/remote.py bg   "<shell cmd>" name   # nohup job, log at logs/<name>.log
  python scripts/remote.py tune                  # queue all tuning jobs sequentially in one bg job 'tune'
  python scripts/remote.py grid                  # queue CPU grid + GPU grid in one bg job 'grid'
  python scripts/remote.py status                # running jobs + tail of every log
  python scripts/remote.py log name [n]          # tail -n of logs/<name>.log
  python scripts/remote.py pull                  # fetch results/ and configs/best/ back to this machine
  python scripts/remote.py kill name             # kill a bg job by name
"""
from __future__ import annotations
import os, shlex, subprocess, sys, tarfile, io

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_env():
    env = {}
    p = os.path.join(ROOT, ".env")
    if not os.path.exists(p):
        sys.exit(".env missing: copy .env.example to .env and fill it in")
    for line in open(p, encoding="utf-8"):
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    for k in ("REMOTE_HOST", "REMOTE_USER"):
        if not env.get(k):
            sys.exit(f"{k} is empty in .env")
    env.setdefault("REMOTE_PORT", "22"); env.setdefault("REMOTE_DIR", "~/hanoi-pm25")
    env.setdefault("REMOTE_PYTHON", "python3"); env.setdefault("REMOTE_CUDA_INDEX", "https://download.pytorch.org/whl/cu124")
    return env


E = load_env()
SSH = ["ssh", "-p", E["REMOTE_PORT"], "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30"]
if E.get("REMOTE_SSH_KEY"):
    SSH += ["-i", E["REMOTE_SSH_KEY"]]
TARGET = f"{E['REMOTE_USER']}@{E['REMOTE_HOST']}"
D = E["REMOTE_DIR"]
VENV = f"{D}/.venv/bin/"


def ssh(cmd: str, check=True, capture=False, stdin=None):
    r = subprocess.run(SSH + [TARGET, cmd], check=check, text=not stdin, input=stdin,
                       capture_output=capture)
    return r.stdout if capture else r.returncode


def in_project(cmd: str) -> str:
    return f"cd {D} && export PATH={VENV}:$PATH && {cmd}"


def check():
    print("target:", TARGET)
    ssh("echo SSH OK; hostname; uname -a; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; "
        f"{E['REMOTE_PYTHON']} --version; df -h ~ | tail -1; ls {D} 2>/dev/null | head")


def push():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for item in ("src", "scripts", "configs", "tests", "pyproject.toml", "README.md"):
            p = os.path.join(ROOT, item)
            if os.path.exists(p):
                tar.add(p, arcname=item, filter=lambda ti: None if "__pycache__" in ti.name or ti.name.endswith(".pyc") else ti)
    data = buf.getvalue()
    print(f"pushing {len(data)/1e3:.0f} kB ...")
    subprocess.run(SSH + [TARGET, f"mkdir -p {D} && cd {D} && tar xzf -"], input=data, check=True)
    print("pushed to", D)


def setup():
    push()
    kag = ""
    if E.get("KAGGLE_USERNAME") and E.get("KAGGLE_KEY"):
        kag = (f"mkdir -p ~/.kaggle && printf '{{\"username\":\"{E['KAGGLE_USERNAME']}\",\"key\":\"{E['KAGGLE_KEY']}\"}}' "
               f"> ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json && ")
    ssh(f"cd {D} && {kag}"
        f"( test -d .venv || ~/.local/bin/virtualenv -q .venv ) && "
        f"export PATH={VENV}:$PATH && pip install -q --upgrade pip && "
        f"pip install -q torch --index-url {E['REMOTE_CUDA_INDEX']} && "
        "pip install -q -e . 'kagglehub[pandas-datasets]' lightgbm xgboost statsmodels optuna seaborn holidays "
        "scikit-posthocs shap pyyaml pytest pyarrow matplotlib && mkdir -p logs && "
        "python -c \"import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\" && "
        "python -m aqi_bench.data && python -m pytest tests -q")


def run(cmd: str):
    ssh(in_project(cmd), check=False)


def bg(cmd: str, name: str):
    full_cmd = in_project(cmd)
    ssh(f"mkdir -p {D}/logs && nohup bash -c {shlex.quote(full_cmd)} > {D}/logs/{name}.log 2>&1 & "
        f"echo $! > {D}/logs/{name}.pid; echo started {name} pid $(cat {D}/logs/{name}.pid)")


def tune():
    cmds = []
    for f in ["F1", "F2", "F3", "F4", "F5"]:
        for m in ["ridge", "lgbm", "xgb"]:
            cmds.append(f"python -W ignore -m aqi_bench.tune --model {m} --fset {f} --trials 50")
    for f in ["F1", "F2", "F3", "F4", "F5"]:
        for m in ["lstm", "gru", "tcn", "transformer"]:
            cmds.append(f"python -W ignore -m aqi_bench.tune --model {m} --fset {f} --trials 30")
    bg(" ; ".join(cmds), "tune")


def grid():
    bg("python -W ignore scripts/02_run_grid.py --models persistence snaive ridge lgbm xgb ; "
       "python -W ignore scripts/02_run_grid.py --models lstm gru tcn transformer", "grid")


def status():
    ssh(in_project("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; "
                   "for p in logs/*.pid; do n=$(basename $p .pid); if kill -0 $(cat $p) 2>/dev/null; then echo \"RUNNING $n\"; else echo \"finished $n\"; fi; done; "
                   "for l in logs/*.log; do echo \"== $l\"; tail -n 3 $l; done; "
                   "echo; echo \"metrics files: $(ls results/metrics/*.csv 2>/dev/null | wc -l)   tuned configs: $(ls configs/best/*.yaml 2>/dev/null | wc -l)\""), check=False)


def log(name: str, n: str = "40"):
    ssh(in_project(f"tail -n {n} logs/{name}.log"), check=False)


def pull():
    out = ssh(f"cd {D} && tar czf - results/metrics results/preds configs/best 2>/dev/null", capture=True, stdin=b"")
    # capture returned bytes because stdin bytes => text=False
    buf = io.BytesIO(out)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        tar.extractall(ROOT)
    n = len(os.listdir(os.path.join(ROOT, "results", "metrics")))
    print(f"pulled: {n} metric files, configs/best has {len(os.listdir(os.path.join(ROOT, 'configs', 'best')))} yaml")


def kill(name: str):
    ssh(in_project(f"pkill -P $(cat logs/{name}.pid) ; kill $(cat logs/{name}.pid) ; echo killed {name}"), check=False)


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a: sys.exit(__doc__)
    cmd = a[0]
    {"check": lambda: check(), "push": lambda: push(), "setup": lambda: setup(), "run": lambda: run(a[1]),
     "bg": lambda: bg(a[1], a[2]), "tune": lambda: tune(), "grid": lambda: grid(), "status": lambda: status(),
     "log": lambda: log(*a[1:]), "pull": lambda: pull(), "kill": lambda: kill(a[1])}[cmd]()
