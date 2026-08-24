import json
import os
import glob
import subprocess
import argparse
import logging

from git import InvalidGitRepositoryError, Repo
from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime


def _git_candidates(cfg):
    """Prefer Hydra working_dir, then this PlanT tree / repo root (never a missing host path)."""
    seen = []
    try:
        wd = getattr(getattr(cfg, "user", None), "working_dir", None)
        if wd:
            seen.append(Path(str(wd)))
    except Exception:
        pass
    here = Path(__file__).resolve()
    # util/logging.py → PlanT → plant2 → traffic-rule-bench
    seen.extend([here.parents[2], here.parents[3], Path.cwd()])
    out = []
    for p in seen:
        try:
            p = p.resolve()
        except Exception:
            continue
        if p.is_dir() and p not in out:
            out.append(p)
    return out


def _resolve_git_dir(cfg):
    for p in _git_candidates(cfg):
        r = subprocess.run(
            ["git", "-C", str(p), "rev-parse", "--show-toplevel"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if r.returncode == 0:
            top = Path(r.stdout.strip())
            if top.is_dir():
                return top
    return None


def _git_output(git_dir, *args):
    return (
        subprocess.check_output(
            ["git", "-C", str(git_dir), *args],
            stderr=subprocess.DEVNULL,
        )
        .decode("ascii", errors="replace")
        .strip()
    )


def setup_logging(cfg):
    # Log args
    Path(cfg.model.training.log_path).mkdir(parents=True, exist_ok=True)
    arg_dict = OmegaConf.to_container(cfg, resolve=True)
    args = argparse.Namespace(**arg_dict)
    with open(os.path.join(cfg.model.training.log_path, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    git_dir = _resolve_git_dir(cfg)
    with open(os.path.join(cfg.model.training.log_path, "git_info.txt"), "w") as f:
        f.write(
            f"Run started at: {str(datetime.now().strftime('%d/%m/%Y %H:%M:%S'))}\n"
        )
        if git_dir is None:
            f.write("Git state: unavailable (no git repo found under working_dir/plant2)\n")
        else:
            try:
                sha = _git_output(git_dir, "rev-parse", "HEAD")
                commit = _git_output(git_dir, "log", "-1")
                branch = _git_output(git_dir, "branch")
                f.write(f"Git dir: {git_dir}\n")
                f.write(f"Git state: {sha}\n")
                f.write(f"Git commit: {commit}\n")
                f.write(f"Git branch: {branch}\n\n")
                try:
                    repo = Repo(str(git_dir), search_parent_directories=True)
                    f.write(f"{repo.git.diff('HEAD')}")
                except (InvalidGitRepositoryError, Exception):
                    diff = subprocess.run(
                        ["git", "-C", str(git_dir), "diff", "HEAD"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        text=True,
                    )
                    f.write(diff.stdout or "")
            except (subprocess.CalledProcessError, OSError) as e:
                f.write(f"Git state: unavailable ({e})\n")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def sync_wandb(cfg):
    # TODO: sync wandb - still not working correctly
    wandb_files = glob.glob(f"./wandb/offline*/*.wandb")
    os.environ["TMPDIR"] = "/home/geiger/krenz73/tmp"
    for wandb_file in wandb_files:
        if os.path.getsize(wandb_file) > 5000000:
            os.system(f"wandb sync {wandb_file}")
