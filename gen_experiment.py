#!/usr/bin/env python
"""Generate validated k8s Job manifests for one-off experiments.

Replaces ad-hoc `sed` substitution, which produced four corrupted manifests in
one session -- including one where `--` was used as a search pattern and every
flag prefix was rewritten. Two of those reached the cluster before being caught.

Two properties make this safe where sed was not:

1. **Keyword arguments, not positional substitution.** Training flags are built
   from an explicit dict, so a misordered argument is a Python error rather
   than a silently mangled file.
2. **Post-generation validation.** Every rendered manifest is parsed back and
   its training command compared against what was requested. A manifest that
   does not round-trip is deleted, not written.

Usage:
    python gen_experiment.py --name m-rf-kernel7 --config M --seeds 0 1 \\
        --git-sha $(git rev-parse HEAD) --gpu both \\
        --set delta=0.09375 recipe=pointnext height_mode=shifted gmp_kernel=7
"""

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(HERE, "k8s", "jobs")
DATA_DIR = "/j-jepa-vol/scanobjectnn/main_split"
RUN_ROOT = "/j-jepa-vol/phatjet-sonn/runs"
REPO = "https://github.com/JavierZhao/PHAT-JeT-scanobjectnn.git"
IMAGE = ("gitlab-registry.nrp-nautilus.io/jmduarte/hbb_interaction_network"
         "@sha256:d876f8329b3345f2086e6cb6f70a3811810acdc87b40b6f15ee90c73b281ad1d")

# Flags that are always present; --set overrides and extends these.
BASE_FLAGS = {"config": "M", "delta": "0.09375", "gmp": "on",
              "ordering": "morton", "patch_size": "32"}

TEMPLATE = """apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: cms-ml
  labels:
    jobgroup: phatjet-sonn
    phase: "{phase}"
spec:
  parallelism: 1
  completions: 1
  # 4, not 1. The plan chose 1 to fail fast, but the observed failure mode is
  # node taint eviction (cluster maintenance), not code error -- five jobs died
  # that way in one batch. Retries are now safe because train_scanobjectnn.py
  # archives a previous attempt's metrics.json instead of overwriting it.
  backoffLimit: 4
  template:
    metadata:
      labels:
        jobgroup: phatjet-sonn
        phase: "{phase}"
    spec:
      restartPolicy: Never
      initContainers:
      - name: init-clone-repo
        image: alpine/git
        command: ["/bin/sh", "-c"]
        args:
        - |
          git clone --no-checkout {repo} /opt/repo/PHAT-JeT && \\
          cd /opt/repo/PHAT-JeT && \\
          git checkout {sha} && \\
          chown -R 1000:1000 /opt/repo/PHAT-JeT
        volumeMounts:
        - name: git-repo
          mountPath: /opt/repo
        resources:
          requests: {{cpu: '1', memory: 1Gi}}
          limits: {{cpu: '1', memory: 1Gi}}
      containers:
      - name: train
        image: {image}
        command: ["/bin/bash", "-c"]
        args:
        - |
          cd /opt/repo/PHAT-JeT && \\
          python scripts/train_scanobjectnn.py \\
            --data_dir {data_dir} \\
            --out {out_dir} \\
{flag_lines}
        env:
        - name: GIT_SHA
          value: "{sha}"
        - name: IMAGE_DIGEST
          value: "{image}"
        volumeMounts:
        - name: git-repo
          mountPath: /opt/repo
        - name: j-jepa-vol
          mountPath: /j-jepa-vol
        resources:
          requests: {{cpu: '4', memory: 32Gi, {gpu_resource}: 1}}
          limits: {{cpu: '4', memory: 32Gi, {gpu_resource}: 1}}
      volumes:
      - name: git-repo
        emptyDir: {{}}
      - name: j-jepa-vol
        persistentVolumeClaim:
          claimName: j-jepa-vol
{placement}"""

AFFINITY_3090 = """      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values:
                - NVIDIA-GeForce-RTX-3090
                - NVIDIA-GeForce-RTX-4090
"""

TOLERATION_A100 = """      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
"""


def render(name, phase, sha, flags, gpu):
    """Render one manifest. `flags` maps CLI option name -> value."""
    lines = [f"            --{k} {v} \\" for k, v in flags.items()]
    lines[-1] = lines[-1].rstrip(" \\")
    return TEMPLATE.format(
        name=name, phase=phase, sha=sha, repo=REPO, image=IMAGE,
        data_dir=DATA_DIR, out_dir=f"{RUN_ROOT}/{name}",
        flag_lines="\n".join(lines),
        gpu_resource="nvidia.com/a100" if gpu == "a100" else "nvidia.com/gpu",
        placement=TOLERATION_A100 if gpu == "a100" else AFFINITY_3090,
    )


def validate(path, name, sha, flags, gpu):
    """Parse the manifest back and confirm it says what we asked for.

    This is the check that ad-hoc sed lacked. Returns a list of problems.
    """
    text = open(path).read()
    problems = []

    if f"name: {name}\n" not in text:
        problems.append("job name missing or altered")
    if text.count(sha) < 2:
        problems.append("pinned SHA not present in both checkout and env")

    # Every requested flag must appear exactly once, with its value intact.
    for key, value in flags.items():
        matches = re.findall(rf"--{re.escape(key)} (\S+)", text)
        if len(matches) != 1:
            problems.append(f"--{key} appears {len(matches)}x, expected once")
        elif matches[0] != str(value):
            problems.append(f"--{key} is {matches[0]!r}, expected {value!r}")

    # No mangled flags: every '--word' in the command must be a real option.
    for token in re.findall(r"--([a-zA-Z_]+)", text.split("train_scanobjectnn.py")[-1]):
        if token not in set(flags) | {"data_dir", "out"}:
            problems.append(f"unexpected flag --{token} (corrupted substitution?)")

    want_gpu = "nvidia.com/a100" if gpu == "a100" else "nvidia.com/gpu"
    if text.count(f"{want_gpu}: 1") != 2:
        problems.append(f"{want_gpu} should appear in both requests and limits")
    if gpu == "a100" and "gpu.product" in text:
        problems.append("a100 variant must not carry the 3090/4090 affinity")
    return problems


def sha_on_remote(sha):
    """Launching a manifest pinning an unpushed commit fails at git checkout."""
    try:
        out = subprocess.check_output(
            ["git", "ls-remote", "fork", "scanobjectnn"], cwd=HERE,
            stderr=subprocess.DEVNULL, timeout=60).decode()
        remote_head = out.split()[0] if out.split() else ""
        if remote_head == sha:
            return True
        merged = subprocess.run(
            ["git", "merge-base", "--is-ancestor", sha, "fork/scanobjectnn"],
            cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return merged.returncode == 0
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="base run name, seed appended")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--git-sha", required=True)
    p.add_argument("--phase", default="exp")
    p.add_argument("--gpu", choices=["3090", "a100", "both"], default="both")
    p.add_argument("--out-dir", default=None, help="defaults to k8s/jobs/<phase>")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="training flags, e.g. gmp_kernel=7 height_mode=shifted")
    args = p.parse_args()

    flags = dict(BASE_FLAGS)
    for item in args.set:
        if "=" not in item:
            p.error(f"--set entries must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        flags[key] = value

    if not sha_on_remote(args.git_sha):
        p.error(f"{args.git_sha[:8]} is not on the remote -- push before "
                "generating, or the init container's git checkout will fail")

    out_dir = args.out_dir or os.path.join(OUT_ROOT, args.phase)
    os.makedirs(out_dir, exist_ok=True)
    gpus = ["3090", "a100"] if args.gpu == "both" else [args.gpu]

    written, failed = [], []
    for seed in args.seeds:
        for gpu in gpus:
            run_flags = dict(flags, seed=seed)
            name = f"{args.name}-s{seed}" + ("-a100" if gpu == "a100" else "")
            path = os.path.join(out_dir, f"{name}.yaml")
            with open(path, "w") as handle:
                handle.write(render(name, args.phase, args.git_sha, run_flags, gpu))
            problems = validate(path, name, args.git_sha, run_flags, gpu)
            if problems:
                os.remove(path)
                failed.append((name, problems))
            else:
                written.append(path)

    for name, problems in failed:
        print(f"REJECTED {name}:", file=sys.stderr)
        for problem in problems:
            print(f"    {problem}", file=sys.stderr)
    for path in written:
        print(f"ok  {os.path.basename(path)}")
    print(f"\n{len(written)} written, {len(failed)} rejected -> {out_dir}")
    if failed:
        sys.exit(1)
    print(f"launch: kubectl apply -f {out_dir}/ -n cms-ml")


if __name__ == "__main__":
    main()
