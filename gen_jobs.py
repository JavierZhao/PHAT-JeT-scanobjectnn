#!/usr/bin/env python
"""Render k8s Job manifests for the pre-registered grid (plan section 4).

The grid is defined once, here. Phases B/C/D need the delta chosen in Phase A,
so they are generated after A finishes, with --delta-star.

    python gen_jobs.py --phase A --git-sha <sha>
    python gen_jobs.py --phase B --git-sha <sha> --delta-star 0.25
"""

import argparse
import os

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "k8s",
                        "job_template.yaml")
OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "k8s", "jobs")

DEFAULT_IMAGE = "gitlab-registry.nrp-nautilus.io/jmduarte/hbb_interaction_network:latest"
DEFAULT_REPO = "https://github.com/aaronw5/PHAT-JeT.git"
DATA_DIR = "/j-jepa-vol/scanobjectnn/main_split"
RUN_ROOT = "/j-jepa-vol/phatjet-sonn/runs"

DELTAS = [0.5, 0.25, 0.125]
SEEDS = [0, 1, 2]
# delta is irrelevant when GMP is absent; this value only labels the run.
GMP_OFF_DELTA = 0.25
PATCH_DEFAULT = {"XS": 32, "S": 32, "M": 32, "L": 64}


def run_name(config, delta, gmp, ordering, patch, seed):
    """Full run identity: every varied knob appears, so names never collide."""
    return (
        f"phatjet-sonn-{config.lower()}-d{int(round(delta * 1000))}"
        f"-gmp{gmp}-{ordering}-p{patch}-s{seed}"
    )


def grid(phase, delta_star):
    """Return the list of runs for a phase. Reused runs are not re-emitted."""
    runs = []
    if phase == "A":
        for delta in DELTAS:
            for seed in SEEDS:
                runs.append(("S", delta, "on", "morton", 32, seed))
        for seed in SEEDS:
            runs.append(("S", GMP_OFF_DELTA, "off", "morton", 32, seed))
    elif phase == "B":
        # S at delta_star is reused from Phase A.
        for config in ("XS", "M", "L"):
            for seed in SEEDS:
                runs.append(
                    (config, delta_star, "on", "morton", PATCH_DEFAULT[config], seed)
                )
    elif phase == "C":
        # Morton at config M is reused from Phase B.
        for seed in SEEDS:
            runs.append(("M", delta_star, "on", "random", 32, seed))
    elif phase == "D":
        # P=32 is reused from Phase B.
        for patch in (16, 64, 128):
            for seed in (0, 1):
                runs.append(("M", delta_star, "on", "morton", patch, seed))
    else:
        raise ValueError(f"unknown phase {phase!r}")
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["A", "B", "C", "D"])
    parser.add_argument("--git-sha", required=True,
                        help="pinned commit the jobs check out")
    parser.add_argument("--delta-star", type=float, default=None,
                        help="best delta from Phase A; required for B/C/D")
    parser.add_argument("--image", default=DEFAULT_IMAGE,
                        help="pin by digest once Phase 0 has resolved it")
    parser.add_argument("--repo-url", default=DEFAULT_REPO)
    parser.add_argument("--namespace", default="cms-ml")
    args = parser.parse_args()

    if args.phase != "A" and args.delta_star is None:
        parser.error("--delta-star is required for phases B, C and D")
    if "@sha256:" not in args.image:
        print("WARNING: image is not pinned by digest; runs are not reproducible.")

    with open(TEMPLATE) as handle:
        template = handle.read()

    out_dir = os.path.join(OUT_ROOT, f"phase{args.phase}")
    os.makedirs(out_dir, exist_ok=True)

    runs = grid(args.phase, args.delta_star)
    for config, delta, gmp, ordering, patch, seed in runs:
        name = run_name(config, delta, gmp, ordering, patch, seed)
        manifest = template.format(
            run_name=name,
            namespace=args.namespace,
            phase=args.phase.lower(),
            repo_url=args.repo_url,
            git_sha=args.git_sha,
            image=args.image,
            data_dir=DATA_DIR,
            out_dir=f"{RUN_ROOT}/{name}",
            config=config,
            delta=delta,
            gmp=gmp,
            ordering=ordering,
            patch_size=patch,
            seed=seed,
        )
        with open(os.path.join(out_dir, f"{name}.yaml"), "w") as handle:
            handle.write(manifest)

    print(f"phase {args.phase}: wrote {len(runs)} manifests to {out_dir}")
    print(f"launch with: kubectl apply -f {out_dir}/ -n {args.namespace}")
    print("keep concurrent jobs at or below 8; launch in batches if needed.")


if __name__ == "__main__":
    main()
