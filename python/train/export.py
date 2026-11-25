"""Export a checkpoint's actor for the game client (and any other TypeScript user of ``src/sim/policy.ts``).

    python python/train/export.py runs/league/latest.pt --id selfplay --label "Self-play" --blurb "PPO, then self-play"

writes ``public/policies/<id>.bin`` and lists it in ``public/policies/index.json``, which is what the
client's menu offers. Only the actor is exported (the critic is a training aid), as 16-bit floats.

File layout, little-endian: ``"AIMP"``, u32 version (1), u32 header length, the header as JSON (padded with
spaces to a multiple of four bytes), then every tensor of ``header["tensors"]`` in order. The header also
carries one worked example (``check``): an observation, a mask and the action-kind log-probabilities the
exported weights give for it, so a reader can verify its own forward pass.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

from policy import load_checkpoint

ACTOR = ("body.", "kind_head.", "dir_head.", "dist_head.", "target_head.")


def export(checkpoint: str, out: Path, meta: dict) -> dict:
    policy, data = load_checkpoint(checkpoint, "cpu")
    cfg = policy.config
    if cfg["kind"] != "mlp":
        raise ValueError("only MLP policies can be exported")
    state = {k: v for k, v in policy.state_dict().items() if k.startswith(ACTOR)}
    # Round the weights first, so the worked example is computed from exactly what is written.
    rounded = {k: v.numpy().astype("<f2") for k, v in state.items()}
    policy.load_state_dict({k: torch.from_numpy(v.astype(np.float32)) for k, v in rounded.items()}, strict=False)
    rng = np.random.default_rng(7)
    obs = rng.random(cfg["obs_size"], dtype=np.float32).round(3)
    hero = cfg["hero_offset"]
    obs[hero:hero + 5] = 0
    obs[hero + 2] = 1
    mask = np.ones(cfg["nvec"][0] + cfg["nvec"][3], np.uint8)
    mask[5] = 0  # one forbidden kind, to exercise the masking
    with torch.no_grad():
        o, m = torch.from_numpy(obs)[None], torch.from_numpy(mask)[None]
        h = policy.body(o)
        kind = policy.kind_log_probs(h, m)[0]
        k = torch.tensor([3])
        lp_dir, lp_dist, lp_target = policy.sub_log_probs(h, o, m, k)
    header = {
        "obs_size": cfg["obs_size"], "nvec": cfg["nvec"], "hidden": cfg["hidden"], "layers": cfg["layers"], "activation": cfg["activation"],
        "hero_offset": cfg["hero_offset"], "dtype": "f16",
        "tensors": [[k, *v.shape] for k, v in rounded.items()],
        "check": {"obs": [round(float(x), 3) for x in obs], "mask": mask.tolist(), "kind": [max(float(x), -1e9) for x in kind], "sub_kind": 3,
                  "dir": [float(x) for x in lp_dir[0]], "dist": [float(x) for x in lp_dist[0]], "target": [max(float(x), -1e9) for x in lp_target[0]]},
        "meta": {**meta, "update": data["meta"].get("update"), "env_steps": data["meta"].get("env_steps"), "trainer": data["meta"].get("trainer")},
    }
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 4)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(b"AIMP" + struct.pack("<II", 1, len(raw)) + raw)
        for v in rounded.values():
            f.write(v.tobytes())
    return header


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--id", required=True, help="file name and the id the client uses (letters, digits, dashes)")
    p.add_argument("--label", default=None, help="name shown in the menu (default: the id)")
    p.add_argument("--blurb", default="", help="one line about the model, shown with it")
    p.add_argument("--greedy", action="store_true", help="the client plays this model's most likely action instead of sampling "
                   "(better for a policy that has only imitated)")
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "public" / "policies"))
    p.add_argument("--no-index", action="store_true", help="only write the .bin file, do not list it in index.json "
                   "(for a league entry: --id model --out bots/<name> --no-index)")
    args = p.parse_args()
    out = Path(args.out)
    entry = {"id": args.id, "label": args.label or args.id, "blurb": args.blurb, "file": f"{args.id}.bin"}
    if args.greedy:
        entry["greedy"] = True
    header = export(args.checkpoint, out / entry["file"], {"id": args.id, "label": entry["label"]})
    if args.no_index:
        print(f"wrote {out / entry['file']} ({(out / entry['file']).stat().st_size / 1e6:.1f} MB, update {header['meta']['update']})")
        raise SystemExit(0)
    index = out / "index.json"
    models = json.loads(index.read_text(encoding="utf-8")) if index.exists() else []
    models = [m for m in models if m["id"] != args.id] + [entry]
    index.write_text(json.dumps(models, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    size = (out / entry["file"]).stat().st_size
    print(f"wrote {out / entry['file']} ({size / 1e6:.1f} MB, update {header['meta']['update']}), {len(models)} models listed in {index}")
