"""Close the independently audited competitive run and hash local artifacts."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from pdhms_restart.data import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    output = ROOT / config["output_root"]
    lock = read(output / "manifests/run_lock.json")
    if sha256_file(args.config) != lock["config_sha256"]:
        raise ValueError("Configuration changed after fitting.")
    if sha256_file(args.config.with_name("DECISION.md")) != lock["protocol_sha256"]:
        raise ValueError("Protocol changed after fitting.")
    for name in ("source_sha256", "input_sha256"):
        for path, expected in lock[name].items():
            if sha256_file(ROOT / path) != expected:
                raise ValueError(f"Changed frozen source/input: {path}")
    verification_path = output / "manifests/verification.json"
    verification = read(verification_path)
    if not verification["complete_run_verified"] or verification["failures"]:
        raise ValueError("Independent complete audit must pass.")
    if verification["verifier_sha256"] != sha256_file(ROOT / "scripts/verify_competitive_task_subset.py"):
        raise ValueError("Verifier changed after audit.")
    manifest_path = output / "manifests/run_manifest.json"
    manifest = read(manifest_path)
    if manifest["analysis_source_sha256"] != sha256_file(ROOT / "scripts/analyze_competitive_task_subset.py"):
        raise ValueError("Analysis source changed after evaluation.")
    checks = [output / "kinematic_features/verification.json",
              output / "transfer_features/manifests/post_extraction_verification.json"]
    if any(read(path)["status"] != "passed" for path in checks):
        raise ValueError("Both feature audits must pass.")
    if not (output / "reports/closeout.md").is_file() or not (output / "reports/figure_qa.md").is_file():
        raise ValueError("Closeout and visual reviews must exist.")
    manifest.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                    verification_sha256=sha256_file(verification_path),
                    feature_verification_sha256={path.relative_to(output).as_posix(): sha256_file(path) for path in checks},
                    finalizer_sha256=sha256_file(Path(__file__)),
                    scope="Four prespecified methods; literature-fixed T2/T3/T4 primary and all8 secondary. No external validation.")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    destination = output / "manifests/artifact_manifest.json"
    artifacts = [{"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                 for path in sorted(output.rglob("*")) if path.is_file() and path != destination]
    destination.write_text(json.dumps({"artifact_count": len(artifacts), "artifacts": artifacts}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "outer_units": manifest["completed_outer_units"],
                      "tuning_fits": manifest["tuning_fits"], "final_fits": manifest["final_fits"],
                      "artifacts": len(artifacts)}, indent=2))


if __name__ == "__main__":
    main()
