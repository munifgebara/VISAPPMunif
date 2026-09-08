"""Mark the audited experiment complete and hash its final local artifacts."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from pdhms_restart.data import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    output = ROOT / config["output_root"]
    verification_path = output / "manifests/verification.json"
    verification = read_json(verification_path)
    if not verification["complete_run_verified"] or verification["failures"]:
        raise ValueError("The complete independent audit must pass first.")
    if verification["config_sha256"] != sha256_file(args.config):
        raise ValueError("Configuration differs from the audited version.")
    if verification["verifier_sha256"] != sha256_file(ROOT / "scripts/verify_nested_selection_controls.py"):
        raise ValueError("Verifier differs from the completed audit.")
    manifest_path = output / "manifests/run_manifest.json"
    manifest = read_json(manifest_path)
    if manifest["analysis_source_sha256"] != sha256_file(ROOT / "scripts/analyze_nested_selection_controls.py"):
        raise ValueError("Analysis source changed after results were generated.")
    for name, expected in read_json(output / "manifests/run_lock.json")["source_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"Fitting source changed: {name}")
    controls = ROOT / config["controls_output_root"]
    control_verification = read_json(controls / "manifests/verification.json")
    if control_verification["status"] != "passed":
        raise ValueError("Control-feature audit has not passed.")
    results = [read_json(path) for path in sorted((output / "units").glob("*/*/result.json"))]
    if len(results) != 175:
        raise ValueError("Missing outer units.")
    manifest.update(status="complete", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                    total_tuning_svm_fits=sum(result["tuning_svm_fits"] for result in results),
                    fit_count_note="Tuning fits only; excludes additional final refits and calibration fits.",
                    verification_sha256=sha256_file(verification_path),
                    control_feature_manifest_sha256=sha256_file(controls / "manifests/feature_manifest.json"),
                    control_verification_sha256=sha256_file(controls / "manifests/verification.json"),
                    visual_and_test_review="reports/closeout.md",
                    finalizer_sha256=sha256_file(Path(__file__)),
                    note="All declared fits, analyses and audits completed. Original paper and earlier experimental runs preserved.")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    artifact_path = output / "manifests/artifact_manifest.json"
    artifacts = [{"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                 for path in sorted(output.rglob("*")) if path.is_file() and path != artifact_path]
    artifact_path.write_text(json.dumps({"artifact_count": len(artifacts), "artifacts": artifacts}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "outer_units": len(results), "tuning_svm_fits": manifest["total_tuning_svm_fits"], "artifacts": len(artifacts)}, indent=2))


if __name__ == "__main__":
    main()
