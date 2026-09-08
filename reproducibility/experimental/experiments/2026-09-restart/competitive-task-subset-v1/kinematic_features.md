# Fixed kinematic and pressure baseline

The descriptor contains 83 fixed features, calculated independently for each
recording. It retains native coordinate and pressure levels and converts recorded
millisecond differences to seconds. No labels, participant identifiers,
recording-level normalization, smoothing or supervised feature selection enter
extraction. Subsequent SVM standardization belongs inside its training folds.

This is an independently specified baseline inspired by the kinematic and
pressure signal families studied by Drotar et al. (2016). It does not reproduce
that work's exact descriptor, feature selection or evaluation protocol.

The 11 global attributes describe valid acquisition time, its contact/air and
transition components, contact-time fraction, numbers of contact/air strokes,
contact/air path lengths and contact bounding-box width/height. Another 72
attributes summarize 12 distributions: stroke duration and length for each
state; speed, acceleration magnitude and jerk magnitude for each state; contact
pressure and its absolute rate of change. Each distribution contributes its
mean, population standard deviation, median, 5th and 95th percentiles and IQR.
The ordered schema and complete conventions are in `kinematic_descriptor.json`.

Velocity is a coordinate secant at the interval midpoint. Acceleration and jerk
are successive vector differences divided by their midpoint time differences.
They describe changes in both direction and speed. Derivatives never cross a
state transition or invalid timestamp interval. A stroke is a maximal run of
one pen state, so a timestamp anomaly does not invent an additional stroke.
Empty distributions contribute six zeros and are identified in diagnostics.
Finite differences can amplify quantization and measurement noise; this
baseline applies no unreported denoising.

A pre-fit input audit, without labels or performance results, found 253
nonpositive increments and two implausible positive jumps of approximately
1.334e12ms. There were also 145 gaps lasting between 1 and 26.61s. The final rule
retains positive intervals up to 60s, preserving those plausible pauses, and
excludes nonpositive or longer intervals from time and length while splitting
derivative runs. Long valid gaps contribute average secant velocity; they do
not reveal the instantaneous motion inside a sampling gap. The resulting time
feature is valid acquisition time, not an assertion about total clinical task
duration. Half of each valid contact/air transition interval goes to each state;
transitions do not contribute path lengths or derivatives.

The local legacy README (line 79) and SVC loader (line 68) identify timestamps as
milliseconds. The median positive step is 8 native units, with the 99th
percentile at 15. The corpus `info.txt` supplies the Y/X column order. Its MATLAB
loader has only a commented coordinate conversion, so this baseline preserves
native coordinate units rather than asserting a millimeter scale. It uses
recorded intervals, independently of conflicting nominal sampling frequencies
in earlier sources.

Extraction produced a finite 597x83 matrix in exactly the original static
metadata order. Of 1,364,022 intervals, 255 were excluded: none within contact,
18 within air and 237 at transitions. The 145 long plausible pauses were
retained. Diagnostics report each record's excluded fraction and distribution
sample counts; `summary.json` reports zero fractions per feature and empty
distributions. Metadata preserves labels only for subsequent evaluation.

Fifteen analytical tests passed, including millisecond conversion, nonuniform
intervals, vector acceleration during constant-speed turning, pen lifts,
timestamp anomalies, the 60s boundary and invariance to coordinate/time origin.
An independent audit passed 5,991 checks: source and artifact hashes, all 597 raw
file hashes, row alignment, finite values and independent recomputation of 28
attributes per record. Its largest duration difference was 5.7e-14s. Neither
extraction nor verification fitted a model.

Reproduction commands, using the experiment environment and `PYTHONPATH=src`:

```powershell
python scripts/prepare_kinematic_features.py
python scripts/verify_kinematic_features.py
python -m pytest tests/test_kinematic_features.py -q
```

The preparation command refuses to overwrite a nonempty output directory.
Its default output is `runs/competitive-task-subset-v1/kinematic_features/`
under the restart folder. Data, source and artifact hashes are recorded in
`manifest.json`; the independent audit is `verification.json`.
