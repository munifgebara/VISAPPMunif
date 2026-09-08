These are exact copies of the manuscript/figure generators and project environment files.
Run them from the original experiment repository with its saved runs and src/ package.
The Overleaf package supplies the rendered figures, editable activity SVG, source manifests,
and generated gallery_start.tex; it does not embed participant recordings or model checkpoints.
Compile main.tex or gallery.tex at the archive root for normal Overleaf editing.
In the experiment repository: python scripts/build_paper.py builds the expanded advisor draft;
python scripts/build_paper.py --submission-check also enforces venue page/character limits.
