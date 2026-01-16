# Plankton-Tracking-Pipeline
Modular video analysis pipeline for plankton behavior studies. Performs detection, tracking, motion and morphology analysis, and rule-based behavior classification (swimming vs attachment). Includes standalone tuning tools for reproducible ROI and threshold calibration.


---

## Design Principles

- **Separation of concerns**
  - Tuning ≠ tracking ≠ behavior analysis
- **Causal computation**
  - No future information used for velocity or state
- **Reproducibility**
  - All parameters recorded in per-run config manifests
- **Biological interpretability**
  - Motion + morphology jointly define behavioral state

---

## Intended Use

1. Tune detection parameters using standalone tools
2. Copy final values into `config.py`
3. Run the main pipeline on individual videos or directories
4. Perform higher-level behavioral analysis in post-processing

---

## License

MIT License (C) 2025, 2026 Thomas G. Zimmerman

