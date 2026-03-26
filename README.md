# Quantifying the Unknowable

### Fisher Information–Based Analysis of Boundary Localization Uncertainty in Medical Image Segmentation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Phase%201%20Live-green?style=flat-square)]()
[![Domain](https://img.shields.io/badge/Domain-Medical%20Imaging-teal?style=flat-square)]()
[![Type](https://img.shields.io/badge/Type-Research%20Prototype-purple?style=flat-square)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

> **🟢 Phase 1 demo is live** — [**Launch the interactive app →**](https://jsdqwgw4nefsm6flaphnjt.streamlit.app/)

---

## Overview

Segmentation masks in medical imaging are routinely treated as ground truth. In practice, they are estimates — and like any estimate, they carry uncertainty. Yet the dominant evaluation framework, built around overlap metrics such as the Dice coefficient, measures agreement between annotations rather than the degree to which the underlying image data actually supports a precise boundary localization.

This project develops a complementary statistical framework grounded in classical estimation theory. By modeling the image intensity profile near a segmentation boundary as a parametric observation process, we can compute the **Fisher Information** associated with boundary position and derive its **Cramér–Rao Lower Bound (CRLB)** — a theoretical floor on the variance of any unbiased boundary estimator. The result is a spatially resolved characterization of where a contour is tightly constrained by image data and where it is not.

The long-term output of this work is a family of **boundary stability maps** — per-pixel or per-boundary-point uncertainty estimates grounded not in model behavior but in image physics.

---

## The Problem

> *When a radiation oncologist redraws a tumor contour, is the disagreement with a prior contour a failure of attention — or a consequence of the image itself?*

Current medical image segmentation evaluation does not answer this question. Overlap metrics like Dice, Hausdorff distance, and surface distance measure the geometric relationship between two contours. They do not assess whether the raw image data contains sufficient information to distinguish between those contours in the first place.

This distinction matters because image uncertainty is irreducible. No model, no annotator, and no post-processing step can recover boundary precision that is not supported by the data. Noise, blur, partial-volume effects, and low tissue contrast impose hard limits on localization accuracy. A model that is highly confident about a boundary in a noisy, low-contrast region is not more accurate — it is overconfident relative to what the data can support.

Current uncertainty quantification methods in deep learning — Monte Carlo Dropout, deep ensembles, probabilistic U-Nets — characterize **model uncertainty**: the spread of predictions across model parameter configurations. They do not characterize **data-supported uncertainty**: the intrinsic precision limit imposed by the image formation process.

This project addresses the latter.

---

## Core Research Question

> **How precisely can a segmentation boundary be localized from the image data itself, independent of any model, annotator, or algorithm?**

This question is answerable in the language of statistical estimation theory. The answer depends on the local image contrast, the point-spread function, the noise model, and the spatial sampling — not on any downstream processing.

---

## Conceptual Framework

### From Observation to Estimation

Consider a 1D image intensity profile crossing a tissue boundary at position θ. If we adopt a generative model for the observed intensity values — for example, a blurred step edge corrupted by additive Gaussian noise — then the log-likelihood of the observed data as a function of θ has a curvature that reflects how sensitively the data responds to shifts in boundary position.

That curvature, evaluated at the true parameter value, is the **Fisher Information**:

```
I(θ) = E[ (∂/∂θ  log p(x; θ))² ]
     = −E[ ∂²/∂θ² log p(x; θ) ]
```

The **Cramér–Rao Lower Bound** then establishes that no unbiased estimator of θ can achieve a variance below:

```
Var(θ̂) ≥ 1 / I(θ)
```

This bound is tight under regularity conditions and is achieved asymptotically by the maximum likelihood estimator. It represents not a property of any algorithm, but a property of the data-generating process itself.

### Extension to Contour Stability

In 2D, a segmentation boundary is a curve parameterized by arc length or angle. At each point along the contour, a local 1D profile can be extracted orthogonal to the boundary. The Fisher Information computed from that profile gives an estimate of the intrinsic localizability of the boundary at that point.

Mapped back onto the image, these local information values produce a **stability map** or **information-limit map**: a spatially resolved representation of where the boundary is tightly constrained by image data and where it lies in a low-information region. In high-contrast, sharp-edge regions, Fisher Information is high and the CRLB is small. In blurry, noisy, or low-contrast regions, Fisher Information falls and the CRLB expands — quantifying the region of legitimate positional ambiguity.

### Key Variables

| Symbol | Interpretation |
|--------|---------------|
| θ | True boundary position (scalar or field) |
| p(x; θ) | Image likelihood given boundary position |
| I(θ) | Fisher Information at θ |
| 1 / I(θ) | Cramér–Rao Lower Bound on boundary localization variance |
| σ_CRLB | Intrinsic boundary localization uncertainty (standard deviation) |

---

## Why This Matters

### Clinical Context

In radiation oncology, gross tumor volume (GTV) delineation directly determines the radiation dose delivered to surrounding healthy tissue. Inter-observer contour variability is well documented and clinically significant — but it is rarely decomposed into components attributable to image quality versus annotator disagreement. A framework that quantifies the image-supported certainty of a boundary could help distinguish cases where contour review is likely to be productive from cases where the image simply does not support greater precision.

Beyond oncology, the same issue arises across medical image analysis: organ segmentation in abdominal CT, white-matter lesion delineation in MRI, vessel boundary extraction in angiography. Wherever a boundary lies near a region of low contrast, partial-volume mixing, or elevated noise, the CRLB framework offers a principled lower bound on achievable localization accuracy.

### Methodological Context

The segmentation uncertainty literature has grown substantially with the adoption of probabilistic deep learning. Most current methods, however, produce **model-conditional** uncertainty: they characterize how much a trained model's predictions vary given data, but they cannot characterize how much any estimator must vary given the image. A tool that measures data-supported uncertainty is complementary to, not a replacement for, model-based uncertainty — and, crucially, it is model-free. It requires no training data, no annotations, and no learned parameters.

This is the kind of tool that can inform evaluation frameworks, support quality assurance pipelines, and serve as a calibration reference for learned uncertainty estimates.

---

## Project Roadmap

The project is organized into six phases, each building on the previous. Early phases establish the theory through controlled synthetic experiments. Later phases extend the framework to realistic imaging conditions and evaluate it against clinical ground truth.

### Milestone Summary

| Phase | Focus | Primary Output | Scientific Question | Status |
|-------|-------|---------------|--------------------|----|
| 1 | 1D boundary localization | [Interactive Streamlit demo](https://jsdqwgw4nefsm6flaphnjt.streamlit.app/) | Can FI/CRLB be visualized and understood from first principles? | 🟢 Live |
| 2 | 2D synthetic phantoms | Contour stability maps | Does the framework generalize to spatial boundary fields? | ⬜ Planned |
| 3 | Real medical image slices | Image-overlaid uncertainty maps | Is the method informative on real MRI/CT data? | ⬜ Planned |
| 4 | Model uncertainty comparison | Side-by-side uncertainty analyses | How does data uncertainty relate to model uncertainty? | ⬜ Planned |
| 5 | Clinical validation | Correlation with annotation disagreement | Does CRLB align with expert-perceived ambiguity? | ⬜ Planned |
| 6 | Research platform maturity | Reproducible codebase and documentation | Is this a complete, publishable research contribution? | ⬜ Planned |

---

### Phase 1 — 1D Boundary Localization Demo

**🟢 Live:** [https://jsdqwgw4nefsm6flaphnjt.streamlit.app/](https://jsdqwgw4nefsm6flaphnjt.streamlit.app/)

**Objective.** Establish and demonstrate the core theory in a fully controlled one-dimensional setting. The user observes a synthetic blurry edge — a step function convolved with a Gaussian PSF and corrupted by additive noise — and interactively varies contrast, blur width, noise level, and sampling density. The application computes and displays the log-likelihood surface as a function of boundary position, extracts its curvature to estimate Fisher Information, and derives the corresponding CRLB.

**Scientific value.** This phase proves that boundary uncertainty can be quantified rigorously from the image formation model alone, without reference to any algorithm. It establishes the theoretical foundation that all subsequent phases extend. It also provides the clearest possible pedagogical demonstration of the framework: a user who sees the CRLB widen as blur increases or contrast decreases has directly observed the informational consequences of image quality.

**Deliverables.**
- Polished Streamlit application with real-time parameter control
- Synthetic 1D edge simulator with configurable PSF and noise model
- Log-likelihood surface visualization and curvature analysis
- Fisher Information and CRLB plots as functions of image parameters
- Parameter sweep notebooks documenting behavior across the relevant regime

**Milestone completion criteria.** ✅ Met. The app correctly computes and displays FI and CRLB for a Gaussian-blurred step edge under Gaussian noise, with interactive parameter controls and interpretable visualization output. The demo is live at [https://jsdqwgw4nefsm6flaphnjt.streamlit.app/](https://jsdqwgw4nefsm6flaphnjt.streamlit.app/).

---

### Phase 2 — 2D Synthetic Phantom Stability Maps

**Objective.** Extend the 1D boundary localization framework to two-dimensional synthetic phantoms: circular lesions, irregular blobs, and multi-region structures embedded in noisy backgrounds. At each point along the contour, the local orthogonal intensity profile is extracted and used to compute a local Fisher Information value. The resulting per-point values are assembled into a **contour stability map** and visualized as a heatmap overlaid on the phantom image.

**Scientific value.** Phase 1 demonstrates the framework in a single scalar-parameter setting. Phase 2 demonstrates that the same reasoning applies pointwise along a spatially varying contour, producing a field of localization uncertainty rather than a single number. This is the step at which the method becomes a genuine analysis tool rather than a didactic illustration. It also enables systematic study of how phantom geometry — boundary curvature, lesion size, proximity to other structures — interacts with blur, noise, and contrast to determine local stability.

**Deliverables.**
- Synthetic phantom generator (circular and irregular lesion geometries)
- Orthogonal profile extraction and local FI computation along 2D contours
- Contour-level CRLB heatmap renderer
- Streamlit app or notebook interface for parameter exploration
- Experiment suite: blur radius, noise level, contrast, lesion size, partial-volume fraction

**Milestone completion criteria.** A 2D phantom with a known ground-truth boundary yields a spatially resolved stability map that visibly responds to local changes in image quality parameters. Map behavior matches theoretical expectations derived from Phase 1.

---

### Phase 3 — Real Medical Image Slice Demo

**Objective.** Apply an approximate local Fisher-information formulation to real MRI or CT image slices, using provided or publicly available segmentation boundaries (e.g., from the BraTS glioma dataset or a comparable CT benchmark). The method is applied pointwise along contours of interest, producing per-point CRLB estimates that are overlaid on the original image for clinical interpretation.

**Scientific value.** Synthetic phantoms provide ground truth and interpretive control, but they do not establish that the framework produces meaningful output on real data. Real medical images deviate from simple generative models: the PSF is anisotropic and scanner-specific, noise is non-Gaussian in many modalities, and tissue boundaries are rarely clean step edges. Phase 3 tests whether a locally applied, simplified FI formulation still provides discriminative and interpretable stability information in this setting. The expected finding is that CRLB estimates are visually correlated with clinically apparent image quality — high uncertainty near infiltrating tumor margins, low uncertainty near sharp organ capsules — though quantitative validation is deferred to Phase 5.

**Deliverables.**
- DICOM/NIfTI image loader and slice viewer
- Segmentation boundary overlay and contour extraction utilities
- Local FI/CRLB computation on real image profiles
- Stability heatmap visualization overlaid on imaging data
- Case studies illustrating high- and low-certainty boundary regions

**Milestone completion criteria.** At least three representative image slices from a public dataset yield CRLB maps that are visually interpretable and qualitatively consistent with image quality as assessed by inspection.

---

### Phase 4 — Comparison with Model-Based Uncertainty

**Objective.** Conduct paired experiments in which both the CRLB-based intrinsic uncertainty and a model-based uncertainty estimate (MC Dropout, deep ensemble, or probabilistic segmentation model) are computed for the same image and boundary. Visualizations and quantitative comparisons examine the spatial relationship between the two uncertainty types.

**Scientific value.** This is the phase that most directly positions the proposed framework relative to the existing literature. The central hypothesis is that data-supported uncertainty and model uncertainty are correlated but distinct: a well-calibrated model will tend to be more uncertain where the image is less informative, but the correspondence will not be perfect, and there will exist cases where a model is highly confident about a boundary that the CRLB identifies as poorly constrained. These cases are scientifically important — they represent regions where model confidence may be misleading and where the proposed framework provides information that no model-based approach can supply.

**Deliverables.**
- Integration with one or more probabilistic segmentation baselines
- MC Dropout and/or ensemble uncertainty computation pipeline
- Paired CRLB and model uncertainty maps on synthetic and real data
- Correlation analysis and case-study visualizations
- Characterization of agreement and disagreement regions

**Milestone completion criteria.** A quantitative comparison between CRLB-based and model-based uncertainty is produced for at least one dataset, with statistical summary and visual case studies demonstrating both correlated and divergent behavior.

---

### Phase 5 — Validation Against Human Ambiguity and Clinical Relevance

**Objective.** Compare CRLB-derived boundary uncertainty against empirical measures of annotation variability: inter-observer contour disagreement, segmentation editing burden, or expert-rated boundary difficulty. The question is whether the theoretical lower bound on localization precision predicts the uncertainty that human annotators actually experience.

**Scientific value.** This phase is where the method is tested against reality. If CRLB-identified high-uncertainty regions correspond to regions of elevated inter-observer disagreement, this validates the core hypothesis: that image-intrinsic information limits are a primary driver of annotation variability. If the correspondence is imperfect, that too is a finding — one that points toward the role of other factors such as clinical interpretation conventions, label definition, or annotator training. Either outcome is scientifically meaningful and publishable.

**Deliverables.**
- Inter-observer variability dataset integration (e.g., multi-annotator segmentation benchmarks)
- Boundary-point-level disagreement quantification
- Spatial correlation analysis between CRLB and disagreement metrics
- Clinician-facing visualization prototypes for boundary review support
- Statistical summary suitable for methods paper inclusion

**Milestone completion criteria.** A statistically characterized relationship between CRLB-derived uncertainty and annotation disagreement is established for at least one dataset, with results reported at the boundary-point level.

---

### Phase 6 — Thesis / Paper / Research Platform Maturity

**Objective.** Consolidate all prior phases into a coherent, reproducible, well-documented research codebase. Produce manuscript-quality figures, a reproducible experiment runner, comprehensive documentation, and a repository structure suitable for thesis submission or journal publication.

**Scientific value.** Research code that cannot be reproduced is of limited scientific value. This phase ensures that all experiments, simulations, and analyses from Phases 1–5 can be re-run from documented starting points, that figures are generated programmatically and labeled consistently, and that the repository communicates a complete and self-contained scientific argument. The goal is a codebase that a reviewer or examiner can clone, execute, and interrogate.

**Deliverables.**
- Cleaned, documented source library with stable public API
- Reproducible experiment runner with configuration files
- Manuscript-quality figure generation pipeline
- Comprehensive README, API documentation, and methods appendix
- Thesis-ready write-up of methods, experiments, and results

**Milestone completion criteria.** All major experiments can be reproduced from a single entry point. Figures match those reported in the associated manuscript or thesis chapter. Repository passes a cold-start reproducibility test on a clean environment.

---

## Current Status

The conceptual framework and repository architecture are established. **Phase 1 is complete and deployed.** The interactive 1D boundary localization demo is publicly accessible at [https://jsdqwgw4nefsm6flaphnjt.streamlit.app/](https://jsdqwgw4nefsm6flaphnjt.streamlit.app/) — it demonstrates Fisher Information and the CRLB in a fully controlled synthetic setting with real-time parameter control.

Phases 2 through 6 are designed and documented but not yet implemented. Each is planned as a milestone-driven research extension, with implementation beginning upon completion and validation of the preceding phase.

The repository is structured from the outset to support later phases without architectural refactoring, with shared source modules, a consistent data interface, and a figures pipeline designed for progressive extension.

---

## Repository Philosophy

This codebase sits at the intersection of three areas that rarely appear in the same repository:

- **Classical statistical estimation theory** — Fisher Information, the Cramér–Rao bound, maximum likelihood, and parametric inference
- **Medical image analysis** — segmentation, contouring, boundary uncertainty, clinical imaging modalities
- **Interpretable research tooling** — interactive simulation, reproducible experiments, visualization-first methodology

The project is motivated by the conviction that these areas have more to say to each other than the current literature reflects. Statistical estimation theory offers a language for principled uncertainty quantification that is model-free, assumption-transparent, and theoretically grounded. Medical image analysis has an urgent need for exactly this kind of tool. The gap between them is largely one of translation — and this repository is an attempt to build that bridge, one reproducible experiment at a time.

---

## Planned Capabilities

- **Interactive simulation dashboards** — parameter-driven visualization of FI and CRLB behavior in 1D and 2D
- **Synthetic boundary and phantom generation** — configurable lesion geometries with controlled blur, noise, contrast, and partial-volume fractions
- **Likelihood surface visualization** — direct observation of how image data constrains boundary position estimates
- **Fisher Information computation** — analytic and numerical FI calculation for parametric edge models
- **Cramér–Rao Lower Bound mapping** — scalar and spatially resolved CRLB computation along boundary curves
- **Local certainty heatmaps** — contour stability maps overlaid on image data for clinical interpretation
- **Model uncertainty comparison** — paired analysis of CRLB-based and learned uncertainty estimates
- **Real-image overlays** — boundary stability visualization on MRI and CT slices with segmentation imports
- **Reproducible experiment runner** — configuration-based experiment management for all phases

---

## Repository Structure

The following directory layout is used throughout the project. Each phase introduces new content within this stable structure.

```
quantifying-the-unknowable/
│
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
│
├── src/                          # Core library (shared across all phases)
│   ├── __init__.py
│   ├── models/                   # Parametric edge and image formation models
│   │   ├── edge_1d.py            #   1D blurred step edge model
│   │   ├── phantom_2d.py         #   2D synthetic lesion generator
│   │   └── noise.py              #   Noise model definitions
│   ├── estimation/               # Fisher Information and CRLB computation
│   │   ├── fisher.py             #   FI computation (analytic and numeric)
│   │   ├── crlb.py               #   CRLB derivation and mapping
│   │   └── likelihood.py         #   Log-likelihood surface utilities
│   ├── contour/                  # Boundary parameterization and profile extraction
│   │   ├── profiles.py           #   Orthogonal profile extraction from 2D images
│   │   ├── stability_map.py      #   Per-point CRLB field assembly
│   │   └── geometry.py           #   Boundary parameterization utilities
│   └── io/                       # Image loading and segmentation import
│       ├── nifti.py
│       └── dicom.py
│
├── app/                          # Streamlit applications (one per phase)
│   ├── phase1_1d_demo/
│   │   └── app.py
│   ├── phase2_phantom/
│   │   └── app.py
│   └── phase3_real_image/
│       └── app.py
│
├── notebooks/                    # Exploratory and expository notebooks
│   ├── 01_fisher_information_theory.ipynb
│   ├── 02_1d_edge_crlb_sweep.ipynb
│   ├── 03_2d_phantom_stability.ipynb
│   ├── 04_real_image_demo.ipynb
│   └── 05_model_comparison.ipynb
│
├── experiments/                  # Reproducible experiment scripts
│   ├── configs/                  #   YAML configuration files
│   ├── phase1_sweeps.py
│   ├── phase2_phantom_study.py
│   └── phase4_model_comparison.py
│
├── data/                         # Data directory (not tracked; see data/README.md)
│   ├── README.md                 #   Data acquisition and preparation instructions
│   ├── synthetic/                #   Generated phantom data
│   └── external/                 #   Public dataset links and download scripts
│
├── figures/                      # Generated figures (programmatically produced)
│   ├── phase1/
│   ├── phase2/
│   └── manuscript/
│
└── docs/                         # Extended documentation
    ├── theory.md                 #   Mathematical background
    ├── methods.md                #   Implementation details
    └── glossary.md               #   Key terms and notation
```

---

## How to Use This Repository

### Running the Phase 1 Demo

The Phase 1 demo is deployed and publicly accessible — no installation required:

**[→ Open the live app](https://jsdqwgw4nefsm6flaphnjt.streamlit.app/)**

Use the sidebar controls to vary contrast, blur width, noise level, and sampling density. The main panel displays the synthetic edge profile, the log-likelihood surface, the Fisher Information, and the CRLB in real time.

To run the app locally:

```bash
git clone https://github.com/<your-org>/quantifying-the-unknowable.git
cd quantifying-the-unknowable
pip install -r requirements.txt
streamlit run app/phase1_1d_demo/app.py
```

### Reproducing Synthetic Experiments

```bash
python experiments/phase1_sweeps.py --config experiments/configs/phase1_default.yaml
```

Output figures are saved to `figures/phase1/`. Experiment configurations are versioned in `experiments/configs/`.

### Exploring the Theory

The notebook `notebooks/01_fisher_information_theory.ipynb` provides a self-contained mathematical walkthrough of Fisher Information and the CRLB, with numerical examples and annotated derivations. It is designed to be readable without prior familiarity with the framework.

### Using the Core Library

```python
from src.models.edge_1d import BlurredStepEdge
from src.estimation.fisher import compute_fisher_information
from src.estimation.crlb import compute_crlb

edge = BlurredStepEdge(contrast=1.0, blur_sigma=2.0, noise_sigma=0.3)
profile = edge.sample(n_points=64)
fi = compute_fisher_information(profile, edge.model)
crlb = compute_crlb(fi)
```

---

## Longer-Term Research Vision

The phases described above represent a proof-of-concept trajectory from theory to clinical relevance. If the framework validates as expected, several extensions become natural:

**Toward informed uncertainty-aware segmentation.** A stability map produced before segmentation could be used to condition a model's loss function, flag high-uncertainty boundary regions for targeted augmentation, or guide active learning data collection toward the most informative image acquisitions.

**Toward quality assurance tooling.** A CRLB-based overlay applied to a completed segmentation could flag boundary segments for review, prioritize regions for manual correction, and provide quantitative justification for contour uncertainty in clinical reports.

**Toward evaluation reform.** If CRLB-derived uncertainty correlates with annotation disagreement as hypothesized, this motivates a rethinking of evaluation metrics: rather than penalizing a model uniformly for boundary error, metrics could weight disagreements by their information-theoretic expectedness. A model that errs in a low-information region is less culpable than one that errs where the image clearly supports precision.

**Toward multi-modal and 3D extension.** The core framework extends naturally to 3D surface stability and to modality-specific noise models (Rician noise in MRI, Poisson noise in PET, structured noise in ultrasound), each of which modifies the FI calculation in a principled, modality-appropriate way.

---

## Limitations and Scope

This project makes deliberate scope choices that should be understood before interpreting results.

**Model dependence.** The CRLB is computed under a parametric model of the image formation process. If that model is misspecified — for example, if the actual PSF is anisotropic or the noise is not Gaussian — the derived bound may not reflect true localization limits. Early phases use simplified Gaussian models; later phases will assess sensitivity to model assumptions.

**Proof-of-concept framing.** Phases 1 and 2 are explicitly proof-of-concept demonstrations. They establish feasibility and build intuition. They do not constitute clinical validation, and their results should not be interpreted as clinically validated uncertainty estimates.

**Lower bounds, not full posteriors.** The CRLB provides a lower bound on estimation variance; it does not provide a full posterior distribution over boundary positions. It answers the question "how good could any estimator be?" rather than "where is the boundary likely to be?" These are complementary but distinct questions.

**No annotation replacement.** Nothing in this framework generates or replaces segmentation annotations. It operates on existing segmentations or known boundaries to assess their data support. It does not propose to automate contouring.

**Computational approximations.** Local FI computation requires profile extraction, local model fitting, and numerical differentiation. On real images, these are approximations; the accuracy of the resulting CRLB depends on local stationarity assumptions that may not hold everywhere.

---

## Contributing and Collaboration

This repository is intended as a research platform, and contributions are welcome in several forms.

**Statistical methods.** If you have expertise in estimation theory, information geometry, or statistical signal processing and see opportunities to strengthen or extend the FI/CRLB formulation, please open an issue or reach out directly.

**Medical imaging.** Input from researchers familiar with specific imaging modalities (MRI, CT, PET, ultrasound) — particularly regarding realistic noise models, PSF characterization, and clinical boundary conventions — would significantly improve the realism of later phases.

**Clinical interpretation.** Feedback from clinicians or clinical researchers on how uncertainty maps should be presented, what boundary regions are clinically most relevant, and how the framework might integrate into existing workflows is actively sought.

**Reproducibility.** If you attempt to reproduce any experiment and encounter difficulty, please file a detailed issue. Reproducibility failures are treated as bugs.

To contribute, please fork the repository, open a branch, and submit a pull request with a clear description of the change and its motivation. For larger contributions or collaborations, opening a discussion issue first is encouraged.

---

## Citation

If this work informs your research, please consider citing the repository until a formal publication is available:

```bibtex
@misc{quantifying-the-unknowable,
  title  = {Quantifying the Unknowable: Fisher Information and the Cramér–Rao Lower Bound
             as Metrics for Segmentation Boundary Stability},
  author = {[Author]},
  year   = {2025},
  url    = {https://github.com/<your-org>/quantifying-the-unknowable}
}
```

---

## License

This repository is released under the [MIT License](LICENSE). Research use, extension, and adaptation are encouraged. Attribution is appreciated.

---

*This project is in active development. Phase 1 is complete and deployed. The framework and interfaces described for Phases 2–6 represent the intended design; sections marked as "Planned" have not yet been implemented.*
