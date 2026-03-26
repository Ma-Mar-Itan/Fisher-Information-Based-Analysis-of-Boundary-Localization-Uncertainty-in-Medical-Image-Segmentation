"""
Phase 1 Pilot: Fisher Information for Blurry Boundary Localization
A 1D estimation-theoretic demonstration of intrinsic boundary certainty in medical imaging.

Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import erf
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import io
import json

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG & CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Fisher Information for Boundary Localization",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400;0,8..60,600;0,8..60,700;1,8..60,400&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --accent: #1a6b7a;
    --accent-light: #e8f4f6;
    --accent-mid: #b8dce3;
    --text-primary: #1a1a2e;
    --text-secondary: #4a4a6a;
    --text-muted: #7a7a9a;
    --bg-white: #ffffff;
    --bg-panel: #f7f8fa;
    --bg-card: #fafbfc;
    --border: #e2e4ea;
    --border-light: #eef0f4;
    --warn-bg: #fff8f0;
    --warn-border: #f0d4a8;
    --success-bg: #f0faf4;
    --success-border: #a8dbb8;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--text-primary);
}

h1, h2, h3 {
    font-family: 'Source Serif 4', serif;
    font-weight: 600;
    color: var(--text-primary);
}

.main .block-container {
    padding-top: 2rem;
    max-width: 1100px;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: var(--bg-panel);
    border-right: 1px solid var(--border-light);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-secondary);
    margin-bottom: 0.25rem;
}

/* Metric card */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    text-align: center;
    transition: box-shadow 0.2s;
}
.metric-card:hover {
    box-shadow: 0 2px 12px rgba(26,107,122,0.08);
}
.metric-card .metric-label {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--text-muted);
    margin-bottom: 0.3rem;
}
.metric-card .metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.45rem;
    font-weight: 500;
    color: var(--accent);
}
.metric-card .metric-sub {
    font-size: 0.73rem;
    color: var(--text-muted);
    margin-top: 0.2rem;
}

/* Info callout */
.info-callout {
    background: var(--accent-light);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.88rem;
    line-height: 1.6;
    color: var(--text-secondary);
}

/* Formula box */
.formula-box {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.1rem 1.4rem;
    margin: 0.8rem 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    line-height: 1.8;
    color: var(--text-primary);
    text-align: center;
}

/* Section divider */
.section-divider {
    border: none;
    border-top: 1px solid var(--border-light);
    margin: 1.5rem 0;
}

/* Certainty gauge */
.certainty-gauge {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.9rem 1.2rem;
    border-radius: 8px;
    margin: 0.5rem 0;
}
.gauge-high {
    background: var(--success-bg);
    border: 1px solid var(--success-border);
}
.gauge-low {
    background: var(--warn-bg);
    border: 1px solid var(--warn-border);
}
.gauge-moderate {
    background: #f5f5ff;
    border: 1px solid #c8c8e8;
}
.gauge-label {
    font-weight: 600;
    font-size: 0.85rem;
}
.gauge-text {
    font-size: 0.8rem;
    color: var(--text-secondary);
}

/* Header subtitle */
.header-subtitle {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 1.02rem;
    font-weight: 300;
    color: var(--text-secondary);
    margin-top: -0.6rem;
    margin-bottom: 1.2rem;
    line-height: 1.5;
}

/* Tab styling refinement */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.02em;
}

/* Hide Streamlit footer */
footer {visibility: hidden;}

/* Expander refinement */
.streamlit-expanderHeader {
    font-size: 0.85rem;
    font-weight: 500;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# PLOTTING STYLE
# ═══════════════════════════════════════════════════════════════════

PLOT_COLORS = {
    "latent": "#1a6b7a",
    "blurred": "#d4782f",
    "noisy": "#8a8aaa",
    "theta_true": "#c0392b",
    "theta_mle": "#2471a3",
    "likelihood": "#1a6b7a",
    "fisher": "#d4782f",
    "crlb": "#c0392b",
    "fill": "#e8f4f6",
    "grid": "#eef0f4",
}


def setup_ax(ax, xlabel="", ylabel="", title=""):
    """Apply consistent styling to a matplotlib axis."""
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors="#666666", labelsize=8.5)
    ax.grid(True, alpha=0.3, color=PLOT_COLORS["grid"], linewidth=0.5)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color="#4a4a6a", fontweight=500)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color="#4a4a6a", fontweight=500)
    if title:
        ax.set_title(title, fontsize=10.5, color="#1a1a2e", fontweight=600, pad=10)


def fig_to_buf(fig):
    """Convert figure to bytes buffer for download."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf


# ═══════════════════════════════════════════════════════════════════
# CORE PHYSICS / MATH FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def generate_spatial_grid(n_points, x_min, x_max):
    """Generate a uniform 1D spatial grid."""
    return np.linspace(x_min, x_max, n_points)


def generate_latent_signal(x, theta, I_left, I_right, edge_model, blur_sigma):
    """
    Generate the mean signal mu(x; theta) for a boundary at position theta.

    Parameters
    ----------
    x : array, spatial grid
    theta : float, boundary location
    I_left, I_right : float, intensity levels
    edge_model : str, one of 'step', 'logistic', 'erf'
    blur_sigma : float, Gaussian blur width

    Returns
    -------
    latent : array, ideal step signal (pre-blur)
    blurred : array, signal after imaging blur
    """
    if edge_model == "Step (ideal)":
        latent = np.where(x < theta, I_left, I_right)
        # Blur via Gaussian filter
        dx = x[1] - x[0] if len(x) > 1 else 1.0
        sigma_px = max(blur_sigma / dx, 0.01)
        blurred = gaussian_filter1d(latent.astype(float), sigma=sigma_px, mode="nearest")
    elif edge_model == "Logistic (smooth)":
        k = 1.0 / max(blur_sigma, 1e-6)
        transition = 1.0 / (1.0 + np.exp(-k * (x - theta)))
        latent = I_left + (I_right - I_left) * np.heaviside(x - theta, 0.5)
        blurred = I_left + (I_right - I_left) * transition
    elif edge_model == "Error function":
        scale = max(blur_sigma, 1e-6) * np.sqrt(2)
        transition = 0.5 * (1.0 + erf((x - theta) / scale))
        latent = I_left + (I_right - I_left) * np.heaviside(x - theta, 0.5)
        blurred = I_left + (I_right - I_left) * transition
    else:
        latent = np.where(x < theta, I_left, I_right)
        blurred = latent.copy()

    return latent, blurred


def mean_signal(x, theta, I_left, I_right, edge_model, blur_sigma):
    """Return only the blurred (mean) signal for a given theta."""
    _, blurred = generate_latent_signal(x, theta, I_left, I_right, edge_model, blur_sigma)
    return blurred


def add_noise(signal, noise_std, rng):
    """Add Gaussian noise to a signal."""
    return signal + rng.normal(0, noise_std, size=signal.shape)


def compute_log_likelihood(observed, x, theta_grid, I_left, I_right, edge_model, blur_sigma, noise_std):
    """
    Compute log-likelihood of observed data for each candidate theta.

    Under Gaussian noise with known variance:
    log L(theta) = -0.5 * sum_j [(y_j - mu_j(theta))^2 / sigma^2] + const
    """
    sigma2 = noise_std ** 2
    ll = np.zeros(len(theta_grid))
    for i, th in enumerate(theta_grid):
        mu = mean_signal(x, th, I_left, I_right, edge_model, blur_sigma)
        residuals = observed - mu
        ll[i] = -0.5 * np.sum(residuals ** 2) / sigma2
    return ll


def compute_fisher_information(x, theta, I_left, I_right, edge_model, blur_sigma, noise_std, delta=1e-4):
    """
    Compute scalar Fisher Information for boundary location theta.

    I(theta) = (1/sigma^2) * sum_j [d mu_j / d theta]^2

    Uses central finite differences for the derivative.
    """
    mu_plus = mean_signal(x, theta + delta, I_left, I_right, edge_model, blur_sigma)
    mu_minus = mean_signal(x, theta - delta, I_left, I_right, edge_model, blur_sigma)
    dmu_dtheta = (mu_plus - mu_minus) / (2 * delta)
    fi = np.sum(dmu_dtheta ** 2) / (noise_std ** 2)
    return fi, dmu_dtheta


def compute_crlb(fisher_info):
    """CRLB = 1 / I(theta) for scalar parameter."""
    if fisher_info > 1e-15:
        return 1.0 / fisher_info
    return np.inf


# ═══════════════════════════════════════════════════════════════════
# UI HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def metric_card(label, value, sub=""):
    """Render a styled metric card."""
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {sub_html}
        </div>""",
        unsafe_allow_html=True,
    )


def info_callout(text):
    st.markdown(f'<div class="info-callout">{text}</div>', unsafe_allow_html=True)


def formula_box(text):
    st.markdown(f'<div class="formula-box">{text}</div>', unsafe_allow_html=True)


def section_divider():
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)


def certainty_gauge(fi_val, crlb_val):
    """Render a qualitative certainty indicator."""
    best_std = np.sqrt(crlb_val) if np.isfinite(crlb_val) else np.inf
    if fi_val > 500:
        cls, icon, label, txt = "gauge-high", "●", "High certainty", f"Best achievable σ ≈ {best_std:.4f} — boundary is well-constrained by the data."
    elif fi_val > 50:
        cls, icon, label, txt = "gauge-moderate", "◐", "Moderate certainty", f"Best achievable σ ≈ {best_std:.4f} — boundary position is estimable but with notable uncertainty."
    else:
        cls, icon, label, txt = "gauge-low", "○", "Low certainty", f"Best achievable σ ≈ {best_std:.3f} — the data provide weak constraints on boundary location."
    st.markdown(
        f"""<div class="certainty-gauge {cls}">
            <span style="font-size:1.4rem">{icon}</span>
            <div><div class="gauge-label">{label}</div><div class="gauge-text">{txt}</div></div>
        </div>""",
        unsafe_allow_html=True,
    )


def dynamic_interpretation(fi_val, crlb_val, contrast, blur_sigma, noise_std):
    """Generate plain-language interpretation of the current scenario."""
    best_std = np.sqrt(crlb_val) if np.isfinite(crlb_val) else np.inf
    parts = []
    # Contrast
    if contrast > 0.6:
        parts.append("The intensity contrast across the boundary is strong")
    elif contrast > 0.3:
        parts.append("The intensity contrast is moderate")
    else:
        parts.append("The intensity contrast is weak")
    # Blur
    if blur_sigma < 0.3:
        parts.append("blur is minimal")
    elif blur_sigma < 1.0:
        parts.append("blur moderately smooths the transition")
    else:
        parts.append("blur substantially broadens the transition")
    # Noise
    if noise_std < 0.05:
        parts.append("and noise is low")
    elif noise_std < 0.15:
        parts.append("and noise is moderate")
    else:
        parts.append("and noise is high")

    summary = ", ".join(parts) + "."

    if fi_val > 500:
        conclusion = f"Under these conditions, the Fisher Information is high ({fi_val:.1f}) and the CRLB is small ({crlb_val:.6f}). The image data strongly constrain the boundary location — any unbiased estimator can localize the edge with a theoretical best standard deviation of {best_std:.4f} spatial units."
    elif fi_val > 50:
        conclusion = f"The Fisher Information is moderate ({fi_val:.1f}), yielding a CRLB of {crlb_val:.4f}. The boundary is estimable but with non-trivial uncertainty. Best achievable standard deviation: {best_std:.4f}."
    else:
        conclusion = f"The Fisher Information is low ({fi_val:.2f}), yielding a large CRLB of {crlb_val:.3f}. The data provide weak evidence for precise boundary localization. Best achievable standard deviation: {best_std:.3f}. A segmentation model producing a crisp contour here would be asserting precision not supported by the image."

    return summary, conclusion


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ◈ Simulation Controls")
    st.caption("Adjust parameters to explore how imaging conditions affect boundary identifiability.")

    # --- Presets ---
    st.markdown("---")
    st.markdown("##### Presets")
    preset = st.selectbox(
        "Load a scenario",
        ["Custom", "Sharp / high-contrast / low-noise", "Blurry / low-contrast / high-noise", "Moderate realistic"],
        index=0,
        help="Load a predefined scenario to quickly see different regimes.",
    )

    preset_vals = {
        "Sharp / high-contrast / low-noise": dict(I_left=0.1, I_right=1.0, blur_sigma=0.15, noise_std=0.03, n_points=200),
        "Blurry / low-contrast / high-noise": dict(I_left=0.35, I_right=0.65, blur_sigma=1.8, noise_std=0.18, n_points=100),
        "Moderate realistic": dict(I_left=0.2, I_right=0.8, blur_sigma=0.6, noise_std=0.08, n_points=150),
    }

    pv = preset_vals.get(preset, {})

    # --- Boundary / Signal ---
    st.markdown("---")
    st.markdown("##### Boundary & Signal")
    I_left = st.slider(
        "Left intensity (I₁)",
        0.0, 1.0, pv.get("I_left", 0.2), 0.01,
        help="Mean intensity on the left side of the boundary (e.g., tissue class A).",
    )
    I_right = st.slider(
        "Right intensity (I₂)",
        0.0, 1.0, pv.get("I_right", 1.0), 0.01,
        help="Mean intensity on the right side of the boundary (e.g., tissue class B).",
    )
    theta_true = st.slider(
        "True boundary position (θ)",
        -3.0, 3.0, 0.0, 0.05,
        help="The true location of the boundary on the spatial axis.",
    )
    edge_model = st.selectbox(
        "Edge model",
        ["Step (ideal)", "Logistic (smooth)", "Error function"],
        index=2,
        help="Step: ideal discontinuity blurred externally. Logistic/Erf: analytically smooth transitions parameterized by blur σ.",
    )

    # --- Imaging Physics ---
    st.markdown("---")
    st.markdown("##### Imaging Physics")
    blur_sigma = st.slider(
        "Blur σ (point spread)",
        0.01, 3.0, pv.get("blur_sigma", 0.5), 0.01,
        help="Width of the Gaussian point-spread function. Larger values broaden the transition and reduce local gradient, weakening boundary information.",
    )

    # --- Noise ---
    st.markdown("---")
    st.markdown("##### Noise Model")
    noise_std = st.slider(
        "Noise σ",
        0.001, 0.5, pv.get("noise_std", 0.08), 0.001,
        format="%.3f",
        help="Standard deviation of additive Gaussian noise. Higher noise reduces the ability to localize the boundary from observed data.",
    )
    n_realizations = st.slider(
        "Noisy realizations to display",
        1, 10, 3,
        help="Number of independent noisy observations to visualize.",
    )

    # --- Sampling ---
    st.markdown("---")
    st.markdown("##### Sampling & Domain")
    n_points = st.slider(
        "Sample points",
        30, 500, pv.get("n_points", 150), 10,
        help="Number of discrete spatial samples. Denser sampling generally provides more information about the transition region.",
    )
    x_range = st.slider(
        "Spatial range (±)",
        1.0, 8.0, 4.0, 0.5,
        help="Half-width of the spatial domain.",
    )
    seed = st.number_input("Random seed", value=42, step=1, help="Seed for reproducible noise realizations.")

    # --- Estimation ---
    st.markdown("---")
    st.markdown("##### Estimation Grid")
    n_theta = st.slider(
        "Candidate θ values",
        50, 500, 200, 10,
        help="Resolution of the boundary-position grid used for likelihood evaluation.",
    )

    # --- Display ---
    st.markdown("---")
    st.markdown("##### Display Options")
    show_truth = st.checkbox("Show true θ marker", True)
    show_annotations = st.checkbox("Show plot annotations", True)
    show_gradient = st.checkbox("Show local gradient curve", False)

    st.markdown("---")
    st.caption("Phase 1 Pilot · Proof-of-concept for estimation-theoretic segmentation uncertainty")


# ═══════════════════════════════════════════════════════════════════
# COMPUTE EVERYTHING
# ═══════════════════════════════════════════════════════════════════

rng = np.random.default_rng(int(seed))
x = generate_spatial_grid(n_points, -x_range, x_range)
latent, blurred = generate_latent_signal(x, theta_true, I_left, I_right, edge_model, blur_sigma)

# Generate noisy realizations
noisy_list = [add_noise(blurred, noise_std, np.random.default_rng(int(seed) + i)) for i in range(n_realizations)]
observed_primary = noisy_list[0]

# Likelihood
theta_grid = np.linspace(-x_range * 0.8, x_range * 0.8, n_theta)
ll = compute_log_likelihood(observed_primary, x, theta_grid, I_left, I_right, edge_model, blur_sigma, noise_std)
ll_norm = ll - ll.max()  # normalize for display
mle_idx = np.argmax(ll)
theta_mle = theta_grid[mle_idx]

# Fisher Information & CRLB
fi_val, dmu_dtheta = compute_fisher_information(x, theta_true, I_left, I_right, edge_model, blur_sigma, noise_std)
crlb_val = compute_crlb(fi_val)
best_std = np.sqrt(crlb_val) if np.isfinite(crlb_val) else np.inf
contrast = abs(I_right - I_left)


# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════

st.markdown("# Phase 1 Pilot: Fisher Information for Blurry Boundary Localization")
st.markdown(
    '<p class="header-subtitle">A 1D estimation-theoretic demonstration of intrinsic boundary certainty in medical imaging</p>',
    unsafe_allow_html=True,
)

st.markdown(
    "This application simulates a one-dimensional tissue boundary and quantifies "
    "how blur, contrast, and noise affect the theoretical precision with which boundary "
    "position can be estimated from image data alone."
)

with st.expander("◈  Why this matters", expanded=False):
    st.markdown("""
**Segmentation systems often produce crisp boundaries even when the image data itself is ambiguous.**
This pilot study introduces a physics-based framework for quantifying that ambiguity.

- **Fisher Information** measures how sensitively the observed signal responds to shifts in the boundary position. Higher sensitivity means more precise estimability.
- **The Cramér–Rao Lower Bound (CRLB)** gives the smallest variance any unbiased estimator could theoretically achieve — it is a fundamental limit imposed by the data, not the model.
- This demonstration is a **proof-of-concept** for identifying "inherent uncertainty zones" from image statistics rather than from black-box model confidence alone.

The goal is to distinguish *model failure* from *information-limited ambiguity* — a distinction rarely made in current segmentation evaluation.
""")

section_divider()


# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview", "Signal Simulation", "Likelihood & Estimation",
    "Fisher Information & CRLB", "Interpretation", "Technical Notes"
])


# ───────────────────────────────────────────────────────────────────
# TAB 1: OVERVIEW
# ───────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Conceptual Overview")

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("""
This pilot study examines a foundational question in medical image segmentation:

> **How precisely can a boundary location be determined from the image data itself?**

In clinical imaging, tissue boundaries are obscured by noise, blur, partial-volume effects, and limited spatial resolution. A segmentation model may output a deterministic contour, but that contour's precision is ultimately limited by the information content of the acquired signal.

This application demonstrates that limit using a simple 1D boundary between two tissue classes. By varying contrast, blur, and noise, you can observe how the **Fisher Information** — a measure of data informativeness — and the **CRLB** — the best-achievable estimation variance — respond to imaging conditions.
""")

    with col_r:
        st.markdown("#### Key Concepts")
        info_callout(
            "<b>Latent boundary</b> — the true tissue transition before imaging degradation.<br>"
            "<b>Blur</b> — point-spread smoothing that broadens the transition.<br>"
            "<b>Noise</b> — random fluctuations from detector/acquisition physics.<br>"
            "<b>Likelihood</b> — probability of the observed data given a candidate boundary position.<br>"
            "<b>Fisher Information</b> — sensitivity of the likelihood to boundary-position changes.<br>"
            "<b>CRLB</b> — minimum variance for any unbiased boundary-location estimator."
        )

    section_divider()

    st.markdown("#### What This App Demonstrates")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Sharp edges** → high contrast + low blur + low noise → **high Fisher Information**, **low CRLB**. The data strongly support precise localization.")
    with c2:
        st.markdown("**Blurry edges** → low contrast + high blur + high noise → **low Fisher Information**, **high CRLB**. No estimator can localize the boundary well.")
    with c3:
        st.markdown("**The CRLB** is a property of the *data-generating process*, not of any particular model or algorithm. It reveals when uncertainty is intrinsic.")

    section_divider()

    st.markdown("#### Core Formulas")
    formula_box(
        "Fisher Information: &nbsp; I(θ) = (1/σ²) Σⱼ [∂μⱼ(θ)/∂θ]²<br><br>"
        "Cramér–Rao Lower Bound: &nbsp; Var(θ̂) ≥ 1 / I(θ)"
    )
    st.caption("For scalar θ under Gaussian noise with known variance σ² and mean signal μ(x; θ).")


# ───────────────────────────────────────────────────────────────────
# TAB 2: SIGNAL SIMULATION
# ───────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Signal Simulation")
    st.markdown("Visualize the clean latent edge, the blurred transition, and noisy observed profiles under the current parameter settings.")

    # Main signal plot
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=120)
    fig.patch.set_facecolor("white")

    ax.plot(x, latent, color=PLOT_COLORS["latent"], linewidth=1.2, linestyle="--", alpha=0.5, label="Latent (ideal)")
    ax.plot(x, blurred, color=PLOT_COLORS["latent"], linewidth=2.2, label="Blurred mean signal")
    for i, noisy in enumerate(noisy_list):
        alpha = 0.45 if n_realizations > 1 else 0.7
        label = "Noisy observation(s)" if i == 0 else None
        ax.plot(x, noisy, color=PLOT_COLORS["noisy"], linewidth=0.7, alpha=alpha, label=label)
    if show_truth:
        ax.axvline(theta_true, color=PLOT_COLORS["theta_true"], linewidth=1.2, linestyle=":", alpha=0.7, label=f"True θ = {theta_true:.2f}")

    setup_ax(ax, xlabel="Spatial position (x)", ylabel="Intensity", title="1D Boundary: Latent, Blurred, and Observed Signals")
    ax.legend(fontsize=7.5, loc="best", framealpha=0.9, edgecolor="#dddddd")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # Download button
    buf = fig_to_buf(fig)
    st.download_button("Download signal plot (PNG)", buf, "signal_simulation.png", "image/png")
    plt.close(fig)

    # Optional gradient subplot
    if show_gradient:
        fig_g, ax_g = plt.subplots(figsize=(9, 2.5), dpi=120)
        fig_g.patch.set_facecolor("white")
        grad = np.gradient(blurred, x)
        ax_g.fill_between(x, 0, grad, alpha=0.25, color=PLOT_COLORS["latent"])
        ax_g.plot(x, grad, color=PLOT_COLORS["latent"], linewidth=1.5)
        if show_truth:
            ax_g.axvline(theta_true, color=PLOT_COLORS["theta_true"], linewidth=1, linestyle=":", alpha=0.6)
        setup_ax(ax_g, xlabel="Spatial position (x)", ylabel="∂μ/∂x", title="Local Intensity Gradient")
        plt.tight_layout()
        st.pyplot(fig_g, use_container_width=True)
        plt.close(fig_g)

    section_divider()

    # Parameter summary
    st.markdown("##### Current Parameters")
    pc1, pc2, pc3, pc4 = st.columns(4)
    with pc1:
        metric_card("Contrast", f"{contrast:.2f}", "| I₂ − I₁ |")
    with pc2:
        metric_card("Blur σ", f"{blur_sigma:.2f}", "PSF width")
    with pc3:
        metric_card("Noise σ", f"{noise_std:.3f}", "Gaussian std")
    with pc4:
        metric_card("Samples", f"{n_points}", "spatial points")

    info_callout(
        "<b>Clinical intuition:</b> In real medical images, sharper transitions between tissues provide more reliable contour placement. "
        "Broad transitions caused by blur or partial-volume effects make the exact boundary inherently ambiguous — "
        "this is a property of the acquired data, not of the segmentation algorithm."
    )


# ───────────────────────────────────────────────────────────────────
# TAB 3: LIKELIHOOD & ESTIMATION
# ───────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Likelihood & Boundary Estimation")
    st.markdown("The log-likelihood surface shows how plausible each candidate boundary position is given the observed noisy signal.")

    fig2, ax2 = plt.subplots(figsize=(9, 4), dpi=120)
    fig2.patch.set_facecolor("white")

    ax2.fill_between(theta_grid, ll_norm, alpha=0.15, color=PLOT_COLORS["likelihood"])
    ax2.plot(theta_grid, ll_norm, color=PLOT_COLORS["likelihood"], linewidth=2)
    if show_truth:
        ax2.axvline(theta_true, color=PLOT_COLORS["theta_true"], linewidth=1.2, linestyle=":", label=f"True θ = {theta_true:.2f}")
    ax2.axvline(theta_mle, color=PLOT_COLORS["theta_mle"], linewidth=1.5, linestyle="--", label=f"MLE θ̂ = {theta_mle:.3f}")

    if show_annotations:
        ax2.annotate(
            f"MLE = {theta_mle:.3f}",
            xy=(theta_mle, 0), xytext=(theta_mle + x_range * 0.1, ll_norm.min() * 0.3),
            fontsize=8, color=PLOT_COLORS["theta_mle"],
            arrowprops=dict(arrowstyle="->", color=PLOT_COLORS["theta_mle"], lw=0.8),
        )

    setup_ax(ax2, xlabel="Candidate boundary position (θ)", ylabel="Log-likelihood (normalized)", title="Log-Likelihood Surface for Boundary Position")
    ax2.legend(fontsize=7.5, framealpha=0.9, edgecolor="#dddddd")
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    buf2 = fig_to_buf(fig2)
    st.download_button("Download likelihood plot (PNG)", buf2, "likelihood.png", "image/png")
    plt.close(fig2)

    section_divider()

    # Metrics
    st.markdown("##### Estimation Summary")
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        metric_card("True θ", f"{theta_true:.3f}", "ground truth")
    with mc2:
        metric_card("MLE θ̂", f"{theta_mle:.3f}", "maximum likelihood")
    with mc3:
        metric_card("Error", f"{abs(theta_mle - theta_true):.4f}", "|θ̂ − θ|")
    with mc4:
        metric_card("Peak log-L", f"{ll.max():.1f}", "unnormalized")

    info_callout(
        "<b>Interpretation:</b> A sharper likelihood peak means the data strongly constrain boundary location — "
        "only a narrow range of θ values is consistent with the observations. A flatter peak means many boundary "
        "positions are similarly plausible, indicating intrinsic ambiguity. The curvature at the peak is directly "
        "related to the Fisher Information."
    )


# ───────────────────────────────────────────────────────────────────
# TAB 4: FISHER INFORMATION & CRLB
# ───────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### Fisher Information & Cramér–Rao Lower Bound")
    st.markdown("The central results: how informative is the image data about the boundary location?")

    # Primary metrics
    fm1, fm2, fm3 = st.columns(3)
    with fm1:
        metric_card("Fisher Information", f"{fi_val:.2f}", "I(θ)")
    with fm2:
        metric_card("CRLB", f"{crlb_val:.6f}" if crlb_val < 1 else f"{crlb_val:.4f}", "Var(θ̂) ≥ CRLB")
    with fm3:
        metric_card("Best σ", f"{best_std:.4f}" if best_std < 1 else f"{best_std:.3f}", "√CRLB")

    certainty_gauge(fi_val, crlb_val)

    section_divider()

    # Sensitivity contribution plot
    st.markdown("##### Sensitivity Decomposition")
    st.markdown("The plot below shows the squared derivative (∂μ/∂θ)² at each sample point — this is the per-sample contribution to Fisher Information before normalization by noise variance.")

    fig3, ax3 = plt.subplots(figsize=(9, 3.5), dpi=120)
    fig3.patch.set_facecolor("white")

    sensitivity = dmu_dtheta ** 2
    ax3.fill_between(x, 0, sensitivity, alpha=0.3, color=PLOT_COLORS["fisher"])
    ax3.plot(x, sensitivity, color=PLOT_COLORS["fisher"], linewidth=1.8)
    if show_truth:
        ax3.axvline(theta_true, color=PLOT_COLORS["theta_true"], linewidth=1, linestyle=":", alpha=0.6)
    setup_ax(ax3, xlabel="Spatial position (x)", ylabel="(∂μ/∂θ)²", title="Per-Sample Sensitivity to Boundary Shift")
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    st.caption("Sensitivity is concentrated near the boundary where intensity changes most rapidly with θ. Samples far from the edge contribute negligibly.")

    section_divider()

    # Parameter sweep: CRLB vs blur
    st.markdown("##### Parameter Sweep: CRLB vs. Blur and Noise")
    st.markdown("How does the minimum achievable localization variance change as imaging conditions degrade?")

    sweep_col1, sweep_col2 = st.columns(2)

    with sweep_col1:
        blur_range = np.linspace(0.05, 3.0, 60)
        crlb_vs_blur = []
        for bs in blur_range:
            fi_tmp, _ = compute_fisher_information(x, theta_true, I_left, I_right, edge_model, bs, noise_std)
            crlb_vs_blur.append(compute_crlb(fi_tmp))
        crlb_vs_blur = np.array(crlb_vs_blur)
        best_std_blur = np.sqrt(np.clip(crlb_vs_blur, 0, 100))

        fig4, ax4 = plt.subplots(figsize=(4.5, 3.2), dpi=120)
        fig4.patch.set_facecolor("white")
        ax4.semilogy(blur_range, best_std_blur, color=PLOT_COLORS["crlb"], linewidth=2)
        ax4.axvline(blur_sigma, color="#aaaaaa", linewidth=1, linestyle="--", alpha=0.7)
        if show_annotations:
            ax4.annotate("current", xy=(blur_sigma, best_std), fontsize=7.5, color="#888888",
                         xytext=(blur_sigma + 0.2, best_std * 1.5),
                         arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=0.6))
        setup_ax(ax4, xlabel="Blur σ", ylabel="Best σ (√CRLB)", title="Localization Limit vs. Blur")
        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)

    with sweep_col2:
        noise_range = np.linspace(0.005, 0.5, 60)
        crlb_vs_noise = []
        for ns in noise_range:
            fi_tmp, _ = compute_fisher_information(x, theta_true, I_left, I_right, edge_model, blur_sigma, ns)
            crlb_vs_noise.append(compute_crlb(fi_tmp))
        crlb_vs_noise = np.array(crlb_vs_noise)
        best_std_noise = np.sqrt(np.clip(crlb_vs_noise, 0, 100))

        fig5, ax5 = plt.subplots(figsize=(4.5, 3.2), dpi=120)
        fig5.patch.set_facecolor("white")
        ax5.semilogy(noise_range, best_std_noise, color=PLOT_COLORS["crlb"], linewidth=2)
        ax5.axvline(noise_std, color="#aaaaaa", linewidth=1, linestyle="--", alpha=0.7)
        if show_annotations:
            ax5.annotate("current", xy=(noise_std, best_std), fontsize=7.5, color="#888888",
                         xytext=(noise_std + 0.05, best_std * 1.5),
                         arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=0.6))
        setup_ax(ax5, xlabel="Noise σ", ylabel="Best σ (√CRLB)", title="Localization Limit vs. Noise")
        plt.tight_layout()
        st.pyplot(fig5, use_container_width=True)
        plt.close(fig5)

    section_divider()

    # 2D heatmap: CRLB as function of blur and noise
    st.markdown("##### CRLB Heatmap: Blur × Noise")
    blur_hm = np.linspace(0.1, 2.5, 30)
    noise_hm = np.linspace(0.01, 0.35, 30)
    crlb_hm = np.zeros((len(noise_hm), len(blur_hm)))
    for i, ns in enumerate(noise_hm):
        for j, bs in enumerate(blur_hm):
            fi_tmp, _ = compute_fisher_information(x, theta_true, I_left, I_right, edge_model, bs, ns)
            crlb_hm[i, j] = np.sqrt(compute_crlb(fi_tmp))

    crlb_hm_clipped = np.clip(crlb_hm, 0, np.percentile(crlb_hm, 98))

    fig6, ax6 = plt.subplots(figsize=(7, 4.5), dpi=120)
    fig6.patch.set_facecolor("white")
    im = ax6.imshow(
        crlb_hm_clipped, aspect="auto", origin="lower",
        extent=[blur_hm[0], blur_hm[-1], noise_hm[0], noise_hm[-1]],
        cmap="YlOrRd", interpolation="bilinear",
    )
    ax6.plot(blur_sigma, noise_std, "o", color="white", markersize=8, markeredgecolor="black", markeredgewidth=1.5, zorder=5)
    if show_annotations:
        ax6.annotate("current", xy=(blur_sigma, noise_std), fontsize=8, color="white", fontweight="bold",
                     xytext=(blur_sigma + 0.15, noise_std + 0.02))
    cbar = plt.colorbar(im, ax=ax6, shrink=0.85)
    cbar.set_label("Best achievable σ (√CRLB)", fontsize=8.5)
    setup_ax(ax6, xlabel="Blur σ", ylabel="Noise σ", title="Localization Precision Limit Across Imaging Conditions")
    plt.tight_layout()
    st.pyplot(fig6, use_container_width=True)
    buf6 = fig_to_buf(fig6)
    st.download_button("Download heatmap (PNG)", buf6, "crlb_heatmap.png", "image/png")
    plt.close(fig6)

    st.caption("Warmer colors indicate worse localization precision. The white dot marks the current parameter setting.")

    section_divider()

    st.markdown("##### Why This Matters for Segmentation")
    info_callout(
        "In real medical imaging, a deep learning model may output a crisp contour. "
        "But if the local Fisher Information is low, that contour position is <b>not strongly supported by the data</b>. "
        "The CRLB provides a principled, model-independent lower bound on localization uncertainty. "
        "This pilot study is a first step toward constructing <b>local stability maps</b> for segmentation boundaries."
    )


# ───────────────────────────────────────────────────────────────────
# TAB 5: INTERPRETATION & TAKEAWAYS
# ───────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("### Interpretation & Research Implications")

    summary_text, conclusion_text = dynamic_interpretation(fi_val, crlb_val, contrast, blur_sigma, noise_std)

    st.markdown("#### What the Simulation Shows")
    st.markdown(f"**Current scenario:** {summary_text}")
    st.markdown(conclusion_text)

    section_divider()

    st.markdown("#### Research Significance")
    st.markdown("""
This demonstration establishes a **boundary-specific notion of intrinsic certainty** grounded in estimation theory rather than model confidence. Key implications:

1. **Data-driven uncertainty**: The CRLB quantifies a fundamental limit imposed by imaging physics. It is not a property of any particular segmentation algorithm.
2. **Complementary to model uncertainty**: Model-based measures (Monte Carlo dropout, ensemble disagreement) capture epistemic uncertainty. The CRLB captures the irreducible component — they are related but distinct.
3. **Foundation for stability maps**: In higher dimensions, local Fisher Information along a contour could produce a spatial map showing where the image supports precise delineation and where it does not.
4. **Clinical relevance**: In radiotherapy contouring, knowing that a boundary is intrinsically uncertain is as important as knowing the contour itself. A CRLB overlay could support peer review, adaptive planning, and human oversight.
""")

    section_divider()

    st.markdown("#### Limitations of This Pilot")
    st.markdown("""
- **1D simplification** — real segmentation boundaries are 2D curves or 3D surfaces.
- **Scalar boundary parameter** — real contours require multi-parameter representations (spline control points, level-set coefficients).
- **Idealized noise model** — Gaussian noise is a starting point; MRI magnitude data follow a Rician distribution, especially at low SNR.
- **Known noise variance** — in practice, σ must be estimated from the data.
- **Unbiased-estimator assumption** — the classical CRLB applies to unbiased estimators; regularized and deep learning methods are typically biased.
- **No partial-volume modeling** — voxel-level tissue mixing is not modeled in this 1D setup.
""")

    section_divider()

    st.markdown("#### Next Phase")
    st.markdown("""
The natural extensions of this pilot are:

- **2D lesion phantoms**: elliptical or irregular boundaries with spatially varying blur and contrast.
- **Local contour parameterization**: represent the boundary via normal displacement at each contour point.
- **Stability heatmaps**: compute Fisher Information along the contour and project it as a color-coded overlay.
- **Real MRI/CT data**: apply the framework to clinical images using modality-specific noise models.
- **Comparison with model uncertainty**: evaluate whether CRLB-derived maps explain uncertainty not captured by dropout or ensemble methods.
""")

    section_divider()

    st.markdown("#### Suggested Thesis or Presentation Wording")
    info_callout(
        "<i>\"We demonstrate that boundary localization precision is fundamentally limited by the information content "
        "of the image signal near the boundary. Using Fisher Information and the Cramér–Rao Lower Bound, we "
        "quantify the best achievable localization variance under a parametric observation model. "
        "This estimation-theoretic framework provides a principled, model-independent measure of segmentation "
        "stability that complements existing overlap metrics and model-based uncertainty quantification.\"</i>"
    )
    info_callout(
        "<i>\"The proposed stability map identifies regions where any unbiased boundary estimator must have high "
        "variance because the raw image data are intrinsically uninformative. This reframes segmentation uncertainty "
        "from a model-only property into a property of the image-data-generating process itself.\"</i>"
    )


# ───────────────────────────────────────────────────────────────────
# TAB 6: TECHNICAL NOTES
# ───────────────────────────────────────────────────────────────────
with tab6:
    st.markdown("### Technical Notes")
    st.markdown("Mathematical details and implementation choices for this pilot demonstration.")

    st.markdown("#### Signal Model")
    st.markdown(r"""
The 1D boundary is modeled as a transition between two intensity levels $I_1$ (left) and $I_2$ (right) at position $\theta$. Three edge models are supported:

**Step (ideal):** A discontinuous step at $\theta$, convolved with a Gaussian kernel of width $\sigma_{\text{blur}}$.

**Logistic:** $\mu(x;\theta) = I_1 + (I_2 - I_1) \cdot \frac{1}{1 + e^{-(x-\theta)/\sigma_{\text{blur}}}}$

**Error function:** $\mu(x;\theta) = I_1 + (I_2 - I_1) \cdot \frac{1}{2}\left[1 + \text{erf}\!\left(\frac{x-\theta}{\sigma_{\text{blur}}\sqrt{2}}\right)\right]$

All three produce a smooth transition whose sharpness is controlled by the blur parameter $\sigma_{\text{blur}}$.
""")

    st.markdown("#### Noise Model")
    st.markdown(r"""
Observations are modeled as independent Gaussian samples:

$$I_j \sim \mathcal{N}(\mu_j(\theta),\, \sigma^2)$$

where $\sigma$ is the noise standard deviation (assumed known). This is a standard starting point; extension to Rician noise for MRI is a planned next step.
""")

    st.markdown("#### Log-Likelihood")
    st.markdown(r"""
Under the Gaussian model with known variance, the log-likelihood for observed data $\{y_j\}$ given boundary parameter $\theta$ is:

$$\log L(\theta) = -\frac{1}{2\sigma^2} \sum_{j=1}^{N} (y_j - \mu_j(\theta))^2 + \text{const.}$$
""")

    st.markdown("#### Fisher Information")
    st.markdown(r"""
For scalar $\theta$ under the Gaussian model:

$$I(\theta) = \frac{1}{\sigma^2} \sum_{j=1}^{N} \left(\frac{\partial \mu_j(\theta)}{\partial \theta}\right)^2$$

The derivative $\partial \mu_j / \partial \theta$ is computed via central finite differences with step size $\delta = 10^{-4}$.
""")

    st.markdown("#### Cramér–Rao Lower Bound")
    st.markdown(r"""
For any unbiased estimator $\hat{\theta}$:

$$\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}$$

This is the minimum achievable variance — a theoretical floor that no unbiased estimator can beat, regardless of algorithmic sophistication.
""")

    section_divider()

    st.markdown("#### Implementation Caveats")
    st.markdown("""
- **Expected vs. observed Fisher Information**: This implementation uses the *expected* Fisher Information, computed from the mean signal model. The *observed* information (computed from a specific realization) would differ sample-to-sample.
- **Finite-difference derivatives**: Analytic derivatives could be used for the logistic and erf models; finite differences are used here for generality and simplicity.
- **Regularization**: If the FIM approaches singularity (e.g., zero contrast), the CRLB diverges. This is handled by returning infinity rather than attempting inversion.
- **Bias**: The CRLB strictly applies to unbiased estimators. In practice, maximum likelihood estimators are asymptotically unbiased and efficient, so the bound is typically tight for moderate-to-large sample sizes.
""")

    section_divider()

    # Export simulation data
    st.markdown("#### Export Simulation Data")
    export_data = {
        "parameters": {
            "I_left": float(I_left),
            "I_right": float(I_right),
            "theta_true": float(theta_true),
            "blur_sigma": float(blur_sigma),
            "noise_std": float(noise_std),
            "n_points": int(n_points),
            "edge_model": edge_model,
            "seed": int(seed),
        },
        "results": {
            "fisher_information": float(fi_val),
            "crlb": float(crlb_val),
            "best_std": float(best_std),
            "mle_theta": float(theta_mle),
            "mle_error": float(abs(theta_mle - theta_true)),
        },
    }
    st.download_button(
        "Download simulation results (JSON)",
        json.dumps(export_data, indent=2),
        "simulation_results.json",
        "application/json",
    )


# ═══════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════

section_divider()
st.markdown(
    "<div style='text-align:center; color:#999; font-size:0.78rem; padding:1rem 0 2rem;'>"
    "Phase 1 Pilot · Fisher Information for Boundary Localization · "
    "A proof-of-concept for estimation-theoretic segmentation uncertainty in medical imaging"
    "</div>",
    unsafe_allow_html=True,
)
