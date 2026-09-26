#!/usr/bin/env python
# =============================================================================
#  SIH 26052 — Hybrid Adaptive Filter (NLMS) + Dual-Mic Simulation
#  Paste this into a SEPARATE Kaggle notebook cell (or new notebook)
#  AFTER the DCCRN model is trained and loaded.
# =============================================================================

# %%  ========================================================================
# CELL A1: Dual-Microphone Simulation
# =========================================================================
"""
In real deployment, a dual-mic setup has:
  - Primary mic: near the speaker's mouth → captures speech + leaked ambient noise
  - Reference mic: away from speaker, facing outward → captures ambient noise + faint speech

We simulate this by modeling the acoustic transfer function between the noise
source and primary mic as a simple FIR filter (room impulse response).
"""

import numpy as np
from scipy.signal import fftconvolve, firwin
import random


def simulate_dual_mic(
    clean_speech: np.ndarray,
    noise: np.ndarray,
    snr_db: float = 0.0,
    speech_leak_db: float = -20.0,
    transfer_filter_len: int = 64,
    sr: int = 16000,
) -> dict:
    """
    Simulate a dual-microphone recording scenario.

    Args:
        clean_speech: Clean speech signal (what the speaker says)
        noise: Environmental noise (gunshot, helicopter, etc.)
        snr_db: Desired SNR at the primary microphone
        speech_leak_db: How much speech leaks into the reference mic (dB, negative = attenuated)
        transfer_filter_len: Length of the FIR filter modeling acoustic path
        sr: Sample rate

    Returns:
        dict with keys:
            - primary: Primary mic signal (speech + filtered noise)
            - reference: Reference mic signal (noise + attenuated speech)
            - clean: Original clean speech (ground truth)
            - noise_at_primary: The noise as it arrives at the primary mic
    """
    # Ensure same length
    min_len = min(len(clean_speech), len(noise))
    clean_speech = clean_speech[:min_len]
    noise = noise[:min_len]

    # 1. Create acoustic transfer function (noise source → primary mic)
    # This is a random FIR filter simulating room reflections / distance effects
    np.random.seed(None)  # different each call
    h = np.random.randn(transfer_filter_len) * 0.1
    h[0] = 1.0  # direct path has strongest coefficient
    # Apply some decay
    decay = np.exp(-np.arange(transfer_filter_len) / (transfer_filter_len / 3))
    h *= decay
    h /= np.sqrt(np.sum(h ** 2))  # normalize energy

    # 2. Filter noise through transfer function (noise as heard at primary mic)
    noise_at_primary = fftconvolve(noise, h, mode='same')

    # 3. Scale noise for desired SNR at primary mic
    clean_power = np.mean(clean_speech ** 2) + 1e-12
    noise_power = np.mean(noise_at_primary ** 2) + 1e-12
    target_noise_power = clean_power / (10 ** (snr_db / 10))
    noise_scale = np.sqrt(target_noise_power / noise_power)
    noise_at_primary *= noise_scale

    # 4. Primary mic = speech + filtered/scaled noise
    primary = clean_speech + noise_at_primary

    # 5. Reference mic = raw noise + leaked speech (very attenuated)
    speech_leak_scale = 10 ** (speech_leak_db / 20)  # e.g., -20dB → 0.1x
    reference = noise * noise_scale + clean_speech * speech_leak_scale

    # Normalize to prevent clipping
    peak = max(np.max(np.abs(primary)), np.max(np.abs(reference)), 1e-6)
    if peak > 0.95:
        scale_down = 0.95 / peak
        primary *= scale_down
        reference *= scale_down
        clean_speech = clean_speech * scale_down
        noise_at_primary *= scale_down

    return {
        "primary": primary,
        "reference": reference,
        "clean": clean_speech,
        "noise_at_primary": noise_at_primary,
        "transfer_function": h,
    }


# %%  ========================================================================
# CELL A2: NLMS Adaptive Filter
# =========================================================================

class NLMSFilter:
    """
    Normalized Least Mean Squares (NLMS) adaptive filter.

    Used as a POST-FILTER after the neural network to suppress residual
    noise that is correlated with the reference mic signal.

    Unlike the Round 1 approach (which fed static repeated noise as reference),
    this operates on a properly correlated reference signal from the simulated
    (or real) secondary microphone.
    """

    def __init__(self, filter_length: int = 512, step_size: float = 0.1,
                 regularization: float = 1e-8):
        self.L = filter_length
        self.mu = step_size
        self.eps = regularization
        self.w = np.zeros(filter_length)  # filter coefficients

    def reset(self):
        """Reset filter coefficients."""
        self.w = np.zeros(self.L)

    def process(self, primary: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        Run NLMS adaptive noise cancellation.

        Args:
            primary: Signal from primary mic (or neural-enhanced signal)
            reference: Reference mic signal (correlated with residual noise)

        Returns:
            output: Enhanced signal with residual noise suppressed
        """
        n_samples = min(len(primary), len(reference))
        output = np.zeros(n_samples)

        # Pad reference for filter window
        ref_padded = np.concatenate([np.zeros(self.L - 1), reference[:n_samples]])

        for i in range(n_samples):
            # Reference signal vector (current + L-1 past samples)
            x = ref_padded[i:i + self.L][::-1]  # reversed for convolution

            # Filter output (estimate of noise in primary)
            noise_estimate = np.dot(self.w, x)

            # Error = primary - estimated noise (this is the cleaned signal)
            error = primary[i] - noise_estimate
            output[i] = error

            # Update filter coefficients (NLMS rule)
            norm = np.dot(x, x) + self.eps
            self.w += self.mu * error * x / norm

        return output

    def get_coefficients(self) -> np.ndarray:
        return self.w.copy()


# %%  ========================================================================
# CELL A3: Hybrid Pipeline — Neural + Adaptive
# =========================================================================

def hybrid_enhance(
    model,
    primary_signal: np.ndarray,
    reference_signal: np.ndarray,
    device: str = "cuda",
    nlms_filter_len: int = 512,
    nlms_step_size: float = 0.1,
) -> dict:
    """
    Full hybrid ANC pipeline:
    1. Neural network (DCCRN) processes primary signal
    2. NLMS adaptive filter uses reference to suppress residual noise

    Args:
        model: Trained DCCRN model
        primary_signal: Primary microphone signal
        reference_signal: Reference microphone signal
        device: 'cuda' or 'cpu'
        nlms_filter_len: NLMS filter length
        nlms_step_size: NLMS step size

    Returns:
        dict with neural-only output, hybrid output, and individual gains
    """
    import torch
    from torch.cuda.amp import autocast

    # Step 1: Neural enhancement
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(primary_signal.astype(np.float32)).unsqueeze(0)
        x = x.to(device)
        with autocast(enabled=True):
            neural_output = model(x)
        neural_output = neural_output.squeeze(0).cpu().numpy()

    # Step 2: NLMS post-filter on neural output using reference
    nlms = NLMSFilter(filter_length=nlms_filter_len, step_size=nlms_step_size)
    min_len = min(len(neural_output), len(reference_signal))
    hybrid_output = nlms.process(
        neural_output[:min_len],
        reference_signal[:min_len]
    )

    return {
        "neural_only": neural_output[:min_len],
        "hybrid": hybrid_output,
        "nlms_coefficients": nlms.get_coefficients(),
    }


# %%  ========================================================================
# CELL A4: Evaluate Hybrid vs Neural-Only
# =========================================================================

def evaluate_hybrid_pipeline(
    model,
    speech_files,
    noise_by_category,
    snr_levels=[-10, -5, 0, 5, 10, 15],
    num_per_snr=20,
    device="cuda",
    sr=16000,
    clip_samples=64000,
):
    """Run full comparative evaluation: Noisy → Neural-Only → Hybrid."""

    records = []

    for snr_db in snr_levels:
        for i in range(num_per_snr):
            # Load test pair
            sp_path = random.choice(speech_files)
            clean = load_audio(sp_path, sr=sr)
            clean = pad_or_crop(clean, clip_samples, random_crop=False)

            cat = random.choice(list(noise_by_category.keys()))
            ns_path = random.choice(noise_by_category[cat])
            noise = load_audio(ns_path, sr=sr)
            noise = pad_or_crop(noise, clip_samples)

            # Simulate dual-mic
            dual = simulate_dual_mic(clean, noise, snr_db=snr_db)

            # Hybrid enhancement
            result = hybrid_enhance(
                model, dual["primary"], dual["reference"], device=device
            )

            min_len = min(len(dual["clean"]), len(result["hybrid"]))
            clean_eval = dual["clean"][:min_len]
            primary_eval = dual["primary"][:min_len]
            neural_eval = result["neural_only"][:min_len]
            hybrid_eval = result["hybrid"][:min_len]

            # Metrics for each stage
            record = {
                "input_snr_db": snr_db,
                "noise_type": cat,
                # Noisy (baseline)
                "noisy_snr": compute_snr(clean_eval, primary_eval),
                "noisy_pesq": compute_pesq_score(clean_eval, primary_eval),
                "noisy_stoi": compute_stoi_score(clean_eval, primary_eval),
                # Neural-only
                "neural_snr": compute_snr(clean_eval, neural_eval),
                "neural_pesq": compute_pesq_score(clean_eval, neural_eval),
                "neural_stoi": compute_stoi_score(clean_eval, neural_eval),
                # Hybrid (neural + NLMS)
                "hybrid_snr": compute_snr(clean_eval, hybrid_eval),
                "hybrid_pesq": compute_pesq_score(clean_eval, hybrid_eval),
                "hybrid_stoi": compute_stoi_score(clean_eval, hybrid_eval),
            }
            records.append(record)

        print(f"SNR {snr_db:+3d} dB: done")

    df = pd.DataFrame(records)

    # Summary comparison
    summary = df.groupby("input_snr_db").agg({
        "noisy_snr": "mean", "noisy_pesq": "mean", "noisy_stoi": "mean",
        "neural_snr": "mean", "neural_pesq": "mean", "neural_stoi": "mean",
        "hybrid_snr": "mean", "hybrid_pesq": "mean", "hybrid_stoi": "mean",
    }).round(3)

    return df, summary


# Run the evaluation (uncomment when model is loaded):
# eval_df, eval_summary = evaluate_hybrid_pipeline(
#     model, speech_files, noise_by_category, device=str(device)
# )
# print("\n" + "=" * 100)
# print("COMPARATIVE RESULTS: Noisy → DCCRN → DCCRN+NLMS Hybrid")
# print("=" * 100)
# print(eval_summary.to_string())

print("Hybrid pipeline module loaded. Uncomment the evaluation cell to run.")
