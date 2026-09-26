#!/usr/bin/env python
# =============================================================================
#  SIH 26052 — AI/ML Adaptive Noise Cancellation (Defence Communications)
#  Stand-alone Live Microphone Demo (No extra files required!)
# =============================================================================

import os
import sys
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import torch
import torch.nn as nn

# =============================================================================
# Audio Configuration
# =============================================================================
SR_MODEL = 16000     # Model processing sample rate
SR_PLAYBACK = 48000  # Standard Windows / Bluetooth headphone playback rate
DURATION = 5         # 5 seconds recording
N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512

# =============================================================================
# DCCRN Model Architecture
# =============================================================================
class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x_real, x_imag):
        out_real = self.conv_real(x_real) - self.conv_imag(x_imag)
        out_imag = self.conv_real(x_imag) + self.conv_imag(x_real)
        return out_real, out_imag

class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.deconv_real = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.deconv_imag = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x_real, x_imag):
        out_real = self.deconv_real(x_real) - self.deconv_imag(x_imag)
        out_imag = self.deconv_real(x_imag) + self.deconv_imag(x_real)
        return out_real, out_imag

class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn_real = nn.BatchNorm2d(num_features)
        self.bn_imag = nn.BatchNorm2d(num_features)

    def forward(self, x_real, x_imag):
        return self.bn_real(x_real), self.bn_imag(x_imag)

class ComplexPReLU(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.prelu_real = nn.PReLU(num_features)
        self.prelu_imag = nn.PReLU(num_features)

    def forward(self, x_real, x_imag):
        return self.prelu_real(x_real), self.prelu_imag(x_imag)

class DCCRN(nn.Module):
    def __init__(self, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer('window', torch.hann_window(win_length))

        encoder_channels = [16, 32, 64, 64, 128]
        kernel_size = (5, 2)
        stride = (2, 1)

        self.encoder_layers = nn.ModuleList()
        in_ch = 1
        padding = (kernel_size[0] // 2, kernel_size[1] - 1)

        for out_ch in encoder_channels:
            self.encoder_layers.append(nn.ModuleDict({
                'conv': ComplexConv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding),
                'bn': ComplexBatchNorm2d(out_ch),
                'act': ComplexPReLU(out_ch),
            }))
            in_ch = out_ch

        lstm_input_size = 128 * 9 * 2
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=128, num_layers=2, batch_first=True)
        self.fc_after_lstm = nn.Linear(128, lstm_input_size)

        self.decoder_layers = nn.ModuleList()
        decoder_channels = list(reversed(encoder_channels))
        for in_ch, out_ch in zip(decoder_channels[:-1], decoder_channels[1:]):
            self.decoder_layers.append(nn.ModuleDict({
                'deconv': ComplexConvTranspose2d(in_ch + out_ch, out_ch, kernel_size, stride=stride, padding=padding),
                'bn': ComplexBatchNorm2d(out_ch),
                'act': ComplexPReLU(out_ch),
            }))

        self.final_decoder = nn.ModuleDict({
            'deconv': ComplexConvTranspose2d(decoder_channels[-1] + 1, 1, kernel_size, stride=stride, padding=padding),
        })

        self.mask_magnitude = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Sigmoid())
        self.mask_phase_real = nn.Conv2d(1, 1, 1)
        self.mask_phase_imag = nn.Conv2d(1, 1, 1)

    def stft(self, x):
        spec = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                          win_length=self.win_length, window=self.window, return_complex=True)
        return spec.real.unsqueeze(1), spec.imag.unsqueeze(1)

    def istft(self, real, imag, length):
        spec = torch.complex(real.squeeze(1), imag.squeeze(1))
        return torch.istft(spec, n_fft=self.n_fft, hop_length=self.hop_length,
                           win_length=self.win_length, window=self.window, length=length)

    def forward(self, noisy_waveform):
        input_length = noisy_waveform.shape[-1]
        noisy_real, noisy_imag = self.stft(noisy_waveform)

        encoder_outs = []
        x_real, x_imag = noisy_real, noisy_imag

        for layer in self.encoder_layers:
            encoder_outs.append((x_real, x_imag))
            x_real, x_imag = layer['conv'](x_real, x_imag)
            x_real, x_imag = x_real[..., :-1], x_imag[..., :-1]
            x_real, x_imag = layer['bn'](x_real, x_imag)
            x_real, x_imag = layer['act'](x_real, x_imag)

        B, C, F_enc, T = x_real.shape
        x_lstm = torch.cat([
            x_real.permute(0, 3, 1, 2).reshape(B, T, -1),
            x_imag.permute(0, 3, 1, 2).reshape(B, T, -1),
        ], dim=-1)

        x_lstm, _ = self.lstm(x_lstm)
        x_lstm = self.fc_after_lstm(x_lstm)

        half = x_lstm.shape[-1] // 2
        x_real = x_lstm[..., :half].reshape(B, T, C, F_enc).permute(0, 2, 3, 1)
        x_imag = x_lstm[..., half:].reshape(B, T, C, F_enc).permute(0, 2, 3, 1)

        for i, layer in enumerate(self.decoder_layers):
            skip_real, skip_imag = encoder_outs[-(i + 1)]
            min_t = min(x_real.shape[-1], skip_real.shape[-1])
            min_f = min(x_real.shape[-2], skip_real.shape[-2])
            x_real = torch.cat([x_real[..., :min_f, :min_t], skip_real[..., :min_f, :min_t]], dim=1)
            x_imag = torch.cat([x_imag[..., :min_f, :min_t], skip_imag[..., :min_f, :min_t]], dim=1)
            x_real, x_imag = layer['deconv'](x_real, x_imag)
            x_real, x_imag = x_real[..., :-1], x_imag[..., :-1]
            x_real, x_imag = layer['bn'](x_real, x_imag)
            x_real, x_imag = layer['act'](x_real, x_imag)

        skip_real, skip_imag = encoder_outs[0]
        min_t = min(x_real.shape[-1], skip_real.shape[-1])
        min_f = min(x_real.shape[-2], skip_real.shape[-2])
        x_real = torch.cat([x_real[..., :min_f, :min_t], skip_real[..., :min_f, :min_t]], dim=1)
        x_imag = torch.cat([x_imag[..., :min_f, :min_t], skip_imag[..., :min_f, :min_t]], dim=1)
        x_real, x_imag = self.final_decoder['deconv'](x_real, x_imag)
        x_real, x_imag = x_real[..., :-1], x_imag[..., :-1]

        min_f = min(x_real.shape[-2], noisy_real.shape[-2])
        min_t = min(x_real.shape[-1], noisy_real.shape[-1])
        x_r = x_real[..., :min_f, :min_t]
        x_i = x_imag[..., :min_f, :min_t]
        nr = noisy_real[..., :min_f, :min_t]
        ni = noisy_imag[..., :min_f, :min_t]

        M = self.mask_magnitude(x_r)
        phase_real = torch.tanh(self.mask_phase_real(x_r))
        phase_imag = torch.tanh(self.mask_phase_imag(x_i))

        enh_real = M * (phase_real * nr - phase_imag * ni)
        enh_imag = M * (phase_imag * nr + phase_real * ni)

        enhanced = self.istft(enh_real, enh_imag, length=input_length)
        return enhanced

# =============================================================================
# Helper Utilities
# =============================================================================
def find_working_mic():
    """Detects active built-in microphone array, avoiding empty jacks."""
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0 and 'microphone array' in d['name'].lower():
            return i, d['name']
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0 and 'headset' in d['name'].lower():
            return i, d['name']
    def_in = sd.default.device[0]
    return def_in, devices[def_in]['name']

# =============================================================================
# Main Interactive Runner
# =============================================================================
def main():
    print("=" * 65)
    print("  SIH 26052 — DEFENCE SPEECH ENHANCEMENT DEMO")
    print("=" * 65)

    weights_file = "dccrn_weights.pt"
    if not os.path.exists(weights_file):
        print(f"\n[ERROR] '{weights_file}' not found in the current folder!")
        print("Please place 'dccrn_weights.pt' in the same folder as this script.")
        return

    mic_id, mic_name = find_working_mic()
    print(f"[MIC DETECTED] Using Device {mic_id}: {mic_name}")

    print("Loading DCCRN model weights...")
    device = torch.device("cpu")
    model = DCCRN().to(device)
    model.load_state_dict(torch.load(weights_file, map_location=device))
    model.eval()
    print("[OK] Model loaded successfully on CPU!\n")

    print("INSTRUCTIONS:")
    print("  1. Press ENTER to begin recording.")
    print("  2. Speak normally into your mic for 5 seconds.")
    print("  3. Play defence noise (helicopter, gunshots) on a phone nearby.")
    print("-" * 65)
    input(">>> Press ENTER to START recording 5 seconds... ")

    print("\n[RECORDING NOW...] Speak and play noise! (5 seconds)")
    raw_audio = sd.rec(int(DURATION * SR_MODEL), samplerate=SR_MODEL, channels=1, dtype='float32', device=mic_id)
    sd.wait()
    raw_audio = raw_audio.flatten()
    print("[RECORDING FINISHED!]")

    peak_raw = float(np.max(np.abs(raw_audio)))
    print(f"Captured audio peak level: {peak_raw:.4f}")

    if peak_raw < 0.001:
        print("\n[WARNING] Microphone recorded silence!")
        print("Check if your mic volume is up and not muted.")
        return

    # Save original noisy recording
    noisy_file = "1_original_noisy.wav"
    sf.write(noisy_file, raw_audio, SR_MODEL)
    print(f"[OK] Saved original noisy audio: {noisy_file}")

    # Scale to standard speech range for DCCRN
    model_input = (raw_audio / peak_raw) * 0.7

    # Process through DCCRN model
    print("\n[PROCESSING...] Eliminating defence noise through DCCRN...")
    t0 = time.time()
    with torch.no_grad():
        x = torch.from_numpy(model_input).unsqueeze(0).float()
        cleaned_audio = model(x).squeeze(0).numpy()
    print(f"[OK] Enhanced in {time.time() - t0:.2f} seconds!")

    # Normalize clean output naturally using 99.5th percentile (ignores edge clicks, brings voice to full volume)
    p99 = float(np.percentile(np.abs(cleaned_audio), 99.5))
    if p99 > 1e-4:
        cleaned_audio = np.clip(cleaned_audio / p99 * 0.95, -0.99, 0.99)

    # Save cleaned output
    clean_file = "2_dccrn_cleaned.wav"
    sf.write(clean_file, cleaned_audio, SR_MODEL)
    print(f"[OK] Saved boosted cleaned output: {clean_file}")

    # Resample to 48kHz for Bluetooth earbuds
    print("\nResampling for Bluetooth headphone playback...")
    clean_48k = librosa.resample(cleaned_audio, orig_sr=SR_MODEL, target_sr=SR_PLAYBACK)
    noisy_48k = librosa.resample(raw_audio, orig_sr=SR_MODEL, target_sr=SR_PLAYBACK)

    print("\n" + "=" * 65)
    print(">>> PLAYING ORIGINAL NOISY AUDIO in 1 second...")
    time.sleep(1)
    try:
        sd.play(noisy_48k, SR_PLAYBACK)
        sd.wait()
    except Exception as e:
        print("Playback note:", e)

    print(">>> NOW PLAYING DCCRN CLEANED AUDIO...")
    time.sleep(1)
    try:
        sd.play(clean_48k, SR_PLAYBACK)
        sd.wait()
    except Exception as e:
        print("Playback note:", e)

    print("=" * 65)
    print("ALL DONE!")
    print(f"You can also open these WAV files in Windows Media Player:")
    print(f"  - Original: {os.path.abspath(noisy_file)}")
    print(f"  - Cleaned:  {os.path.abspath(clean_file)}")

if __name__ == "__main__":
    main()
