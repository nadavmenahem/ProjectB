#!/usr/bin/env python3
"""
Enhanced noise-adder for outage dataset.
Supports Ornstein–Uhlenbeck (OU) colored noise and damped sinusoidal oscillations.
"""
import os
import numpy as np
import argparse
import shutil
import glob


def gen_ou_noise(n_steps, n_buses, dt, sigma, tau):
    """
    Generate Ornstein–Uhlenbeck noise.
    theta_{t+1} = theta_t * (1 - dt/tau) + sigma * sqrt(dt) * N(0,1)
    Returns shape (n_steps, n_buses).
    """
    theta = np.zeros(n_buses)
    noise = np.zeros((n_steps, n_buses))
    for t in range(n_steps):
        dtheta = -(theta / tau) * dt + sigma * np.sqrt(dt) * np.random.randn(n_buses)
        theta = theta + dtheta
        noise[t] = theta
    return noise


def gen_damped_sin_noise(n_steps, n_buses, dt, A, f0, damping):
    """
    Generate a damped sinusoidal oscillation:
    noise_t = A * exp(-damping * t) * sin(2*pi*f0*t)
    Broadcasts same oscillation to all buses.
    Returns shape (n_steps, n_buses).
    """
    t = np.arange(n_steps) * dt
    osc = A * np.exp(-damping * t) * np.sin(2 * np.pi * f0 * t)
    return np.tile(osc[:, None], (1, n_buses))


def process_dataset(input_dir, output_dir, model, **kwargs):
    # Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)
    input_cases = os.path.join(input_dir, "cases")
    output_cases = os.path.join(output_dir, "cases")
    os.makedirs(output_cases, exist_ok=True)

    # Gather all case files
    case_files = sorted(glob.glob(os.path.join(input_cases, "case_*.npz")))
    if not case_files:
        print(f"No case files found in {input_cases}")
        return

    # Load one sample to infer shape and key
    sample = np.load(case_files[0])
    # detect array key
    if 'data' in sample:
        key = 'data'
    elif 'angles' in sample:
        key = 'angles'
    else:
        key = list(sample.keys())[0]
    data0 = sample[key]
    n_steps, n_buses = data0.shape
    dt = 1.0 / kwargs.get('sampling_rate', 8.0)

    # Pre-generate noise matrix once per dataset
    if model == 'ou':
        sigma = kwargs.get('sigma', 0.5)
        tau   = kwargs.get('tau', 5.0)
        noise_matrix = gen_ou_noise(n_steps, n_buses, dt, sigma, tau)
    elif model == 'damped':
        A       = kwargs.get('A', 0.3)
        f0      = kwargs.get('f0', 0.8)
        damping = kwargs.get('damping', 0.1)
        noise_matrix = gen_damped_sin_noise(n_steps, n_buses, dt, A, f0, damping)
    else:
        raise ValueError(f"Unknown noise model: {model}")

    # Apply noise to each case
    for file in case_files:
        arr = np.load(file)
        data = arr[key]
        noisy = data + noise_matrix
        out_name = os.path.basename(file)
        out_path = os.path.join(output_cases, out_name)
        np.savez(out_path, **{key: noisy})
        print(f"Saved noisy case: {out_path}")

    # Copy over graph and metadata
    for extra in ['graph.npy', 'meta.json']:
        src = os.path.join(input_dir, extra)
        dst = os.path.join(output_dir, extra)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"Copied {extra}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add colored or oscillatory noise to outage dataset")
    parser.add_argument('input_dir',  help='Input dataset directory (base path)')
    parser.add_argument('output_dir', help='Output directory for noisy dataset')
    parser.add_argument('--model', choices=['ou','damped'], default='ou', help='Noise model to use')
    parser.add_argument('--sigma',     type=float, default=0.5, help='OU noise sigma (std dev)')
    parser.add_argument('--tau',       type=float, default=5.0, help='OU correlation time constant (s)')
    parser.add_argument('--A',         type=float, default=0.3, help='Damped sinusoid amplitude (deg)')
    parser.add_argument('--f0',        type=float, default=0.8, help='Damped sinusoid frequency (Hz)')
    parser.add_argument('--damping',   type=float, default=0.1, help='Damped sinusoid decay rate')
    parser.add_argument('--sampling_rate', type=float, default=8.0, help='Samples per second in dataset')
    args = parser.parse_args()

    process_dataset(
        args.input_dir,
        args.output_dir,
        args.model,
        sigma=args.sigma,
        tau=args.tau,
        A=args.A,
        f0=args.f0,
        damping=args.damping,
        sampling_rate=args.sampling_rate
    )
