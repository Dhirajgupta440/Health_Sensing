import os
import argparse
import pandas as pd
import numpy as np
from scipy import signal
from datetime import datetime
import pickle
from tqdm import tqdm

# ----------------------------------------------------------------------
def parse_timestamp_ms(ts_str):
    """Convert '30.05.2024 20:59:00,031' → Unix timestamp"""
    try:
        dt_str, ms = ts_str.rsplit(',', 1)
        dt = datetime.strptime(dt_str, '%d.%m.%Y %H:%M:%S')
        return dt.timestamp() + float('0.' + ms.lstrip('0'))
    except:
        return None

# ----------------------------------------------------------------------
def load_signal(filepath, target_fs=32):
    """Load signal, parse timestamp, resample to target_fs Hz"""
    print(f"  Loading {os.path.basename(filepath)} → {target_fs} Hz")
    data = []
    in_data = False
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Data:'):
                in_data = True
                continue
            if not in_data:
                continue
            parts = line.split(';')
            if len(parts) != 2:
                continue
            ts_sec = parse_timestamp_ms(parts[0].strip())
            if ts_sec is None:
                continue
            try:
                val = float(parts[1].strip())
                data.append([ts_sec, val])
            except:
                continue
    if not data:
        raise ValueError(f"No data in {filepath}")
    df = pd.DataFrame(data, columns=['ts', 'val'])
    df = df.sort_values('ts').reset_index(drop=True)

    # Resample to target_fs
    t_start = df['ts'].min()
    t_end = df['ts'].max()
    t_reg = np.arange(t_start, t_end, 1/target_fs)
    val_interp = np.interp(t_reg, df['ts'], df['val'])
    return pd.DataFrame({'t': t_reg, 'val': val_interp})

# ----------------------------------------------------------------------
def load_events(filepath):
    print(f"  Loading events: {os.path.basename(filepath)}")
    data = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    if len(lines) > 0 and ('Start' in lines[0] or 'Event' in lines[0]):
        lines = lines[1:]
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split(',')
        if len(parts) >= 4:
            try:
                start = float(parts[1])
                stop = float(parts[2])
                typ = parts[3].strip()
                data.append([start, stop, typ])
                continue
            except:
                pass
        parts = line.split()
        if len(parts) >= 3:
            try:
                start = float(parts[0])
                stop = float(parts[1])
                typ = ' '.join(parts[2:])
                data.append([start, stop, typ])
            except:
                continue
    return pd.DataFrame(data, columns=['Start', 'Stop', 'Type'])

# ----------------------------------------------------------------------
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

# ----------------------------------------------------------------------
def create_windows(signal_df, window_sec=30, overlap=0.5, fs=32):
    step = int(window_sec * fs * (1 - overlap))
    window_samples = int(window_sec * fs)
    windows = []
    for start in range(0, len(signal_df) - window_samples + 1, step):
        end = start + window_samples
        window = signal_df.iloc[start:end]['val'].values
        t_center = signal_df.iloc[start + window_samples//2]['t']
        windows.append({'data': window, 't_center': t_center})
    return windows

# ----------------------------------------------------------------------
def label_window(t_center, events, threshold=0.5):
    for _, ev in events.iterrows():
        overlap = max(0, min(t_center + 15, ev['Stop']) - max(t_center - 15, ev['Start']))
        if overlap >= threshold * 30:
            if 'Hypopnea' in ev['Type']:
                return 'Hypopnea'
            elif 'Apnea' in ev['Type']:
                return 'Obstructive Apnea'
    return 'Normal'

# ----------------------------------------------------------------------
def main(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    participants = [f for f in os.listdir(in_dir) if f.startswith('AP')]
    dataset = []

    print(f"Processing {len(participants)} participants...")
    for pid in tqdm(participants, desc="Participants"):
        folder = os.path.join(in_dir, pid)
        files = {
            'flow': os.path.join(folder, "Flow - 30-05-2024.txt"),
            'thoracic': os.path.join(folder, "Thorac - 30-05-2024.txt"),
            'spo2': os.path.join(folder, "SPO2 - 30-05-2024.txt"),
            'events': os.path.join(folder, "Flow Events - 30-05-2024.txt")
        }

        try:
            # Resample ALL to 32 Hz
            flow = load_signal(files['flow'], target_fs=32)
            thoracic = load_signal(files['thoracic'], target_fs=32)
            spo2 = load_signal(files['spo2'], target_fs=32)  # ← NOW 32 Hz
            events = load_events(files['events'])
        except Exception as e:
            print(f"  [SKIP] {pid}: {e}")
            continue

        # Align time
        t0 = min(flow['t'].min(), thoracic['t'].min(), spo2['t'].min())
        flow['t'] -= t0
        thoracic['t'] -= t0
        spo2['t'] -= t0
        events['Start'] -= t0
        events['Stop'] -= t0

        # Filter breathing signals (0.17 - 0.4 Hz)
        flow['val_f'] = butter_bandpass_filter(flow['val'], 0.17, 0.4, 32)
        thoracic['val_f'] = butter_bandpass_filter(thoracic['val'], 0.17, 0.4, 32)

        # Create windows from flow (longest)
        windows = create_windows(flow, fs=32)
        for w in windows:
            t = w['t_center']
            # Extract 960-sample windows
            flow_win = flow[(flow['t'] >= t-15) & (flow['t'] < t+15)]['val_f'].values
            thor_win = thoracic[(thoracic['t'] >= t-15) & (thoracic['t'] < t+15)]['val_f'].values
            spo2_win = spo2[(spo2['t'] >= t-15) & (spo2['t'] < t+15)]['val'].values

            if len(flow_win) != 960 or len(thor_win) != 960 or len(spo2_win) != 960:
                continue

            X = np.stack([flow_win, thor_win, spo2_win], axis=0)  # (3, 960)
            y = label_window(t, events)

            dataset.append({
                'participant': pid,
                'X': X,
                'y': y,
                't_center': t
            })

    # Save
    out_path = os.path.join(out_dir, "breathing_dataset.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"\nSUCCESS → Dataset saved: {out_path}")
    print(f"Total windows: {len(dataset)}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-in_dir", "--in_dir", required=True, help="e.g. Data")
    parser.add_argument("-out_dir", "--out_dir", required=True, help="e.g. Dataset")
    args = parser.parse_args()
    main(args.in_dir, args.out_dir)