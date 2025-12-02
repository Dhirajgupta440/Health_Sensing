import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

# ----------------------------------------------------------------------
def parse_timestamp_ms(ts_str):
    """Convert '30.05.2024 20:59:00,031' → Unix timestamp in seconds"""
    try:
        dt_str, ms = ts_str.rsplit(',', 1)
        dt = datetime.strptime(dt_str, '%d.%m.%Y %H:%M:%S')
        return dt.timestamp() + float('0.' + ms.lstrip('0'))
    except:
        return None

# ----------------------------------------------------------------------
def load_signal_file(filepath):
    """Load Flow, Thoracic, SPO2 – handles headers, 'Data:', ; delimiter"""
    print(f"[LOAD] {os.path.basename(filepath)}")
    data = []
    in_data_section = False

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('Data:'):
                in_data_section = True
                continue
            if not in_data_section:
                continue
            parts = line.split(';')
            if len(parts) != 2:
                continue
            ts_str, val_str = parts
            ts_sec = parse_timestamp_ms(ts_str.strip())
            if ts_sec is None:
                continue
            try:
                val = float(val_str.strip())
                data.append([ts_sec, val])
            except:
                continue

    if not data:
        raise ValueError(f"No valid data in {filepath}")

    df = pd.DataFrame(data, columns=['ts', 'val'])
    print(f"  [OK] {len(df)} samples | First 3:")
    print(df.head(3).to_string(index=False))
    return df

# ----------------------------------------------------------------------
def load_events_file(filepath):
    """Load Flow Events – handles headers, commas, spaces, any delimiter"""
    print(f"[LOAD] Events: {os.path.basename(filepath)}")
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    print("First 10 lines of events file:")
    for i, line in enumerate(lines[:10]):
        print(f"  {i}: {line.strip()}")
    if len(lines) > 10:
        print(f"  ... and {len(lines) - 10} more lines")

    # Skip header if present
    if len(lines) > 0 and ('Start' in lines[0] or 'Event' in lines[0] or 'Type' in lines[0]):
        lines = lines[1:]

    data = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # Try comma first
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
        # Fallback: space or tab
        parts = line.split()
        if len(parts) >= 3:
            try:
                start = float(parts[0])
                stop = float(parts[1])
                typ = ' '.join(parts[2:])
                data.append([start, stop, typ])
            except:
                continue
    
    df = pd.DataFrame(data, columns=['Start', 'Stop', 'Type'])
    print(f"  [OK] {len(df)} events loaded")
    return df

# ----------------------------------------------------------------------
def plot_participant(folder, out_dir="Visualizations"):
    os.makedirs(out_dir, exist_ok=True)
    pid = os.path.basename(folder)

    # EXACT filenames from your data
    files = {
        'nasal':    os.path.join(folder, "Flow - 30-05-2024.txt"),
        'thoracic': os.path.join(folder, "Thorac - 30-05-2024.txt"),
        'spo2':     os.path.join(folder, "SPO2 - 30-05-2024.txt"),
        'events':   os.path.join(folder, "Flow Events - 30-05-2024.txt")
    }

    # Check missing files
    missing = [k for k, v in files.items() if not os.path.exists(v)]
    if missing:
        print(f"[SKIP] {pid} – Missing files: {missing}")
        return

    try:
        nasal    = load_signal_file(files['nasal'])
        thoracic = load_signal_file(files['thoracic'])
        spo2     = load_signal_file(files['spo2'])
        events   = load_events_file(files['events'])
    except Exception as e:
        print(f"[ERROR] {pid} → {e}")
        return

    # Align to t=0
    t0 = min(nasal['ts'].min(), thoracic['ts'].min(), spo2['ts'].min())
    nasal['t'] = nasal['ts'] - t0
    thoracic['t'] = thoracic['ts'] - t0
    spo2['t'] = spo2['ts'] - t0
    events['start_sec'] = events['Start'] - t0
    events['stop_sec']  = events['Stop']  - t0

    # Plot
    pdf_path = os.path.join(out_dir, f"{pid}_visualization.pdf")
    with PdfPages(pdf_path) as pdf:
        fig, axs = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

        axs[0].plot(nasal['t']/3600, nasal['val'], color='tab:blue', lw=0.7)
        axs[0].set_title(f'{pid} – Nasal Airflow')
        axs[0].set_ylabel('Amplitude')
        axs[0].grid(True)

        axs[1].plot(thoracic['t']/3600, thoracic['val'], color='tab:green', lw=0.7)
        axs[1].set_title('Thoracic Movement')
        axs[1].set_ylabel('Amplitude')
        axs[1].grid(True)

        axs[2].plot(spo2['t']/3600, spo2['val'], color='tab:red', lw=1)
        axs[2].set_title('SpO₂')
        axs[2].set_ylabel('SpO₂ (%)')
        axs[2].set_xlabel('Time (hours)')
        axs[2].grid(True)

        # Shade events
        for _, ev in events.iterrows():
            color = 'orange' if 'Apnea' in str(ev['Type']) else 'purple'
            for ax in axs:
                ax.axvspan(ev['start_sec']/3600, ev['stop_sec']/3600, color=color, alpha=0.3)

        # Legend
        handles, labels = axs[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            axs[0].legend(by_label.values(), by_label.keys(), loc='upper right')

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"SUCCESS → {pdf_path}\n")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", "--folder", required=True, help="e.g. Data/AP01")
    args = parser.parse_args()
    plot_participant(args.folder)