import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)

rng = np.random.default_rng(42)
n = 1000

rssi_normal  = rng.uniform(-80, -60, n)
snr_normal   = rng.uniform(5, 12, n)
loss_normal  = rng.uniform(0, 5, n)
label_normal = np.zeros(n)

rssi_jammed  = rng.uniform(-120, -90, n)
snr_jammed   = rng.uniform(-5, 4, n)
loss_jammed  = rng.uniform(10, 80, n)
label_jammed = np.ones(n)

df = pd.DataFrame({
    "rssi":      np.concatenate([rssi_normal, rssi_jammed]),
    "snr":       np.concatenate([snr_normal,  snr_jammed]),
    "loss":      np.concatenate([loss_normal, loss_jammed]),
    "label":     np.concatenate([label_normal, label_jammed])
})

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("data/signal_log.csv", index=False)
print(f"Synthetic dataset created: data/signal_log.csv ({len(df)} rows)")
