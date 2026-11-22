import glob
import json
import os
import shutil
import time

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import serial
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from streamlit_autorefresh import st_autorefresh

# ─── CONFIGURATION ────────────────────────────────────────
# Paths are anchored to this script's own folder (not the current working
# directory), so the dataset and model are found at the same place every
# time, however and wherever `streamlit run app.py` is launched from.
APP_DIR       = os.path.dirname(os.path.abspath(__file__))
RECEIVER_PORT = "COM4"          # only the Receiver is connected to the PC
BAUD_RATE     = 115200
CSV_PATH      = os.path.join(APP_DIR, "data", "signal_log.csv")
MODEL_PATH    = os.path.join(APP_DIR, "ml", "model.pkl")
STATS_PATH    = os.path.join(APP_DIR, "ml", "trained_stats.json")
MODEL_BACKUP_DIR = os.path.join(APP_DIR, "ml", "backups")
MAX_BACKUPS       = 10           # keep the 10 most recent trained models
WINDOW        = 5               # packets used for variance / delta features
REFRESH_MS    = 1000
BASE_FREQ     = 433_000_000
HOP_FREQ      = 433_750_000     # assumed-clean channel
BASE_SF       = 8               # normal operating Spreading Factor
HOP_SF        = 12              # more robust (slower) SF used once jamming is detected
HOP_CONFIRM   = 2               # consecutive ML "jammed" packets before hopping
MIN_PER_CLASS = 20              # samples per class needed to train
# ──────────────────────────────────────────────────────────

FEATS      = ["rssi", "snr", "loss", "rssi_var", "rssi_delta", "snr_delta"]
TRAIN_COLS = ["timestamp", "pkt"] + FEATS + ["label"]
LIVE_COLS  = ["timestamp", "pkt"] + FEATS + ["ml_pred", "ml_prob", "freq", "sf", "phase"]

os.makedirs(os.path.join(APP_DIR, "data"), exist_ok=True)
os.makedirs(os.path.join(APP_DIR, "ml"), exist_ok=True)
os.makedirs(MODEL_BACKUP_DIR, exist_ok=True)

st.set_page_config(page_title="LoRa AI Jamming Detection & Adaptive Control", layout="wide")
st_autorefresh(interval=REFRESH_MS, key="autorefresh")


# ─── Session state ────────────────────────────────────────
def load_csv():
    if os.path.exists(CSV_PATH):
        try:
            d = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
            return d[TRAIN_COLS]
        except Exception:
            pass
    return pd.DataFrame(columns=TRAIN_COLS)


def load_trained_stats():
    """Load trained_stats from disk so it survives a full browser refresh."""
    if os.path.exists(STATS_PATH):
        try:
            with open(STATS_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def init_state():
    if "data" not in st.session_state:
        st.session_state["data"] = load_csv()
    # Load model from disk on a fresh browser session
    if "model" not in st.session_state:
        m = None
        if os.path.exists(MODEL_PATH):
            try:
                m = joblib.load(MODEL_PATH)
            except Exception:
                m = None
        st.session_state["model"] = m
    # Auto-fix: session says None but disk already has a saved model (e.g. after rerun edge case)
    elif st.session_state["model"] is None and os.path.exists(MODEL_PATH):
        try:
            st.session_state["model"] = joblib.load(MODEL_PATH)
        except Exception:
            pass
    defaults = {
        "live_data": pd.DataFrame(columns=LIVE_COLS),
        "running": False,        # Tab 1 collecting
        "live_running": False,   # Tab 2 detecting
        "ser_rx": None,
        "jam_active": False,
        "ml_pred": 0,
        "ml_prob": 0.0,
        "jam_streak": 0,
        "freq_hopped": False,
        "live_freq": BASE_FREQ,
        "live_sf": BASE_SF,
        "tx_freq": None,
        "tx_sf": None,
        "cur_pkt": {},
        "raw_terminal": [],
        # trained_stats / tab1_done come from disk, not a session-only flag, so they
        # survive a full browser refresh (new session) as long as the model file exists.
        "trained_stats": load_trained_stats(),
        "tab1_done": os.path.exists(MODEL_PATH),
        "last_pkt_time": None,
        "events": {"jam_on": None, "hop": None, "recovered": None},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ─── Serial helpers (Receiver only) ───────────────────────
def open_rx():
    ser = st.session_state["ser_rx"]
    if ser is not None and ser.is_open:
        return None
    try:
        s = serial.Serial()
        s.port = RECEIVER_PORT
        s.baudrate = BAUD_RATE
        s.timeout = 0.1
        s.dtr = False            # stops the ESP32 from auto-resetting on open
        s.rts = False
        s.open()
        st.session_state["ser_rx"] = s
        return None
    except Exception as e:
        return f"{RECEIVER_PORT}: {e}"


def close_rx():
    try:
        if st.session_state["ser_rx"]:
            st.session_state["ser_rx"].close()
    except Exception:
        pass
    st.session_state["ser_rx"] = None


def flush_rx():
    try:
        st.session_state["ser_rx"].reset_input_buffer()
    except Exception:
        pass
    st.session_state["cur_pkt"] = {}


def send_to_rx(cmd):
    """Receiver forwards JAM/FREQ/RESET to the Transmitter over ESP-NOW."""
    try:
        ser = st.session_state["ser_rx"]
        if ser and ser.is_open:
            ser.write((cmd + "\n").encode())
    except Exception:
        pass


def reset_link():
    send_to_rx("RESET")
    st.session_state.update(freq_hopped=False, jam_streak=0, ml_pred=0, ml_prob=0.0,
                            live_freq=BASE_FREQ, live_sf=BASE_SF)


def set_jam(on):
    prev = st.session_state["jam_active"]
    st.session_state["jam_active"] = on
    if st.session_state["live_running"] and on != prev:
        if on:
            st.session_state["events"] = {"jam_on": time.time(), "hop": None, "recovered": None}
        elif st.session_state["freq_hopped"]:
            reset_link()          # jammer gone -> go back to baseline channel


# ─── Feature helper ───────────────────────────────────────
def window_feats(df, rssi, snr):
    if len(df) >= WINDOW:
        rr = df["rssi"].tail(WINDOW).astype(float).values
        rs = df["snr"].tail(WINDOW).astype(float).values
        return float(np.var(np.append(rr, rssi))), float(rssi - rr[0]), float(snr - rs[0])
    return 0.0, 0.0, 0.0


# ─── TAB 1: training packet (label from ground truth, NO hopping) ───
def commit_training_packet(p):
    rssi, snr, loss = float(p["rssi"]), float(p["snr"]), float(p.get("loss", 0.0))
    var, d_rssi, d_snr = window_feats(st.session_state["data"], rssi, snr)
    label = int(p["gt"]) if "gt" in p else int(st.session_state["jam_active"])
    row = pd.DataFrame([{
        "timestamp": pd.Timestamp.now(), "pkt": p.get("pkt", 0),
        "rssi": rssi, "snr": snr, "loss": loss,
        "rssi_var": var, "rssi_delta": d_rssi, "snr_delta": d_snr, "label": label,
    }])
    st.session_state["data"] = pd.concat([st.session_state["data"], row], ignore_index=True)
    st.session_state["data"].to_csv(CSV_PATH, index=False)


# ─── TAB 2: live packet (ML decides, hop on detection) ───
def commit_live_packet(p):
    rssi, snr, loss = float(p["rssi"]), float(p["snr"]), float(p.get("loss", 0.0))
    ld = st.session_state["live_data"]
    var, d_rssi, d_snr = window_feats(ld, rssi, snr)

    pred, prob = 0, 0.0
    model = st.session_state["model"]
    if model is not None:
        X = np.array([[rssi, snr, loss, var, d_rssi, d_snr]])
        pred = int(model.predict(X)[0])
        classes = list(model.classes_)
        prob = float(model.predict_proba(X)[0][classes.index(1)]) if 1 in classes else 0.0
    st.session_state["ml_pred"], st.session_state["ml_prob"] = pred, prob

    # confirm jamming over consecutive packets, then hop
    st.session_state["jam_streak"] = st.session_state["jam_streak"] + 1 if pred == 1 else 0
    ev = st.session_state["events"]
    if st.session_state["jam_streak"] >= HOP_CONFIRM and not st.session_state["freq_hopped"]:
        send_to_rx(f"FREQ:{HOP_FREQ}")          # Receiver hops + tells Transmitter via ESP-NOW
        send_to_rx(f"SF:{HOP_SF}")              # also move to a more robust Spreading Factor
        st.session_state["freq_hopped"] = True
        ev["hop"] = time.time()

    # recovery = first clean, zero-loss packet that actually arrived on the hop channel
    if (st.session_state["freq_hopped"] and ev["hop"] and ev["recovered"] is None
            and p.get("freq") == HOP_FREQ and loss == 0.0 and pred == 0):
        ev["recovered"] = time.time()

    phase = ("hopped" if st.session_state["freq_hopped"]
             else "jammed" if st.session_state["jam_active"] else "clean")
    row = pd.DataFrame([{
        "timestamp": pd.Timestamp.now(), "pkt": p.get("pkt", 0),
        "rssi": rssi, "snr": snr, "loss": loss,
        "rssi_var": var, "rssi_delta": d_rssi, "snr_delta": d_snr,
        "ml_pred": pred, "ml_prob": prob, "freq": p.get("freq", BASE_FREQ),
        "sf": p.get("sf", BASE_SF), "phase": phase,
    }])
    st.session_state["live_data"] = pd.concat([ld, row], ignore_index=True)


# ─── Serial reader (one parser, two modes) ────────────────
def read_rx(mode):
    ser = st.session_state["ser_rx"]
    if not ser or not ser.is_open:
        return
    try:
        while ser.in_waiting > 0:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue
            if line == "JAM:1":
                set_jam(True)
                continue
            if line == "JAM:0":
                set_jam(False)
                continue

            term = st.session_state["raw_terminal"]
            term.append(line)
            del term[:-15]

            cur = st.session_state["cur_pkt"]
            try:
                if line == "---":
                    st.session_state["cur_pkt"] = {}
                elif line.startswith("FREQACK:"):
                    st.session_state["live_freq"] = int(line[8:])
                elif line.startswith("SFACK:"):
                    st.session_state["live_sf"] = int(line[6:])
                elif line.startswith("PKT:"):
                    cur["pkt"] = int(line[4:])
                elif line.startswith("RSSI:"):
                    cur["rssi"] = float(line[5:].replace("dBm", "").strip())
                elif line.startswith("SNR:"):
                    cur["snr"] = float(line[4:].replace("dB", "").strip())
                elif line.startswith("LOSS:"):
                    cur["loss"] = float(line[5:].replace("%", "").strip())
                elif line.startswith("GT:"):
                    cur["gt"] = int(line[3:])
                elif line.startswith("TXFREQ:"):
                    st.session_state["tx_freq"] = int(line[7:])
                elif line.startswith("TXSF:"):
                    st.session_state["tx_sf"] = int(line[5:])
                elif line.startswith("SF:"):
                    sf_val = int(line[3:])
                    cur["sf"] = sf_val
                    st.session_state["live_sf"] = sf_val   # update every packet, not just on SFACK
                elif line.startswith("FREQ:"):         # last line of a packet block
                    f = int(line[5:])
                    st.session_state["live_freq"] = f
                    cur["freq"] = f
                    if "rssi" in cur and "snr" in cur:
                        (commit_training_packet if mode == "train" else commit_live_packet)(cur)
                    st.session_state["cur_pkt"] = {}
                    st.session_state["last_pkt_time"] = pd.Timestamp.now()
            except ValueError:
                pass
    except Exception:
        pass


if st.session_state["running"]:
    read_rx("train")
elif st.session_state["live_running"]:
    read_rx("live")


# ─── Training ─────────────────────────────────────────────
def train_model():
    d = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else st.session_state["data"]
    X, y = d[FEATS].values, d["label"].astype(int).values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    clf.fit(Xtr, ytr)
    pte = clf.predict(Xte)
    test_acc = accuracy_score(yte, pte)
    report = classification_report(yte, pte, target_names=["Normal", "Jammed"], zero_division=0)
    clf.fit(X, y)                                   # final model uses all data
    joblib.dump(clf, MODEL_PATH)

    # keep a timestamped backup so a previous trained model is never silently lost
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy(MODEL_PATH, f"{MODEL_BACKUP_DIR}/model_{stamp}.pkl")
    backups = sorted(glob.glob(os.path.join(MODEL_BACKUP_DIR, "model_*.pkl")))
    for old in backups[:-MAX_BACKUPS]:               # trim to the newest MAX_BACKUPS
        os.remove(old)
    n, j = d[d["label"] == 0], d[d["label"] == 1]
    st.session_state["model"] = clf
    st.session_state["tab1_done"] = True
    st.session_state["trained_stats"] = {
        "acc": test_acc, "report": report, "n": len(n), "j": len(j),
        "norm_rssi": f"{n['rssi'].min():.0f} to {n['rssi'].max():.0f} dBm",
        "jam_rssi": f"{j['rssi'].min():.0f} to {j['rssi'].max():.0f} dBm",
        "norm_loss": f"{n['loss'].min():.1f}% to {n['loss'].max():.1f}%",
        "jam_loss": f"{j['loss'].min():.1f}% to {j['loss'].max():.1f}%",
        "norm_var": f"{n['rssi_var'].mean():.2f}", "jam_var": f"{j['rssi_var'].mean():.2f}",
        "importances": dict(zip(FEATS, clf.feature_importances_)),
    }
    # persist stats to disk so they survive a full browser refresh (new Streamlit session)
    with open(STATS_PATH, "w") as f:
        json.dump(st.session_state["trained_stats"], f)


# ═══════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════
st.title("📡 LoRa AI Jamming Detection & Adaptive Link Control")
st.caption("Train on labelled data (no hopping) ➔ live ML detection ➔ ESP-NOW coordinated hop to 433.75 MHz ➔ measure recovery")

df = st.session_state["data"]
ld = st.session_state["live_data"]
model_trained = st.session_state["model"] is not None
rx = st.session_state["ser_rx"]
rx_ok = bool(rx and rx.is_open)
cur_mhz = st.session_state["live_freq"] / 1e6

# status bar
s1, s2, s3, s4 = st.columns(4)
s1.info(f"RX {RECEIVER_PORT}: {'🟢 Connected' if rx_ok else '⚪ Not connected'}")
s2.info(f"Jammer: {'🔴 ON' if st.session_state['jam_active'] else '⚪ OFF'}  (BOOT on Receiver or Transmitter toggles it)")
age = (pd.Timestamp.now() - st.session_state["last_pkt_time"]).seconds if st.session_state["last_pkt_time"] is not None else None
s3.info(f"Last packet: {'--' if age is None else f'{age}s ago'}  |  Link: {cur_mhz:.2f} MHz")
if model_trained:
    s4.success("🧠 Model loaded ✔")
else:
    s4.error("🧠 No model — train in Tab 1")

tab_feed, tab_live, tab_diff, tab_data = st.tabs([
    "📥 Tab 1: Data Feeding & Training",
    "🛡️ Tab 2: Live AI Detection ✅" if model_trained else "🔒 Tab 2: Live AI Detection (train first)",
    "🔬 Tab 3: Attenuation vs Jamming",
    "📋 Tab 4: Dataset & Reset",
])


# ───────────── TAB 1 ─────────────
def render_feed():
    st.subheader("Step 1: Collect labelled Normal and Jammed data (no hopping here)")
    st.markdown(
        "- **Jammer OFF → label 0** (normal link)\n"
        "- **Jammer ON → label 1** (press BOOT on the Receiver, or use the button below)\n"
        "- The label comes from the Receiver's own jam flag at the moment each packet arrives.\n"
        "- The frequency stays on 433.00 MHz the whole time."
    )
    if st.session_state["tab1_done"]:
        st.success("✅ Model trained and saved. Go to Tab 2 for live detection.")

    with st.expander("💾 Dataset & model persistence (always on)"):
        n_rows = len(df)
        on_disk = os.path.exists(CSV_PATH)
        st.write(
            f"- Dataset file: `{CSV_PATH}`\n"
            f"- Model file: `{MODEL_PATH}`\n"
            f"- **{n_rows} rows on disk right now.** Both paths are locked to this script's own folder, "
            "so they're found at the same place every time, no matter which folder you launch "
            "`streamlit run app.py` from.\n"
            f"- Each time you click **Train ML Model**, the previous model is also copied into "
            f"`{MODEL_BACKUP_DIR}` with a timestamp, so retraining never erases an earlier working model "
            f"(the {MAX_BACKUPS} most recent are kept)."
        )
        if not on_disk and n_rows == 0:
            st.caption("No dataset file yet — collect some packets in Tab 1 to create it.")

        backups = sorted(glob.glob(os.path.join(MODEL_BACKUP_DIR, "model_*.pkl")), reverse=True)
        if backups:
            st.write("**Restore an earlier trained model:**")
            pick = st.selectbox("Backup", [os.path.basename(b) for b in backups], key="restore_pick")
            if st.button("♻ Restore this backup as the active model", key="restore_btn"):
                shutil.copy(os.path.join(MODEL_BACKUP_DIR, pick), MODEL_PATH)
                st.session_state["model"] = joblib.load(MODEL_PATH)
                st.session_state["tab1_done"] = True
                st.success(f"Restored {pick} as the active model.")
                st.rerun()

        st.write("**Import a dataset you saved earlier** (e.g. from another PC, or a downloaded CSV):")
        up = st.file_uploader("Upload a signal_log.csv to merge into the current dataset",
                              type="csv", key="import_csv")
        if up is not None:
            try:
                new_d = pd.read_csv(up, parse_dates=["timestamp"])[TRAIN_COLS]
                merged = pd.concat([st.session_state["data"], new_d], ignore_index=True)
                merged = merged.drop_duplicates(subset=["timestamp", "pkt", "rssi", "snr"])
                st.session_state["data"] = merged
                merged.to_csv(CSV_PATH, index=False)
                st.success(f"Merged {len(new_d)} rows — dataset now has {len(merged)} rows total.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not import that file: {e}")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        if not st.session_state["running"]:
            if st.button("▶ Start Collecting (COM4)", use_container_width=True):
                err = open_rx()
                if err:
                    st.error(f"Cannot open {err}")
                else:
                    flush_rx()
                    send_to_rx("RESET")                      # known baseline state
                    st.session_state.update(running=True, live_running=False, tab1_done=False,
                                            jam_active=False, freq_hopped=False)
                    st.rerun()
        else:
            if st.button("⏹ Stop Collecting", use_container_width=True):
                st.session_state["running"] = False
                send_to_rx("JAM:0")
                st.session_state["jam_active"] = False
                st.rerun()

    with c2:
        lbl = "🔴 Jammer ON (label 1) – click to stop" if st.session_state["jam_active"] else "⚡ Turn Jammer ON (label 1)"
        if st.button(lbl, use_container_width=True, disabled=not rx_ok):
            new = not st.session_state["jam_active"]
            st.session_state["jam_active"] = new
            send_to_rx("JAM:1" if new else "JAM:0")
            st.rerun()

    with c3:
        n0 = int((df["label"] == 0).sum()) if not df.empty else 0
        n1 = int((df["label"] == 1).sum()) if not df.empty else 0
        ready = n0 >= MIN_PER_CLASS and n1 >= MIN_PER_CLASS
        if st.button("🧠 Train ML Model" if ready else f"🧠 Train (need {MIN_PER_CLASS}+ each)",
                     use_container_width=True, disabled=not ready):
            st.session_state["running"] = False
            send_to_rx("JAM:0")
            st.session_state["jam_active"] = False
            train_model()
            st.rerun()

    with c4:
        if st.button("💾 Save Dataset", use_container_width=True, disabled=df.empty):
            st.session_state["data"].to_csv(CSV_PATH, index=False)
            st.success(f"Saved {len(st.session_state['data'])} rows to `{CSV_PATH}`. "
                       "This is what Train ML Model will use, now and next time you open the app.")

    with c5:
        confirm = st.checkbox("Confirm", key="reset_confirm",
                              help="Tick this, then press Reset. Deletes the dataset, the trained "
                                   "model and all backups — cannot be undone.")
        if st.button("🗑️ Reset Everything", use_container_width=True, disabled=not confirm):
            st.session_state["data"] = pd.DataFrame(columns=TRAIN_COLS)
            st.session_state["model"] = None
            st.session_state["trained_stats"] = None
            st.session_state["tab1_done"] = False
            if os.path.exists(CSV_PATH):
                os.remove(CSV_PATH)
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            if os.path.exists(STATS_PATH):
                os.remove(STATS_PATH)
            for b in glob.glob(os.path.join(MODEL_BACKUP_DIR, "model_*.pkl")):
                os.remove(b)
            st.session_state["reset_confirm"] = False
            st.success("Dataset, model and all backups deleted.")
            st.rerun()

    st.divider()
    if st.session_state["jam_active"]:
        st.error("🔴 Feeding state: JAMMER ON ➔ packets labelled 1")
    else:
        st.success("🟢 Feeding state: JAMMER OFF ➔ packets labelled 0")

    if not df.empty:
        k1, k2, k3, k4 = st.columns(4)
        lat = df.iloc[-1]
        k1.metric("🟢 Normal (0)", n0)
        k2.metric("🔴 Jammed (1)", n1)
        k3.metric("RSSI / SNR", f"{lat['rssi']:.0f} dBm / {lat['snr']:.1f} dB")
        k4.metric("Ready to train?", "✅ Yes" if ready else "❌ Not yet")

    stt = st.session_state["trained_stats"]
    if stt:
        with st.expander("🎓 What the model learned", expanded=True):
            st.write(f"**Hold-out test accuracy:** `{stt['acc']*100:.1f}%` "
                     f"({stt['n']} normal / {stt['j']} jammed samples). Adjacent packets are correlated, "
                     "so treat this as optimistic.")
            st.code(stt["report"])
            b1, b2, b3 = st.columns(3)
            b1.info(f"**RSSI**\nNormal: `{stt['norm_rssi']}`\nJammed: `{stt['jam_rssi']}`")
            b2.info(f"**Loss**\nNormal: `{stt['norm_loss']}`\nJammed: `{stt['jam_loss']}`")
            b3.info(f"**RSSI variance (mean)**\nNormal: `{stt['norm_var']}`\nJammed: `{stt['jam_var']}`")
            imp = pd.DataFrame(list(stt["importances"].items()), columns=["Feature", "Importance"])
            st.plotly_chart(px.bar(imp, x="Feature", y="Importance"), use_container_width=True, key="imp")

    if not df.empty:
        a, b = st.columns(2)
        a.plotly_chart(px.line(df.tail(200), x="timestamp", y="rssi", color="label",
                               title="RSSI (dBm) by label"), use_container_width=True, key="t1rssi")
        b.plotly_chart(px.line(df.tail(200), x="timestamp", y="loss", title="Packet loss (%)"),
                       use_container_width=True, key="t1loss")
    with st.expander("📺 Raw terminal from Receiver"):
        st.code("\n".join(st.session_state["raw_terminal"]) or "Waiting for data...")


# ───────────── TAB 2 ─────────────
def render_live():
    if not model_trained:
        st.warning("🔒 Train the model in Tab 1 first.")
        return

    st.subheader("Step 2: Live ML detection + autonomous frequency hop")
    st.caption("No labels are used here. The model classifies each packet; after "
               f"{HOP_CONFIRM} consecutive 'jammed' predictions the Receiver hops to "
               f"{HOP_FREQ/1e6:.2f} MHz **and** switches to SF{HOP_SF} (more robust, slower), "
               "telling the Transmitter over ESP-NOW.")

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        if not st.session_state["live_running"]:
            if st.button("▶ Start Live Detection", type="primary", use_container_width=True):
                err = open_rx()
                if err:
                    st.error(f"Cannot open {err}")
                else:
                    flush_rx()
                    st.session_state["live_data"] = pd.DataFrame(columns=LIVE_COLS)
                    st.session_state["events"] = {"jam_on": None, "hop": None, "recovered": None}
                    st.session_state.update(running=False, live_running=True, jam_active=False)
                    reset_link()
                    st.rerun()
        else:
            if st.button("⏹ Stop Live Detection", use_container_width=True):
                st.session_state["live_running"] = False
                st.session_state["jam_active"] = False
                reset_link()
                st.rerun()
    with c2:
        if st.session_state["live_running"]:
            lbl = "⏹ Stop Jammer" if st.session_state["jam_active"] else "⚡ Start Jammer (simulated)"
            if st.button(lbl, use_container_width=True):
                new = not st.session_state["jam_active"]
                send_to_rx("JAM:1" if new else "JAM:0")
                set_jam(new)
                st.rerun()
    with c3:
        if st.session_state["live_running"] and st.session_state["live_freq"] != BASE_FREQ:
            if st.button("🔄 Manual reset to 433.00 MHz", use_container_width=True):
                reset_link()
                st.rerun()

    if not st.session_state["live_running"]:
        st.info("Press ▶ Start Live Detection.")
        return

    pred, prob = st.session_state["ml_pred"], st.session_state["ml_prob"]
    hopped = st.session_state["freq_hopped"]
    st.divider()
    if hopped and st.session_state["jam_active"]:
        st.success(f"🛡️ ML detected jamming → link hopped to **{cur_mhz:.2f} MHz**. "
                   "The (emulated) jammer stays on 433.00 MHz.")
    elif hopped:
        st.warning("Hopped — waiting for the jammer to stop or for a manual reset.")
    elif pred == 1:
        st.error(f"🚨 ML says JAMMED (p={prob*100:.0f}%) — confirming before hop "
                 f"({st.session_state['jam_streak']}/{HOP_CONFIRM})")
    else:
        st.info(f"🟢 ML says NORMAL (p(jam)={prob*100:.0f}%) on {cur_mhz:.2f} MHz")

    st.write("#### Channel indicator")
    chans = [("433.00", 433_000_000, "CH1 baseline"), ("433.25", 433_250_000, "CH2"),
             ("433.50", 433_500_000, "CH3"), ("433.75", 433_750_000, "CH4 anti-jam"),
             ("434.00", 434_000_000, "CH5")]
    for col, (name, f, desc) in zip(st.columns(5), chans):
        with col:
            if f == st.session_state["live_freq"]:
                (st.success if f == BASE_FREQ else st.error)(f"**{name} MHz**\nACTIVE – {desc}")
            else:
                st.caption(f"**{name} MHz**\n{desc}")

    tx_f = st.session_state.get("tx_freq")
    rx_f = st.session_state["live_freq"]
    if tx_f:
        if tx_f == rx_f:
            st.success(f"✔ Hardware check: the Transmitter reports {tx_f/1e6:.2f} MHz inside the packet "
                       f"it just sent, and the Receiver is tuned to {rx_f/1e6:.2f} MHz.")
        else:
            st.warning(f"Transmitter reports {tx_f/1e6:.2f} MHz, Receiver is on {rx_f/1e6:.2f} MHz "
                       "(normal for a moment during a hop).")

    tx_sf = st.session_state.get("tx_sf")
    rx_sf = st.session_state["live_sf"]
    if tx_sf:
        if tx_sf == rx_sf:
            st.success(f"✔ Spreading Factor check: both sides are on SF{rx_sf} "
                       f"({'baseline' if rx_sf == BASE_SF else 'hopped — more robust, lower data rate'}).")
        else:
            st.warning(f"Transmitter reports SF{tx_sf}, Receiver is on SF{rx_sf} (normal briefly during a hop).")

    m = st.columns(6)
    if not ld.empty:
        lp = ld.iloc[-1]
        m[0].metric("Freq", f"{cur_mhz:.2f} MHz")
        m[1].metric("SF", f"{st.session_state['live_sf']}")
        m[2].metric("RSSI", f"{lp['rssi']:.0f} dBm")
        m[3].metric("SNR", f"{lp['snr']:.1f} dB")
        m[4].metric("Loss", f"{lp['loss']:.1f}%")
        m[5].metric("Packets", len(ld))
    else:
        for c, l in zip(m, ["Freq", "SF", "RSSI", "SNR", "Loss", "Packets"]):
            c.metric(l, "--")

    g1, g2 = st.columns([1, 2])
    with g1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=round(prob * 100, 1),
            title={"text": "Jamming probability (%)"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "darkred" if prob > 0.5 else "green"},
                   "steps": [{"range": [0, 40], "color": "#06d6a0"},
                             {"range": [40, 70], "color": "#ffd166"},
                             {"range": [70, 100], "color": "#ef476f"}]}))
        st.plotly_chart(fig, use_container_width=True, key="gauge")
    with g2:
        if not ld.empty:
            t = ld.tail(60).copy()
            t["ML"] = t["ml_pred"].map({0: "Normal", 1: "Jammed"})
            st.plotly_chart(px.scatter(t, x="timestamp", y="rssi", color="ML",
                                       color_discrete_map={"Normal": "#06d6a0", "Jammed": "#ef476f"},
                                       title="RSSI with ML classification"),
                            use_container_width=True, key="pred")

    if not ld.empty:
        t = ld.tail(80)
        a, b = st.columns(2)
        a.plotly_chart(px.line(t, x="timestamp", y="rssi", title="RSSI (dBm)"), use_container_width=True, key="l1")
        b.plotly_chart(px.line(t, x="timestamp", y="snr", title="SNR (dB)"), use_container_width=True, key="l2")
        c, d = st.columns(2)
        c.plotly_chart(px.line(t, x="timestamp", y="rssi_var", title="RSSI variance"), use_container_width=True, key="l3")
        d.plotly_chart(px.scatter(t, x="timestamp", y="loss", color="phase",
                                  title="Measured packet loss (%) by phase"), use_container_width=True, key="l4")

    st.write("### ⚖️ Static LoRa (no AI) vs Adaptive LoRa (with AI) — packet loss")
    if not ld.empty and (ld["phase"] == "jammed").any():
        t = ld.reset_index(drop=True).copy()
        jam_avg = t.loc[t["phase"] == "jammed", "loss"].mean()
        static_loss = t["loss"].copy()
        static_loss[t["phase"] == "hopped"] = jam_avg   # counterfactual: jammer never escaped
        comp = pd.DataFrame({
            "timestamp": t["timestamp"],
            "Adaptive LoRa — measured (with AI hop)": t["loss"],
            "Static LoRa — estimated (no hop)": static_loss,
        })
        st.plotly_chart(
            px.line(comp, x="timestamp",
                   y=["Adaptive LoRa — measured (with AI hop)", "Static LoRa — estimated (no hop)"],
                   color_discrete_map={"Adaptive LoRa — measured (with AI hop)": "green",
                                       "Static LoRa — estimated (no hop)": "red"}),
            use_container_width=True, key="static_vs_adaptive")
        st.caption(
            "Both curves are the same real link up to the hop — there's only one radio, so they can't "
            "be measured at the same time after that. From the hop onward, the **green** curve is real "
            f"measured loss on the new channel. The **red** curve is not measured; it assumes the jammer "
            f"kept jamming on 433.00 MHz forever and holds loss at the jammed-phase average "
            f"({jam_avg:.1f}%), since that's what our emulated jammer actually does while it's on."
        )
        cm1, cm2, cm3 = st.columns(3)
        cm1.metric("Avg Static Loss (estimated)", f"{static_loss.mean():.1f}%")
        cm2.metric("Avg Adaptive Loss (measured)", f"{t['loss'].mean():.1f}%")
        cm3.metric("Improvement", f"{static_loss.mean() - t['loss'].mean():.1f} pts lower loss")
    else:
        st.caption("Trigger the jammer in this tab — this comparison appears once jamming has been detected.")

    st.write("#### Packets by phase")
    if not ld.empty:
        summ = ld.groupby("phase").agg(packets=("pkt", "count"), avg_loss=("loss", "mean"),
                                       avg_rssi=("rssi", "mean"), avg_snr=("snr", "mean")).round(2)
        st.dataframe(summ, use_container_width=True)
    ev = st.session_state["events"]
    e1, e2 = st.columns(2)
    e1.metric("Detection → hop latency",
              f"{ev['hop'] - ev['jam_on']:.1f} s" if ev["jam_on"] and ev["hop"] else "--")
    e2.metric("Jam start → first clean packet on new channel",
              f"{ev['recovered'] - ev['jam_on']:.1f} s" if ev["jam_on"] and ev["recovered"] else "--")


# ───────────── TAB 3 ─────────────
def render_diff():
    st.subheader("Attenuation vs jamming")
    st.markdown("""
RSSI or SNR alone cannot prove jamming: distance, walls, antenna problems and interference overlap in symptoms.
The model uses several features together; a result is only *confirmed* jamming when your test setup provides evidence.

| Characteristic | Attenuation (distance / wall) | Interference / jamming |
| :-- | :-- | :-- |
| RSSI | Low but fairly stable | Often fluctuates |
| Packet loss | Usually low until near sensitivity limit | Bursty, can be severe |
| SNR | Falls together with RSSI | Falls while RSSI may stay high |
| Typical label | Normal (0) | Jammed (1) |
""")
    st.warning("In this prototype the jamming is **emulated** by the Transmitter (packet drops + weak, "
               "varying TX power). The model learns that emulated pattern, not a real RF jammer.")
    if not df.empty:
        a, b = st.columns(2)
        a.plotly_chart(px.line(df.tail(200), x="timestamp", y="rssi_var", title="RSSI variance"),
                       use_container_width=True, key="d1")
        b.plotly_chart(px.line(df.tail(200), x="timestamp", y="loss", title="Packet loss (%)"),
                       use_container_width=True, key="d2")


# ───────────── TAB 4 ─────────────
def render_data():
    st.subheader("Raw dataset (signal_log.csv)")
    if df.empty:
        st.info("No packets logged yet.")
        return
    n0, n1 = int((df["label"] == 0).sum()), int((df["label"] == 1).sum())
    st.write(f"Total `{len(df)}` | Normal `{n0}` | Jammed `{n1}`")
    st.dataframe(df.tail(100).sort_index(ascending=False), use_container_width=True)
    a, b = st.columns(2)
    a.download_button("⬇ Download CSV", df.to_csv(index=False), file_name="signal_log.csv", mime="text/csv")
    confirm2 = b.checkbox("Confirm clear", key="reset_confirm2")
    if b.button("🗑 Clear dataset & model", use_container_width=True, disabled=not confirm2):
        st.session_state.update(data=pd.DataFrame(columns=TRAIN_COLS), model=None,
                                trained_stats=None, tab1_done=False, cur_pkt={})
        for pth in (CSV_PATH, MODEL_PATH, STATS_PATH):
            if os.path.exists(pth):
                os.remove(pth)
        for bpath in glob.glob(os.path.join(MODEL_BACKUP_DIR, "model_*.pkl")):
            os.remove(bpath)
        st.session_state["reset_confirm2"] = False
        st.rerun()


with tab_feed:
    render_feed()
with tab_live:
    render_live()
with tab_diff:
    render_diff()
with tab_data:
    render_data()
