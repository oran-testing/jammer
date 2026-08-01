# MILCOM Demo — Running the Jammer Attacks

Step-by-step guide for running the software-defined USRP jammer during the
MILCOM demo. Covers the two attack modes used in the demo (**barrage** and
**random**) and the **clean** baseline (jammer off).

> **Safety / legal:** This jammer transmits real RF. Only run it over the air
> inside a shielded lab or on frequencies you are authorized to use. Keep the
> jammer co-channel with the victim gNB and nothing else.

---

## 1. What each attack does

| Mode | `attack_type` | RF behavior | Config file |
|------|---------------|-------------|-------------|
| **Clean** | *(jammer off)* | No interference — baseline traffic | *(none — do not start the jammer)* |
| **Barrage** | `barrage` | Wideband noise, transmitted **continuously** across the band | `configs/barrage.yaml` |
| **Random** | `random` | Same wideband noise, gated **50 ms ON / 50 ms OFF** (bursty) | `configs/random.yaml` |

The demo classifier is trained on **clean / barrage / random**.

---

## 2. Requirements

- Linux host with a **USRP B210** connected (attack radio)
- **UHD** drivers installed and the device visible (`uhd_find_devices`)
- Build tools: `cmake`, `make`, a C++ compiler

```bash
sudo apt update
sudo apt install build-essential cmake libuhd-dev uhd-host
```

---

## 3. Build the jammer (one time)

From the `jammer/` directory of the repo:

```bash
mkdir -p build && cd build
cmake ..
make -j
```

This produces the `jammer` binary in `jammer/build/`.

> Rebuild (`make -j`) only if you change C++ source. Editing a YAML config does
> **not** require a rebuild.

---

## 4. Confirm the USRP is detected

```bash
uhd_find_devices
```

You should see your B210 listed. If not, replug the USRP and re-run before
starting the demo.

---

## 5. Run each attack

All commands are run from `jammer/build/`. Stop any attack with **Ctrl+C**.

### 5.1 Clean (baseline)

Do **not** start the jammer. Keep the gNB + UE link running with traffic so the
detector sees a clean KPI baseline.

### 5.2 Barrage jamming

```bash
./jammer --config ../configs/barrage.yaml
```

You will see repeating `Transmitting (barrage, fresh buffer)....` lines while it
runs. This transmits continuous wideband noise.

### 5.3 Random (bursty) jamming

```bash
./jammer --config ../configs/random.yaml
```

Output alternates between `Transmitting (random) ON ...` and
`Transmitting (random) IDLE ...` every 50 ms, producing the bursty signature.

---

## 6. Key config parameters

Edit these in the YAML files under `configs/` before the demo to match your
lab. Defaults below are from the checked-in configs.

| Field | Meaning | Barrage / Random default |
|-------|---------|--------------------------|
| `attack_type` | `barrage` or `random` | per file |
| `center_frequency` | Victim carrier center, Hz | `1842050000` (Band 3) |
| `bandwidth` | Jam bandwidth, Hz | `40e6` |
| `sampling_freq` | USRP TX sample rate | `23.04e6` |
| `tx_gain` | USRP TX gain, dB — **controls jamming strength** | `50` |
| `amplitude` | Waveform amplitude (0–1) | `0.7` |
| `num_samples` | Samples per TX buffer | `20000` |
| `burst_duration_ms` | (random only) ON time | `50` |
| `idle_duration_ms` | (random only) OFF time | `50` |
| `device_args` | UHD device selector | `type=b200` |

> **Match `center_frequency` and `sampling_freq` to your gNB.** The jammer must
> sit on the same carrier as the victim gNB for the attack to register in the
> KPIs.

---

## 7. Command-line overrides (no rebuild needed)

Any config field can be overridden on the command line, which is handy for
tuning gain live during the demo without editing files:

```bash
# Raise barrage power on the fly
./jammer --config ../configs/barrage.yaml --tx_gain 55

# Retune to a different carrier
./jammer --config ../configs/barrage.yaml --center_frequency 3619200000

# Switch mode without editing YAML
./jammer --config ../configs/barrage.yaml --attack_type random

# Change the random duty cycle
./jammer --config ../configs/random.yaml --burst_duration_ms 100 --idle_duration_ms 100
```

Supported override flags: `--attack_type`, `--tx_gain`, `--center_frequency`,
`--bandwidth`, `--sampling_freq`, `--amplitude`, `--amplitude_width`,
`--num_samples`, `--initial_phase`, `--burst_duration_ms`, `--idle_duration_ms`,
`--device_args`, `--write_iq`, `--write_csv`.

If KPIs still look clean when jamming, raise `--tx_gain` in ~2 dB steps.

---

## 8. Suggested demo sequence

Run the detector alongside so the audience sees the label change as you toggle
the jammer:

```text
1. Clean      — jammer OFF                → detector shows "clean"
2. Barrage    — ./jammer --config ../configs/barrage.yaml
                (Ctrl+C to stop)          → detector shows "barrage"
3. Clean      — jammer OFF again          → detector returns to "clean"
4. Random     — ./jammer --config ../configs/random.yaml
                (Ctrl+C to stop)          → detector shows "random"
```

The detector needs ~30 s of KPI history (its sliding window) before the label
settles after each toggle.

**Live detector (separate terminal, from the `ml/` directory of the repo):**

```bash
source .venv/bin/activate
python detect_live_gnb.py --config config_gnb.yaml --run-id 20260702_190948
```

---

## 9. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Error: center_frequency, sampling_freq, and tx_gain must be set` | Fill those fields in the YAML (or pass via CLI). |
| No USRP found / UHD error | Run `uhd_find_devices`; replug the B210; check `device_args`. |
| KPIs stay "clean" under jamming | Increase `tx_gain`; confirm `center_frequency` matches the gNB. |
| `Unknown attack_type` | Use exactly `barrage` or `random`. |
| Old behavior after editing `.cc` | Rebuild with `make -j` in `build/`. |
