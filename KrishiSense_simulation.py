"""KrishiSense demo simulation (pure Python/NumPy).
Three virtual testbeds: (A) irrigation on a 6x8 zone farm using an FAO-56 style
root-zone water balance, (B) disease spread + camera detection, (C) 30-day produce storage.
All numbers produced here are SIMULATION outputs under the stated assumptions."""
import numpy as np, json

R, Cn, DAYS = 6, 8, 90
NZ = R * Cn

def kc_curve(d):
    if d < 20: return 0.4
    if d < 50: return 0.4 + (1.15 - 0.4) * (d - 20) / 30
    if d < 75: return 1.15
    return 1.15 - (1.15 - 0.7) * (d - 75) / 15

def make_world(seed):
    rng = np.random.default_rng(seed)
    gx = np.tile(np.linspace(0, 1, Cn), (R, 1))
    taw = 70 + 45 * gx + rng.normal(0, 6, (R, Cn))            # mm, sandy (left) to clay (right)
    taw = np.clip(taw, 60, 125)
    et0 = 5 + 1.2 * np.sin(2 * np.pi * np.arange(DAYS) / 45) + rng.normal(0, 0.6, DAYS)
    for _ in range(2):                                         # two hot spells
        s = rng.integers(10, DAYS - 12); et0[s:s + 6] += 1.8
    et0 = np.clip(et0, 3, 8.5)
    rain = np.where(rng.random(DAYS) < 0.07, np.clip(rng.exponential(14, DAYS), 3, 45), 0.0)
    bias = rng.normal(0, 2.0, (R, Cn))                         # per-sensor calibration error (mm)
    return dict(rng=rng, taw=taw, et0=et0, rain=rain, bias=bias)

def run_irrigation(seed, mode, record=False):
    w = make_world(seed); rng = np.random.default_rng(seed + 1000)
    taw, et0, rain, bias = w['taw'], w['et0'], w['rain'], w['bias']
    raw = 0.5 * taw
    dr = 0.3 * raw
    irr_tot = np.zeros((R, Cn)); dp_tot = np.zeros((R, Cn)); eta_s = np.zeros((R, Cn)); etc_s = np.zeros((R, Cn))
    stress_days = np.zeros((R, Cn))
    hist = []
    for d in range(DAYS):
        irr = np.zeros((R, Cn)); reading = None; fc = 0.0; skip = np.zeros((R, Cn), bool)
        if mode == 'baseline':
            if d % 5 == 0: irr[:] = 30.0
        else:
            reading = dr + bias + rng.normal(0, 2.0, (R, Cn))
            fc = (rain[d + 1] if d + 1 < DAYS else 0) * max(0.0, 1 + rng.normal(0, 0.5))
            if fc < 10:
                need = reading >= 0.7 * raw
                irr = np.where(need, np.minimum(np.maximum(reading, 0), 40.0), 0.0)
            else:
                skip = reading >= 0.7 * raw
        dr_after_irr = dr - irr
        dp = np.maximum(-dr_after_irr, 0); dr_after_irr = np.maximum(dr_after_irr, 0)
        dr_mid = dr_after_irr - rain[d]
        dp += np.maximum(-dr_mid, 0); dr_mid = np.maximum(dr_mid, 0)
        etc = kc_curve(d) * et0[d]
        ks = np.where(dr_mid <= raw, 1.0, np.clip((taw - dr_mid) / (taw - raw), 0, 1))
        eta = etc * ks
        dr = np.clip(dr_mid + eta, 0, taw)
        irr_tot += irr; dp_tot += dp; eta_s += eta; etc_s += etc; stress_days += (ks < 0.999)
        if record:
            hist.append(dict(day=d, dr=dr.copy(), irr=irr.copy(), ks=ks.copy(), fc=float(fc), skip=skip.copy(), rain=float(rain[d]),
                             et0=float(et0[d]), reading=None if reading is None else reading.copy(),
                             irr_cum=irr_tot.copy().mean(), dp_cum=dp_tot.copy().mean(), taw=taw.copy(), raw=raw.copy()))
    ry = np.clip(1 - 1.0 * (1 - eta_s / etc_s), 0, 1)
    out = dict(irr_mm=float(irr_tot.mean()), dp_mm=float(dp_tot.mean()), yield_rel=float(ry.mean()),
               stress_days=float(stress_days.mean()), rain_mm=float(rain.sum()))
    return (out, hist) if record else out

def neighbours(r, c):
    for dr_, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr_, c + dc
        if 0 <= rr < R and 0 <= cc < Cn: yield rr, cc

def run_disease(seed, mode, record=False):
    rng = np.random.default_rng(seed + 2000)
    s = np.zeros((R, Cn)); prot = np.zeros((R, Cn))
    for _ in range(int(rng.integers(1, 3))):
        r, c = int(rng.integers(0, R)), int(rng.integers(0, Cn)); s[r, c] = 0.02
    first_inf_day = None
    sev_sum = np.zeros((R, Cn)); sprays = 0; detect_day = None; fp = 0; hist = []; prev_seen = np.zeros((R, Cn), bool)
    TPR, FPR = 0.85, 0.02                                       # ASSUMED detector behaviour
    for d in range(60):
        infected = s > 0
        if infected.any() and first_inf_day is None: first_inf_day = d
        new = s.copy()
        growth = 0.30 * s * (1 - s) * np.where(prot > 0, 0.15, 1.0)
        new += growth
        for r in range(R):
            for c in range(Cn):
                if s[r, c] == 0:
                    pr = sum(0.35 * s[rr, cc] for rr, cc in neighbours(r, c))
                    pr *= 0.15 if prot[r, c] > 0 else 1.0
                    if rng.random() < pr: new[r, c] = 0.01
        s = np.clip(new, 0, 1); prot = np.maximum(prot - 1, 0)
        sprayed = np.zeros((R, Cn), bool); det = np.zeros((R, Cn), bool)
        if mode == 'baseline':
            if d % 7 == 6 and (s > 0.25).any():
                sprayed[:] = True
        else:
            raw_seen = np.where(s >= 0.03, rng.random((R, Cn)) < TPR, rng.random((R, Cn)) < FPR)
            seen = raw_seen & prev_seen                              # confirm on 2 consecutive days
            prev_seen = raw_seen
            det = seen
            for r, c in zip(*np.where(seen)):
                sprayed[r, c] = True
                for rr, cc in neighbours(r, c): sprayed[rr, cc] = True
            fp += int(((s < 0.03) & seen).sum())
            if detect_day is None and ((s >= 0.03) & seen).any(): detect_day = d
        if sprayed.any():
            s[sprayed] *= 0.15; prot[sprayed] = 7; sprays += int(sprayed.sum())
            s[s < 0.004] = 0
            if mode == 'baseline' and detect_day is None: detect_day = d
        sev_sum += s
        if record: hist.append(dict(day=d, s=s.copy(), sprayed=sprayed.copy(), det=det.copy(), sprays=sprays))
    loss = np.clip(0.8 * sev_sum / 60, 0, 1)
    out = dict(yield_loss=float(loss.mean()), sprays=sprays, first_inf=first_inf_day, detect_day=detect_day, fp=fp)
    return (out, hist) if record else out

def run_storage(seed, mode, record=False):
    rng = np.random.default_rng(seed + 3000)
    H = 30 * 24
    t = np.arange(H)
    t_out = 31 + 6 * np.sin(2 * np.pi * (t % 24 - 9) / 24) + 2.0 * np.sin(2 * np.pi * t / (24 * 9)) + rng.normal(0, 0.3, H) + rng.normal(0, 1.0)
    tin = 28.0; loss_acc = 0.0; energy = 0.0
    C = 4.9                                              # kWh/K thermal mass (about 5 t produce)
    tau_closed, tau_vent = 12.0, 3.0
    T_SET, k_ref, Q10 = 24.0, 0.0025, 2.2
    hist = []
    for h in range(H):
        if mode == 'baseline':
            tin += (t_out[h] - tin) / tau_closed
        else:
            if t_out[h] < tin - 1.0:
                tin += (t_out[h] - tin) / tau_vent; energy += 0.3          # fan, kWh per hour
            else:
                tin += (t_out[h] - tin) / tau_closed
                if tin > T_SET + 0.5:
                    q = min(3.0, (tin - T_SET) * C)                         # kWh thermal removed this hour
                    tin -= q / C; energy += q / 3.0                         # COP = 3
        k = k_ref * Q10 ** ((tin - 20) / 10) / 24
        loss_acc += k
        if record: hist.append((h, float(t_out[h]), float(tin), 1 - np.exp(-loss_acc), energy))
    out = dict(spoilage=float(1 - np.exp(-loss_acc)), energy_kwh=float(energy), mean_t=float(np.mean([x[2] for x in hist])) if record else None)
    return (out, hist) if record else out

def monte_carlo(n=30):
    res = {}
    for name, fn in (('irr', run_irrigation), ('dis', run_disease), ('sto', run_storage)):
        for mode in ('baseline', 'smart'):
            rows = [fn(1000 + i, mode) for i in range(n)]
            res[(name, mode)] = rows
    return res

def summarize(res):
    def m(name, mode, key):
        v = np.array([r[key] for r in res[(name, mode)] if r[key] is not None], float)
        return float(v.mean()), float(v.std())
    S = {}
    ib, is_ = m('irr','baseline','irr_mm')[0], m('irr','smart','irr_mm')[0]
    S['water_baseline_mm'], S['water_smart_mm'] = ib, is_
    S['water_saved_pct'] = 100 * (ib - is_) / ib
    S['dp_baseline_mm'], S['dp_smart_mm'] = m('irr','baseline','dp_mm')[0], m('irr','smart','dp_mm')[0]
    S['yield_baseline'], S['yield_smart'] = m('irr','baseline','yield_rel')[0], m('irr','smart','yield_rel')[0]
    S['yield_baseline_sd'], S['yield_smart_sd'] = m('irr','baseline','yield_rel')[1], m('irr','smart','yield_rel')[1]
    S['stress_baseline'], S['stress_smart'] = m('irr','baseline','stress_days')[0], m('irr','smart','stress_days')[0]
    sb = np.array([r['water_saved'] for r in []]) if False else None
    # per-seed paired water saving
    pairs = [(a['irr_mm'], b['irr_mm']) for a, b in zip(res[('irr','baseline')], res[('irr','smart')])]
    sv = np.array([100 * (a - b) / a for a, b in pairs]); S['water_saved_pct_sd'] = float(sv.std()); S['water_saved_min'] = float(sv.min()); S['water_saved_max'] = float(sv.max())
    S['dis_loss_baseline'], S['dis_loss_smart'] = m('dis','baseline','yield_loss')[0], m('dis','smart','yield_loss')[0]
    S['sprays_baseline'], S['sprays_smart'] = m('dis','baseline','sprays')[0], m('dis','smart','sprays')[0]
    S['spray_saved_pct'] = 100 * (S['sprays_baseline'] - S['sprays_smart']) / S['sprays_baseline']
    dl = []
    for rows in (res[('dis','baseline')], res[('dis','smart')]):
        dl.append(np.mean([r['detect_day'] - r['first_inf'] for r in rows if r['detect_day'] is not None and r['first_inf'] is not None]))
    S['detect_delay_baseline'], S['detect_delay_smart'] = float(dl[0]), float(dl[1])
    S['fp_per_season'] = m('dis','smart','fp')[0]
    S['spoil_baseline'], S['spoil_smart'] = m('sto','baseline','spoilage')[0], m('sto','smart','spoilage')[0]
    S['spoil_reduction_pct'] = 100 * (S['spoil_baseline'] - S['spoil_smart']) / S['spoil_baseline']
    S['energy_kwh'] = m('sto','smart','energy_kwh')[0]
    return S

if __name__ == '__main__':
    res = monte_carlo(30)
    S = summarize(res)
    for k, v in S.items(): print(f'{k:28s} {v:.3f}')
    json.dump(S, open('results.json', 'w'), indent=2)
