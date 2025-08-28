import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def ads_model(t, w1, t1, w2, t2, b, m):
    return b + m * ((1 / (1 + np.exp(-w1 * (t - t1)))) *
                    (1 - (1 / (1 + np.exp(-w2 * (t - t2))))))

def fit_gcc_series(dates, gcc_values):
    doy = pd.to_datetime(dates).dt.dayofyear
    t = np.array(doy)
    y = np.array(gcc_values)
    
    p0 = [0.2, 130, 0.2, 280, 0.33, 0.06]
    bounds = ([0.01, 90, 0.01, 220, 0.3, 0.01],
              [2.0, 160, 2.0, 330, 0.4, 0.15])

    popt, _ = curve_fit(ads_model, t, y, p0=p0, bounds=bounds, maxfev=20000)

    t_fit = np.linspace(t.min(), t.max(), 300)
    y_fit = ads_model(t_fit, *popt)
    
    return t_fit, y_fit, popt

def extract_season_dates(t_range, popt):
    y_fit = ads_model(t_range, *popt)
    b, m = popt[4], popt[5]

    peak_idx = np.argmax(y_fit)
    threshold_10 = b + 0.1 * m
    threshold_90 = b + 0.9 * m

    sos, eos = None, None

    for t, y in zip(t_range[:peak_idx], y_fit[:peak_idx]):
        if y > threshold_10:
            sos = t
            break

    for t, y in zip(t_range[peak_idx:], y_fit[peak_idx:]):
        if y < threshold_90:
            eos = t
            break

    los = eos - sos if sos is not None and eos is not None else None
    peak_time = t_range[peak_idx]
    peak_value = y_fit[peak_idx]
    
    return sos, eos, los, peak_time, peak_value



df = pd.read_csv("../gcc.csv")
df['datetime'] = pd.to_datetime(df['datetime'])
for col in ['mean', 'std', 'min', 'max']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

mask = df['mean'].notna()
dates = df.loc[mask, 'datetime']
vals  = df.loc[mask, 'mean']
t_fit, y_fit, popt = fit_gcc_series(dates, vals)
sos, eos, los, peak_time, peak_value = extract_season_dates(t_fit, popt)
b, m = popt[4], popt[5]
thr10 = b + 0.1 * m
thr90 = b + 0.9 * m

t_raw = pd.to_datetime(dates).dt.dayofyear.values
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(t_raw, vals, s=14, alpha=0.25, label='GCC (mean)', zorder=2, color='green')
ax.plot(t_fit, y_fit, linewidth=3.5, label='ADS fit', zorder=3, color='green')

if sos is not None:
    ax.axvline(sos, linestyle='--', linewidth=1.6, alpha=0.9, label='SOS (10% up)', zorder=1, color='red')
if peak_time is not None:
    ax.axvline(peak_time, linestyle='--', linewidth=1.6, alpha=0.9, label='Peak', zorder=1, color='purple')
if eos is not None:
    ax.axvline(eos, linestyle='--', linewidth=1.6, alpha=0.9, label='EOS (90% down)', zorder=1, color='blue')
if (sos is not None) and (eos is not None):
    ax.annotate(
        f'LOS ≈ {eos - sos:.0f} days',
        xy=((sos + eos) / 2, b + 0.5 * m),
        xytext=(0, 18),
        textcoords='offset points',
        ha='center', va='bottom',
        arrowprops=dict(arrowstyle='-|>', lw=1, shrinkA=0, shrinkB=0)
    )

ax.set_xlim(1, 366)
ax.set_xlabel('Day of Year')
ax.set_ylabel('GCC')
ax.set_title('GCC with asymmetric double sigmoid fit and Key Phenology Dates')
ax.grid(True, alpha=0.25)
ax.legend(ncol=2, frameon=False)
plt.tight_layout()
plt.show()

print(f"SOS (10% up): {sos:.1f} DOY" if sos is not None else "SOS: None")
print(f"Peak time: {peak_time:.1f} DOY, Peak GCC: {peak_value:.4f}" if peak_time is not None else "Peak: None")
print(f"EOS (90% down): {eos:.1f} DOY" if eos is not None else "EOS: None")
print(f"LOS: {los:.1f} days" if los is not None else "LOS: None")
