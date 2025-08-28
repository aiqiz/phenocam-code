import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../phenocam02/Aug_calibration/gcc.csv")


df['datetime'] = pd.to_datetime(df['datetime'])
for col in ['mean', 'std', 'min', 'max']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df['datetime'], df['mean'], color='green', label='Mean')
plt.fill_between(df['datetime'],
                 df['mean'] - df['std'],
                 df['mean'] + df['std'],
                 color='green', alpha=0.2, label='±1 Std Dev')

plt.xlabel("Datetime")
plt.ylabel("Value")
plt.title("gcc raw")
plt.legend()
plt.tight_layout()
plt.show()
