import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use("Agg")

df = pd.read_table('metric/resource_consumption.tsv', sep='\t')
df.columns = ['Cell num', 'Gene num', 'Max Mem(GB)', 'Time(min)']

df['Cell * Gene'] = df['Cell num'] * df['Gene num']  / 1e7
df['Cell * Gene'] = df['Cell * Gene'].astype('int')

plt.figure(figsize=(12, 6))

# Plot for Cell * Gene vs Time
plt.subplot(1, 2, 1)
plt.scatter(df["Cell * Gene"], df["Time(min)"], label="Cell * Gene vs Time", color='green')
p1 = Polynomial.fit(df["Cell * Gene"], df["Time(min)"], deg=2)
plt.plot(*p1.linspace(), label="Fitted curve for Cell * Gene vs Time", color='green')
plt.xlabel('Cell * Gene (x10⁷)')
plt.ylabel('Time (min)')
plt.legend()
plt.title('Fitted Curve for Cell * Gene vs Time')

# Plot for Cell * Gene vs Max Memory
plt.subplot(1, 2, 2)
plt.scatter(df["Cell * Gene"], df["Max Mem(GB)"], label="Cell * Gene vs Max Mem", color='green')
p2 = Polynomial.fit(df["Cell * Gene"], df["Max Mem(GB)"], deg=2)
plt.plot(*p2.linspace(), label="Fitted curve for Cell * Gene vs Max Mem", color='green')
plt.xlabel('Cell * Gene (x10⁷)')
plt.ylabel('Max Memory (GB)')
plt.legend()
plt.title('Fitted Curve for Cell * Gene vs Max Memory')

plt.tight_layout()

plt.savefig(f'figure/fig_S1A_resource_consumption.pdf', dpi=300, bbox_inches="tight")
plt.close()
