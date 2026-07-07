import matplotlib.pyplot as plt
import numpy as np

# Data
datasets = ['3DRoad\n(434K pts)', 'KITTI\n(1.3M pts)', '3DIono\n(1.8M pts)', 'Porto\n(10M pts)']
arkade_times = [3.3769, 4.47654, 26.7097, 3.5577]
gprt_times = [2.09393, 4.62302, 4.99716, 15.5375]

x = np.arange(len(datasets))  # the label locations
width = 0.35  # the width of the bars

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars
bars1 = ax.bar(x - width/2, arkade_times, width, label='Arkade (C++/OWL)', color='#4C72B0', edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, gprt_times, width, label='GPRT DSL (Rust/OptiX)', color='#DD8452', edgecolor='black', linewidth=1.2)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Execution Time (Seconds)', fontsize=12, fontweight='bold')
ax.set_title('Exact Unbounded k-NN Performance (k=5, 400K Queries, RTX 4070 Ti)', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.legend(fontsize=11, loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Function to add value labels on top of bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

add_labels(bars1)
add_labels(bars2)


plt.tight_layout()
plt.savefig('knn_head_to_head.png', dpi=300, bbox_inches='tight')
print("✅ Chart saved as knn_head_to_head.png")
plt.show()