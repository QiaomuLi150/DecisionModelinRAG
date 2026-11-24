# Reproducible figure-data entry point
# This script intentionally consumes only the adjacent CSV artifacts.
import csv, pathlib
root=pathlib.Path(__file__).parent.parent
for name in ['E6_4_FIGURE_DATA.csv','E6_4_CROSS_WORKLOAD_METRICS.csv']:
    p=root/'scientific_synthesis'/name
    print(name, sum(1 for _ in p.open())-1 if p.exists() else 'MISSING')
