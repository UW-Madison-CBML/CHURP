#!/bin/bash
pip install pandas matplotlib seaborn


# 2. Define the bird array
bird_ids=( 
    "0"
    "1"
    "2"
    "3"
    "4"
    "5"
    "6"
    "7"
    "8"
    "9"
    "10" 
)
for id in "${bird_ids[@]}"; do
	mkdir "comparison_${id}"
	tar -xzf "comparison_${id}.tar.gz" -C "comparison_${id}" 
done

touch metrics.csv
sed -n '1p' "comparison_0/comparison/comparison_data.csv" > metrics.csv

for id in "${bird_ids[@]}"; do
        sed -n '2p' "comparison_${id}/comparison/comparison_data.csv" >> metrics.csv
done

mkdir metrics

python plot_metrics.py "metrics.csv"

mv metrics.csv metrics/
mv "bird_model_comparison.png" metrics/

tar -czf metrics.tar.gz metrics/
