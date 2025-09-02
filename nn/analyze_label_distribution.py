#!/usr/bin/env python3
"""
Script to analyze label distribution and proportions in the training dataset.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

# Configuration
TRAIN_SET_DIR = "train_data/train_set"
LABELS_FILE = "train_data/labels.csv"
OUTPUT_DIR = "train_data"  # output directory

# Configure matplotlib fonts (keeps Chinese-capable fonts available if needed)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_label_mapping():
    """Load label mapping from the labels CSV file."""
    label_df = pd.read_csv(LABELS_FILE)
    label_to_id = {row["label_name"]: index for index, row in label_df.iterrows()}
    return label_to_id

def process_labels_from_string(labels_str):
    """Extract a list of labels from a label string (separator '|')."""
    if not isinstance(labels_str, str):
        return []
    
    labels = []
    for label in labels_str.split("|"):
        label = label.strip()
        if label:
            labels.append(label)
    return labels

def analyze_label_distribution():
    """Analyze label distribution across the training set files."""
    print("=== Training set label distribution analysis ===\n")

    # Load label mapping
    label_to_id = load_label_mapping()
    total_labels = len(label_to_id)
    print(f"Total number of labels: {total_labels}")
    print(f"Label list: {list(label_to_id.keys())}\n")

    # Read all training CSV files
    csv_files = [f for f in os.listdir(TRAIN_SET_DIR) if f.endswith(".csv")]
    print(f"Training set files: {csv_files}")

    all_labels = []
    all_samples = []
    total_samples = 0

    for csv_file in csv_files:
        file_path = os.path.join(TRAIN_SET_DIR, csv_file)
        print(f"Processing: {file_path}")

        df = pd.read_csv(file_path)
        file_samples = len(df)
        total_samples += file_samples

        for index, row in df.iterrows():
            labels = process_labels_from_string(row["labels"])
            all_labels.extend(labels)
            all_samples.append({
                'text': row['text'],
                'labels': labels,
                'label_count': len(labels)
            })

        print(f"  - Samples count: {file_samples}")

    print(f"\nTotal samples: {total_samples}")
    print(f"Total label instances: {len(all_labels)}")
    print(f"Average number of labels per sample: {len(all_labels)/total_samples:.2f}")

    # Count label frequencies
    label_counter = Counter(all_labels)

    # Distribution statistics
    print("\n=== Label distribution statistics ===")
    print(f"{'Label':<20} {'Count':<8} {'Percentage(%)':<15} {'Cumulative(%)':<12}")
    print("-" * 65)

    # Sort by frequency
    sorted_labels = label_counter.most_common()
    cumulative_count = 0
    label_stats = []

    for label, count in sorted_labels:
        percentage = (count / len(all_labels)) * 100
        cumulative_count += count
        cumulative_percentage = (cumulative_count / len(all_labels)) * 100

        print(f"{label:<20} {count:<8} {percentage:<15.2f} {cumulative_percentage:<12.2f}")
        label_stats.append({
            'label': label,
            'count': count,
            'percentage': percentage,
            'cumulative_percentage': cumulative_percentage
        })

    # Check for unused labels
    unused_labels = set(label_to_id.keys()) - set(all_labels)
    if unused_labels:
        print(f"\nUnused labels ({len(unused_labels)}): {', '.join(sorted(unused_labels))}")
    else:
        print("\nAll defined labels are used")

    # Distribution of label counts per sample
    label_count_per_sample = [sample['label_count'] for sample in all_samples]
    label_count_distribution = Counter(label_count_per_sample)

    print(f"\n=== Label count per sample distribution ===")
    print(f"{'Label Count':<12} {'Sample Count':<12} {'Percentage(%)':<12}")
    print("-" * 40)

    for label_count in sorted(label_count_distribution.keys()):
        sample_count = label_count_distribution[label_count]
        percentage = (sample_count / total_samples) * 100
        print(f"{label_count:<12} {sample_count:<12} {percentage:<12.2f}")

    print(f"\nLabel count stats:")
    print(f"  - Minimum labels per sample: {min(label_count_per_sample)}")
    print(f"  - Maximum labels per sample: {max(label_count_per_sample)}")
    print(f"  - Average labels per sample: {np.mean(label_count_per_sample):.2f}")
    print(f"  - Median labels per sample: {np.median(label_count_per_sample):.2f}")

    # Create visualizations
    create_visualizations(label_stats, label_count_distribution, total_samples)

    return label_stats, all_samples

def create_visualizations(label_stats, label_count_distribution, total_samples):
    """Create visualization charts for label statistics and distributions."""

    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(24, 16))

    # Frequency distribution of all labels (x-axis labels rotated)
    plt.subplot(3, 3, 1)
    all_labels = [stat['label'] for stat in label_stats]
    all_counts = [stat['count'] for stat in label_stats]
    
    bars = plt.bar(range(len(all_labels)), all_counts, color='skyblue', alpha=0.7)
    plt.xlabel('Label')
    plt.ylabel('Frequency')
    plt.title(f'All label frequency distribution (total {len(all_labels)} labels)')
    plt.xticks(range(len(all_labels)), all_labels, rotation=90, ha='center', fontsize=8)
    
    # Add values on bars with count > 5 to avoid clutter
    for i, (bar, count) in enumerate(zip(bars, all_counts)):
        if count > 5:  # only label taller bars to reduce clutter
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom', fontsize=6)
    
    plt.tight_layout()
    
    # Top 20 labels frequency distribution (horizontal)
    plt.subplot(3, 3, 2)
    top_20_labels = label_stats[:20]
    labels = [stat['label'] for stat in top_20_labels]
    counts = [stat['count'] for stat in top_20_labels]
    
    bars = plt.barh(range(len(labels)), counts, color='lightcoral', alpha=0.7)
    plt.ylabel('Label')
    plt.xlabel('Frequency')
    plt.title('Label frequency distribution (top 20)')
    plt.yticks(range(len(labels)), labels)
    
    # Add values on horizontal bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                str(count), ha='left', va='center', fontsize=8)
    
    # Label proportion pie chart (top 8 + Other)
    plt.subplot(3, 3, 3)
    top_8_labels = label_stats[:8]
    pie_labels = [stat['label'] for stat in top_8_labels]
    pie_sizes = [stat['count'] for stat in top_8_labels]
    
    # Compute count for other labels
    other_count = sum(stat['count'] for stat in label_stats[8:])
    if other_count > 0:
        pie_labels.append('Other')
        pie_sizes.append(other_count)
    
    plt.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%', startangle=90)
    plt.title('Label proportion distribution (top 8)')
    
    # Low-frequency label distribution (frequency <= 10)
    plt.subplot(3, 3, 4)
    low_freq_labels = [stat for stat in label_stats if stat['count'] <= 10]
    if low_freq_labels:
        low_labels = [stat['label'] for stat in low_freq_labels]
        low_counts = [stat['count'] for stat in low_freq_labels]
        bars = plt.bar(range(len(low_labels)), low_counts, color='orange', alpha=0.7)
        plt.xlabel('Label')
        plt.ylabel('Frequency')
        plt.title(f'Low-frequency label distribution (freq ≤ 10, total {len(low_freq_labels)})')
        plt.xticks(range(len(low_labels)), low_labels, rotation=45, ha='right', fontsize=8)

        # Add values on all bars
        for bar, count in zip(bars, low_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    str(count), ha='center', va='bottom', fontsize=8)
    else:
        plt.text(0.5, 0.5, 'No low-frequency labels', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Low-frequency label distribution (freq ≤ 10)')
    
    # Distribution of label counts per sample
    plt.subplot(3, 3, 5)
    label_counts = sorted(label_count_distribution.keys())
    sample_counts = [label_count_distribution[lc] for lc in label_counts]
    
    bars = plt.bar(label_counts, sample_counts, color='lightgreen', alpha=0.7)
    plt.xlabel('Labels per sample')
    plt.ylabel('Number of samples')
    plt.title('Labels per sample distribution')
    
    # Add values on bars
    for bar, count in zip(bars, sample_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom')
    
    # Cumulative distribution plot
    plt.subplot(3, 3, 6)
    cumulative_percentages = [stat['cumulative_percentage'] for stat in label_stats]
    plt.plot(range(1, len(cumulative_percentages) + 1), cumulative_percentages, 
             marker='o', markersize=2, linewidth=1.5, color='purple')
    plt.xlabel('Label rank')
    plt.ylabel('Cumulative percentage (%)')
    plt.title('Cumulative label distribution')
    plt.grid(True, alpha=0.3)
    
    # Label frequency log-scale distribution
    plt.subplot(3, 3, 7)
    counts = [stat['count'] for stat in label_stats]
    plt.semilogy(range(1, len(counts) + 1), counts, marker='o', markersize=2, color='brown')
    plt.xlabel('Label rank')
    plt.ylabel('Frequency (log scale)')
    plt.title('Label frequency distribution (log scale)')
    plt.grid(True, alpha=0.3)
    
    # Percentage of samples for each label-count
    plt.subplot(3, 3, 8)
    percentages = [(label_count_distribution[lc] / total_samples) * 100 
                   for lc in label_counts]
    
    plt.bar(label_counts, percentages, color='gold', alpha=0.7)
    plt.xlabel('Labels per sample')
    plt.ylabel('Sample percentage (%)')
    plt.title('Labels per sample percentage distribution')
    
    # Unused label statistics
    plt.subplot(3, 3, 9)
    # Load all defined labels
    label_to_id = load_label_mapping()
    all_defined_labels = set(label_to_id.keys())
    used_labels = set(stat['label'] for stat in label_stats)
    unused_labels = all_defined_labels - used_labels
    
    categories = ['Used', 'Unused']
    counts = [len(used_labels), len(unused_labels)]
    colors = ['lightblue', 'lightcoral']
    
    bars = plt.bar(categories, counts, color=colors, alpha=0.7)
    plt.ylabel('Number of labels')
    plt.title(f'Label usage statistics (total {len(all_defined_labels)})')
    
    # Add values on top of bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(count), ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    
    # Save to train_data directory
    output_path = os.path.join(OUTPUT_DIR, 'label_distribution_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.show()

def save_detailed_report(label_stats, all_samples):
    """Save detailed reports to CSV files."""
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save label statistics
    label_df = pd.DataFrame(label_stats)
    label_report_path = os.path.join(OUTPUT_DIR, 'label_distribution_report.csv')
    label_df.to_csv(label_report_path, index=False, encoding='utf-8-sig')
    
    # Save sample label count statistics
    sample_label_counts = pd.DataFrame([
        {'sample_index': i, 'text_preview': sample['text'][:50] + '...', 
         'label_count': sample['label_count'], 'labels': '|'.join(sample['labels'])}
        for i, sample in enumerate(all_samples)
    ])
    sample_report_path = os.path.join(OUTPUT_DIR, 'sample_label_count_report.csv')
    sample_label_counts.to_csv(sample_report_path, index=False, encoding='utf-8-sig')
    
    # Save detailed label usage analysis
    label_to_id = load_label_mapping()
    all_defined_labels = set(label_to_id.keys())
    used_labels = set(stat['label'] for stat in label_stats)
    unused_labels = all_defined_labels - used_labels
    
    # Create label usage report
    label_usage_report = []
    
    # Add used labels
    for stat in label_stats:
        label_usage_report.append({
            'label_name': stat['label'],
            'status': 'Used',
            'count': stat['count'],
            'percentage': stat['percentage'],
            'rank': label_stats.index(stat) + 1
        })
    
    # Add unused labels
    for label in sorted(unused_labels):
        label_usage_report.append({
            'label_name': label,
            'status': 'Unused',
            'count': 0,
            'percentage': 0.0,
            'rank': None
        })
    
    label_usage_df = pd.DataFrame(label_usage_report)
    usage_report_path = os.path.join(OUTPUT_DIR, 'label_usage_report.csv')
    label_usage_df.to_csv(usage_report_path, index=False, encoding='utf-8-sig')
    
    print(f"Detailed reports saved to {OUTPUT_DIR}:")
    print(f"  - Label distribution report: {label_report_path}")
    print(f"  - Sample label count report: {sample_report_path}")
    print(f"  - Label usage report: {usage_report_path}")

if __name__ == "__main__":
    try:
        label_stats, all_samples = analyze_label_distribution()
        save_detailed_report(label_stats, all_samples)
        print(f"\n=== Analysis complete ===")
        print(f"Visualization and report files have been created. Please check the {OUTPUT_DIR} directory.")
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()
