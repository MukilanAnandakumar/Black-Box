import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import json

def calculate_l_inf(x, x_adv):
    return torch.max(torch.abs(x - x_adv)).item()

def calculate_l2(x, x_adv):
    return torch.norm(x - x_adv, p=2).item()

def save_research_results(results, filename="results.json"):
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    path = os.path.join(results_dir, filename)
    with open(path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {path}")

def plot_distortion_vs_queries(query_counts, distortions, title="Distortion vs Queries"):
    plt.figure(figsize=(8, 5))
    plt.scatter(query_counts, distortions, alpha=0.5, color='blue')
    plt.xlabel("Query Count")
    plt.ylabel("L2 Distortion")
    plt.title(title)
    plt.grid(True)
    results_dir = "results"
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    plt.savefig(os.path.join(results_dir, "distortion_vs_queries.png"))
    plt.close()

def plot_success_rate_curve(query_counts, max_queries, title="Success Rate Curve"):
    if not query_counts: return
    query_counts = np.sort(query_counts)
    success_rates = np.arange(1, len(query_counts) + 1) / len(query_counts)
    
    plt.figure(figsize=(8, 5))
    plt.plot(query_counts, success_rates, marker='o', linestyle='-', color='green')
    plt.xlim(0, max_queries)
    plt.ylim(0, 1.05)
    plt.xlabel("Query Budget")
    plt.ylabel("Success Rate")
    plt.title(title)
    plt.grid(True)
    results_dir = "results"
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    plt.savefig(os.path.join(results_dir, "success_vs_queries.png"))
    plt.close()

def plot_original_vs_adversarial(original, adversarial, title="Original vs Adversarial"):
    # original/adversarial should be torch tensors [1, 1, 28, 28]
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original.squeeze().cpu().numpy(), cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(adversarial.squeeze().cpu().numpy(), cmap='gray')
    plt.title("Adversarial Image")
    plt.axis('off')
    
    plt.suptitle(title)
    results_dir = "results"
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    plt.savefig(os.path.join(results_dir, "original_vs_adversarial.png"))
    plt.close()

def plot_confidence_drop(conf_drops, title="Confidence Drop Distribution"):
    plt.figure(figsize=(8, 5))
    plt.hist(conf_drops, bins=20, color='red', alpha=0.7)
    plt.xlabel("Confidence Drop")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True)
    results_dir = "results"
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    plt.savefig(os.path.join(results_dir, "confidence_drop_plot.png"))
    plt.close()
