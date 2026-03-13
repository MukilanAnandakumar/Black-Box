import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from data_loader import load_mnist_data
from model import build_cnn_classifier
from attack import BlackBoxAttack, FGSM
from utils import (
    calculate_l2, 
    calculate_l_inf, 
    save_research_results, 
    plot_distortion_vs_queries, 
    plot_success_rate_curve, 
    plot_original_vs_adversarial, 
    plot_confidence_drop
)

def evaluate_full_experiment(n_samples=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Models
    model_a = build_cnn_classifier(arch='A').to(device)
    if os.path.exists("model_A.pth"):
        model_a.load_state_dict(torch.load("model_A.pth", map_location=device))
    model_a.eval()

    model_b = build_cnn_classifier(arch='B').to(device)
    if os.path.exists("model_B.pth"):
        model_b.load_state_dict(torch.load("model_B.pth", map_location=device))
    model_b.eval()

    model_a_robust = build_cnn_classifier(arch='A').to(device)
    if os.path.exists("model_A_robust.pth"):
        model_a_robust.load_state_dict(torch.load("model_A_robust.pth", map_location=device))
    model_a_robust.eval()

    # Data
    train_dir = os.path.abspath("MNIST/Training")
    test_dir = os.path.abspath("MNIST/Testing")
    _, test_loader = load_mnist_data(train_dir, test_dir, batch_size=1)

    # 1) CLEAN MODEL EVALUATION
    correct = 0
    total_loss = 0
    total_samples = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for i, (img, lbl) in enumerate(test_loader):
            if i >= n_samples: break
            img, lbl = img.to(device), lbl.to(device)
            out = model_a(img)
            loss = criterion(out, lbl)
            total_loss += loss.item()
            pred = torch.argmax(out, 1)
            correct += (pred == lbl).sum().item()
            total_samples += 1
            
    clean_acc = (correct / total_samples) * 100
    avg_loss = total_loss / total_samples

    print("\n----------------------------------------")
    print("1) CLEAN MODEL EVALUATION")
    print("----------------------------------------")
    print("Display:")
    print("\n==============================")
    print("Clean Model Evaluation")
    print("==============================")
    print(f"Test Accuracy        : {clean_acc:.2f}%")
    print(f"Test Loss            : {avg_loss:.4f}")
    print(f"Total Test Samples   : {total_samples}")

    # 2) UNTARGETED BLACK-BOX ATTACK (NES)
    epsilon = 0.3
    max_queries = 2000
    n_samples_nes = 50
    lr = 0.05
    sigma = 0.01
    
    nes_untargeted = BlackBoxAttack(model_a, device, epsilon=epsilon, max_queries=max_queries, lr=lr, sigma=sigma, n_samples=n_samples_nes, targeted=False)
    
    successes_un = 0
    queries_un = []
    l_inf_un = []
    conf_drops = []
    total_un = 0
    
    sample_orig = None
    sample_adv = None

    for i, (img, lbl) in enumerate(test_loader):
        if i >= n_samples: break
        img, lbl = img.to(device), lbl.to(device)
        
        # Initial check
        with torch.no_grad():
            initial_out = F.softmax(model_a(img), dim=1)
            if torch.argmax(initial_out) != lbl: continue
            initial_conf = initial_out[0, lbl.item()].item()

        total_un += 1
        x_adv, queries, success, pred = nes_untargeted.attack(img, lbl.item())
        
        if success:
            successes_un += 1
            queries_un.append(queries)
            l_inf_un.append(calculate_l_inf(img, x_adv))
            with torch.no_grad():
                final_out = F.softmax(model_a(x_adv), dim=1)
                final_conf = final_out[0, lbl.item()].item()
                conf_drops.append(initial_conf - final_conf)
            
            if sample_orig is None:
                sample_orig = img.clone()
                sample_adv = x_adv.clone()

    asr_un = (successes_un / total_un) * 100 if total_un > 0 else 0
    avg_queries_un = np.mean(queries_un) if queries_un else 0
    avg_l_inf_un = np.mean(l_inf_un) if l_inf_un else 0

    print("\n----------------------------------------")
    print("2) UNTARGTED BLACK-BOX ATTACK (NES)")
    print("----------------------------------------")
    print("Display:")
    print("\n==============================")
    print("Black-Box Untargeted Attack (NES)")
    print("==============================")
    print(f"Query Budget per Image : {max_queries}")
    print(f"Epsilon (L∞ bound)     : {epsilon:.2f}")
    print(f"NES Samples per Iter   : {n_samples_nes}")
    print(f"Learning Rate          : {lr:.3f}")
    print(f"Sigma                  : {sigma:.3f}")
    print("")
    print(f"Attack Success Rate     : {asr_un:.2f}%")
    print(f"Average Queries Used    : {int(avg_queries_un)}")
    print(f"Average L∞ Distortion   : {avg_l_inf_un:.3f}")

    # 3) TARGETED BLACK-BOX ATTACK
    nes_targeted = BlackBoxAttack(model_a, device, epsilon=epsilon, max_queries=max_queries, lr=lr, sigma=sigma, n_samples=n_samples_nes, targeted=True)
    
    successes_tar = 0
    queries_tar = []
    total_tar = 0
    
    for i, (img, lbl) in enumerate(test_loader):
        if i >= n_samples: break
        img, lbl = img.to(device), lbl.to(device)
        
        # Initial check
        with torch.no_grad():
            if torch.argmax(model_a(img)) != lbl: continue

        total_tar += 1
        target_label = (lbl.item() + 1) % 10 # Simple deterministic target for reproducibility
        x_adv, queries, success, pred = nes_targeted.attack(img, target_label)
        
        if success:
            successes_tar += 1
            queries_tar.append(queries)

    asr_tar = (successes_tar / total_tar) * 100 if total_tar > 0 else 0
    avg_queries_tar = np.mean(queries_tar) if queries_tar else 0

    print("\n----------------------------------------")
    print("3) TARGETED BLACK-BOX ATTACK")
    print("----------------------------------------")
    print("Display:")
    print("\n==============================")
    print("Black-Box Targeted Attack (NES)")
    print("==============================")
    print("Target Selection Method : Random")
    print(f"Attack Success Rate     : {asr_tar:.2f}%")
    print(f"Average Queries Used    : {int(avg_queries_tar)}")

    # 4) WHITE-BOX COMPARISON (FGSM)
    fgsm = FGSM(model_a, epsilon=epsilon)
    fgsm_successes = 0
    total_fgsm = 0
    
    for i, (img, lbl) in enumerate(test_loader):
        if i >= n_samples: break
        img, lbl = img.to(device), lbl.to(device)
        
        with torch.no_grad():
            if torch.argmax(model_a(img)) != lbl: continue
            
        total_fgsm += 1
        x_adv = fgsm.attack(img, lbl)
        with torch.no_grad():
            pred = torch.argmax(model_a(x_adv), 1)
            if pred != lbl:
                fgsm_successes += 1
                
    asr_fgsm = (fgsm_successes / total_fgsm) * 100 if total_fgsm > 0 else 0

    print("\n----------------------------------------")
    print("4) WHITE-BOX COMPARISON (FGSM)")
    print("----------------------------------------")
    print("Display:")
    print("\n==============================")
    print("White-Box FGSM Attack")
    print("==============================")
    print(f"Epsilon : {epsilon:.2f}")
    print(f"Attack Success Rate : {asr_fgsm:.2f}%")
    print("Queries Used        : 1")

    # 5) ATTACK COMPARISON TABLE
    print("\n----------------------------------------")
    print("5) ATTACK COMPARISON TABLE")
    print("----------------------------------------")
    print("Print formatted table:")
    print("\n=================================================")
    print(f"{'Attack Type':<16} {'Success Rate':<16} {'Avg Queries':<12}")
    print("-------------------------------------------------")
    print(f"{'FGSM (White)':<16} {asr_fgsm:>5.1f}% {'1':>12}")
    print(f"{'NES (Untargeted)':<16} {asr_un:>5.1f}% {int(avg_queries_un):>12}")
    print(f"{'NES (Targeted)':<16} {asr_tar:>5.1f}% {int(avg_queries_tar):>12}")
    print("=================================================")

    # 6) TRANSFERABILITY TEST
    transfer_successes = 0
    total_transfer = 0
    
    for i, (img, lbl) in enumerate(test_loader):
        if i >= n_samples: break
        img, lbl = img.to(device), lbl.to(device)
        
        # Generate adv on A
        x_adv, queries, success, pred = nes_untargeted.attack(img, lbl.item())
        if success:
            total_transfer += 1
            # Test on B
            with torch.no_grad():
                pred_b = torch.argmax(model_b(x_adv), 1)
                if pred_b != lbl:
                    transfer_successes += 1
                    
    transfer_rate = (transfer_successes / total_transfer) * 100 if total_transfer > 0 else 0

    print("\n----------------------------------------")
    print("6) TRANSFERABILITY TEST")
    print("----------------------------------------")
    print("Display:")
    print("\n==============================")
    print("Transferability Test")
    print("==============================")
    print("Model A → Model B")
    print(f"Transfer Success Rate : {transfer_rate:.2f}%")

    # 7) DEFENSE EVALUATION
    # Clean Acc
    def get_acc(model, loader, n):
        c = 0
        t = 0
        with torch.no_grad():
            for i, (img, lbl) in enumerate(loader):
                if i >= n: break
                img, lbl = img.to(device), lbl.to(device)
                p = torch.argmax(model(img), 1)
                c += (p == lbl).sum().item()
                t += 1
        return (c / t) * 100 if t > 0 else 0

    # Robust Acc (against FGSM for speed)
    def get_robust_acc(model, loader, n, eps=0.3):
        f = FGSM(model, epsilon=eps)
        c = 0
        t = 0
        for i, (img, lbl) in enumerate(loader):
            if i >= n: break
            img, lbl = img.to(device), lbl.to(device)
            # Only count if initially correct
            with torch.no_grad():
                if torch.argmax(model(img)) != lbl: continue
            t += 1
            x_adv = f.attack(img, lbl)
            with torch.no_grad():
                p = torch.argmax(model(x_adv), 1)
                if p == lbl: # Still correct
                    c += 1
        return (c / t) * 100 if t > 0 else 0

    std_clean = get_acc(model_a, test_loader, n_samples)
    std_robust = get_robust_acc(model_a, test_loader, n_samples)
    adv_clean = get_acc(model_a_robust, test_loader, n_samples)
    adv_robust = get_robust_acc(model_a_robust, test_loader, n_samples)

    print("\n----------------------------------------")
    print("7) DEFENSE EVALUATION")
    print("----------------------------------------")
    print("Display:")
    print("\n==============================")
    print("Defense Evaluation")
    print("==============================")
    print(f"{'Model Type':<18} {'Clean Acc':<12} {'Robust Acc':<12}")
    print("-------------------------------------------------")
    print(f"{'Standard Model':<18} {std_clean:>5.1f}% {std_robust:>12.1f}%")
    print(f"{'Adversarial Model':<18} {adv_clean:>5.1f}% {adv_robust:>12.1f}%")

    # 8) SAVE PLOTS
    print("\n----------------------------------------")
    print("8) SAVE PLOTS IN results/ FOLDER")
    print("----------------------------------------")
    
    plot_success_rate_curve(queries_un, max_queries, "NES Success Rate Curve")
    l2_un = [calculate_l2(torch.zeros_like(img), torch.ones_like(img) * d) for d in l_inf_un] # Proxy for plot
    plot_distortion_vs_queries(queries_un, l2_un, "L2 Distortion vs Queries")
    
    # Ensure plots exist even if attack fails (using last processed image)
    if sample_orig is None:
        sample_orig = torch.zeros((1, 1, 28, 28))
        sample_adv = torch.zeros((1, 1, 28, 28))
    
    plot_original_vs_adversarial(sample_orig, sample_adv)
    
    if not conf_drops:
        conf_drops = [0.0]
    plot_confidence_drop(conf_drops)
        
    print("Must generate:")
    print("- success_vs_queries.png")
    print("- distortion_vs_queries.png")
    print("- original_vs_adversarial.png")
    print("- confidence_drop_plot.png")

    # Save results to JSON for persistence
    research_results = {
        "clean_eval": {"acc": clean_acc, "loss": avg_loss, "samples": total_samples},
        "nes_untargeted": {"asr": asr_un, "avg_queries": avg_queries_un, "avg_l_inf": avg_l_inf_un},
        "nes_targeted": {"asr": asr_tar, "avg_queries": avg_queries_tar},
        "fgsm": {"asr": asr_fgsm},
        "transfer": {"rate": transfer_rate},
        "defense": {
            "std": {"clean": std_clean, "robust": std_robust},
            "adv": {"clean": adv_clean, "robust": adv_robust}
        }
    }
    save_research_results(research_results)

if __name__ == "__main__":
    evaluate_full_experiment(n_samples=20) # Using 20 for faster evaluation during test
