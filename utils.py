import numpy as np

def print_feature_importance(feature_names, weights, pos_label="POSITIVE", neg_label="NEGATIVE"):
    if len(feature_names) != len(weights):
        print("[ERROR] Length of feature names and weights must match.")
        return

    print("\n" + "="*60)
    print(f"{'FEATURE NAME':<22} | {'OPTIMIZED WEIGHT':<16} | {'INFLUENCE DIRECTION'}")
    print("="*60)

    importance_list = list(zip(feature_names, weights))
    importance_list.sort(key=lambda x: abs(x[1]), reverse=True)

    for name, weight in importance_list:
        if weight > 0:
            direction = f"[+] Pulls toward {pos_label}"
        elif weight < 0:
            direction = f"[-] Pulls toward {neg_label}"
        else:
            direction = "[ ] Neutral"
            
        print(f"{name:<22} | {weight:>15.4f}  | {direction}")

    print("="*60 + "\n")
    
def mae_score(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
    
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)