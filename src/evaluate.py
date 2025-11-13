import pandas as pd
import matplotlib.pyplot as plt

def append_metrics(csv_path, row_dict):
    row = pd.DataFrame([row_dict])
    try:
        df = pd.read_csv(csv_path)
        df = pd.concat([df, row], ignore_index=True)
    except FileNotFoundError:
        df = row
    df.to_csv(csv_path, index=False)
    return df

def plot_loss(history, title, out_png=None):
    ep = range(1, len(history["loss"]) + 1)
    plt.figure()
    plt.plot(ep, history["loss"], marker="o", label="Train Loss")
    plt.plot(ep, history["val_loss"], marker="o", label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(title); plt.legend()
    if out_png:
        plt.savefig(out_png, bbox_inches="tight")
    plt.close()
