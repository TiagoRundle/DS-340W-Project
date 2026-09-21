from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm
import zipfile


def iqr(x: pd.Series) -> float:
    return float(x.quantile([0.25, 0.75]).diff().tail(1))


if __name__ == "__main__":
    archive_path = "gazebasevr.zip"
    data_dir = Path("./data")
    if not data_dir.exists():
        with zipfile.ZipFile(archive_path) as zip_ref:
            zip_ref.extractall(data_dir)
    
    output_file = Path("durations.csv")
    if not output_file.exists():
        tasks = ("VRG", "PUR", "VID", "TEX", "RAN")
        task_durations = {}
        for task in tasks:
            durations = []
            files_list = list(data_dir.rglob(f"*_{task}.csv"))
            for f in tqdm(sorted(files_list), desc=task):
                df = pd.read_csv(f)
                t = float(df["n"].tail(1))
                durations.append(t)
            task_durations[task] = durations
        pd.DataFrame(task_durations).to_csv(output_file, index=None)
    
    df = pd.read_csv(output_file)
    df = df / 1000.0  # convert milliseconds to seconds
    df_agg = df.agg(["min", "max", "mean", "std", "median", iqr])
    print(df_agg)

    cmap = plt.get_cmap("tab10")
    hatch_dict = {"VRG": "//", "PUR": "//", "VID": "//", "TEX": "\\\\", "RAN": "//"}

    plt.figure(figsize=(5, 3), dpi=300)
    for i, (task, durations) in enumerate(df.items()):
        color = cmap(i)
        plt.hist(durations.to_numpy(), bins=np.arange(0, 180 + 5, 5), histtype="stepfilled", facecolor=color, edgecolor=color, hatch=hatch_dict[task], alpha=0.5, label=task)
    plt.legend(loc="upper right", bbox_to_anchor=(0.85, 0.99))  # fine-tune legend position
    plt.yscale("log")
    plt.xlim([0, 180])
    plt.gca().xaxis.set_major_locator(MultipleLocator(30))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(5))
    plt.xlabel("Task duration (s)")
    plt.ylabel("log(Frequency)")
    plt.tight_layout()
    # plt.show()
    plt.savefig("task_durations.png", bbox_inches="tight")
