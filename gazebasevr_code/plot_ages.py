import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


if __name__ == "__main__":
    filename = "participant_details.xlsx"

    df = pd.read_excel(filename)
    df_round1 = df.loc[df["Recording round"] == 1, :]
    ages = df_round1["Age"].to_numpy()

    min_age = min(ages)
    max_age = max(ages)
    print(min_age, max_age)

    plt.figure(figsize=(5,3), dpi=300)
    hist_values, *_ = plt.hist(ages, bins=np.arange(min_age - 0.5, max_age + 1.5), histtype="bar", edgecolor="k", linewidth=1)
    max_freq = max(hist_values)
    round_up_max = int((max_freq + 5 - 1) / 5) * 5  # rounded up to the nearest multiple of 5
    plt.xlim([min_age - 0.5, max_age + 0.5])
    plt.ylim([0, round_up_max])
    plt.gca().xaxis.set_major_locator(MultipleLocator(5))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(10))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(2))
    plt.grid(which="major", axis="y")
    plt.xlabel("Age (years)")
    plt.ylabel(f"Number of participants (N = {len(ages)})")
    plt.tight_layout()
    # plt.show()
    plt.savefig("ages.png", bbox_inches="tight")