import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


if __name__ == "__main__":
    filename = "participant_details.xlsx"

    df = pd.read_excel(filename)
    df_round1 = df.loc[df["Recording round"] == 1, :]
    df_round2 = df.loc[df["Recording round"] == 2, :]
    df_round3 = df.loc[df["Recording round"] == 3, :]

    # r1_start = datetime.date(2019, 9, 12)
    # r1_end = datetime.date(2020, 3, 6)
    r1_start = df_round1["Date"].min().date()
    r1_end = df_round1["Date"].max().date()

    # r2_start = datetime.date(2020, 1, 30)
    # r2_end = datetime.date(2020, 3, 12)
    r2_start = df_round2["Date"].min().date()
    r2_end = df_round2["Date"].max().date()

    # r3_start = datetime.date(2021, 10, 4)
    # r3_end = datetime.date(2021, 11, 19)
    r3_start = df_round3["Date"].min().date()
    r3_end = df_round3["Date"].max().date()

    cmap = plt.get_cmap("tab10")
    bar_color = cmap(0)

    buffer = 0

    plt.figure(figsize=(7, 2), dpi=300)
    plt.fill_betweenx([2 + buffer, 3 - buffer], [r1_start] * 2, [r1_end] * 2, facecolor=bar_color, edgecolor="k")
    plt.fill_betweenx([1 + buffer, 2 - buffer], [r2_start] * 2, [r2_end] * 2, facecolor=bar_color, edgecolor="k")
    plt.fill_betweenx([0 + buffer, 1 - buffer], [r3_start] * 2, [r3_end] * 2, facecolor=bar_color, edgecolor="k")
    plt.ylim([0 + buffer, 3 - buffer])
    plt.yticks(ticks=[2.5, 1.5, 0.5], labels=["R1", "R2", "R3"])
    # plt.xlim([datetime.date(2019, 9, 1), datetime.date(2021, 12, 1)])
    x0 = r1_start.replace(day=1)
    x1 = r3_end
    if r3_end.day != 1:
        x1 = (r3_end.replace(day=1) + datetime.timedelta(days=32)).replace(day=1)  # get first day of next month
    plt.xlim([x0, x1])
    plt.xticks(rotation=90)
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    # plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.grid(which="major", axis="x")
    plt.tight_layout()
    plt.savefig("overview_dates.png", bbox_inches="tight")