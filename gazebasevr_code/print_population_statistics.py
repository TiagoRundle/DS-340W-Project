import pandas as pd


if __name__ == "__main__":
    filename = "participant_details.xlsx"

    df = pd.read_excel(filename, keep_default_na=False)

    # Population size and start/end dates of each round
    for round in range(3):
        dates = df.loc[df["Recording round"] == round + 1, "Date"]
        start_date, end_date = dates.min().date(), dates.max().date()
        population = len(dates)
        print(f"Round {round + 1}: {population} participants from {start_date} to {end_date}")
    print()
    
    # Race/ethnicity statistics
    df_round1 = df.loc[df["Recording round"] == 1, :]
    print(df_round1.groupby("Race and ethnicity")["Participant ID"].count())
    print()

    # Gender statistics
    print(df_round1.groupby("Gender")["Participant ID"].count())
    print()

    # Minimum separation between Rounds 1 and 2
    dates_round1 = df_round1[["Participant ID", "Date"]].set_index("Participant ID")
    dates_round2 = df.loc[df["Recording round"] == 2, ["Participant ID", "Date"]].set_index("Participant ID")
    dates_r12 = dates_round1.join(dates_round2, on="Participant ID", how="inner", lsuffix="R1", rsuffix="R2")
    separation = dates_r12["DateR2"] - dates_r12["DateR1"]
    print("Minimum separation between Round 1 and 2:", separation.min().days, "days")