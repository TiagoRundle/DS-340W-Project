import pandas as pd
import datetime


def is_0_or_1_or_empty(df: pd.DataFrame, col: str) -> bool:
    return df[col].isin([0, 1, ""]).all()


def is_between_or_empty(df: pd.DataFrame, col: str, low: int, high: int, inclusive: str = "both") -> bool:
    series = df[col]
    empty_rows = series.eq("").index
    return series.drop(index=empty_rows).between(low, high, inclusive=inclusive).all()


if __name__ == "__main__":
    filename = "participant_details.xlsx"

    df = pd.read_excel(filename, keep_default_na=False)

    assert df["Recording round"].isin([1, 2, 3]).all()

    assert df["Participant ID"].between(1, 465, inclusive="both").all()
    round1_participants = df.loc[df["Recording round"] == 1, "Participant ID"]
    round2_participants = df.loc[df["Recording round"] == 2, "Participant ID"]
    round3_participants = df.loc[df["Recording round"] == 3, "Participant ID"]
    assert not round1_participants.duplicated().any()  # only one entry per participant per round
    assert not round2_participants.duplicated().any()
    assert not round3_participants.duplicated().any()
    assert round2_participants.isin(round1_participants).all()  # Round 2 participants are in Round 1
    assert round3_participants.isin(round1_participants).all()  # Round 3 participants are in Round 1
    # Rounds 2 and 3 both recruited from the Round 1 population; a participant in Round 3 may not have been in Round 2, and a participant in Round 2 may not be in Round 3

    assert df["Dominant eye"].isin(["Left", "Right", "Neither", ""]).all()
    assert df["Age"].ge(18).all()
    assert df["Gender"].isin(["Male", "Female", "Other", "Prefer not to answer"]).all()
    assert df["Race and ethnicity"].isin(["American Indian or Alaska Native", "Asian", "Black or African American", "Hispanic or Latino", "Native Hawaiian or Other Pacific Islander", "White", "Mixed", "Prefer not to answer"]).all()
    
    assert is_0_or_1_or_empty(df, "Head injury in past 12 months")

    assert is_0_or_1_or_empty(df, "Wear corrective lenses")
    assert df["Lens type"].isin(["Glasses", "Contacts", ""]).all()
    assert df.loc[df["Wear corrective lenses"] == 0, "Lens type"].eq("").all()  # no lens type if no corrective lenses

    assert all([x == "" or isinstance(x, (int, float)) for x in df["Hours slept last night"].unique()])
    for value in ("Caffeine", "Medicine to aid sleep", "Headache", "Alcohol"):
        assert is_0_or_1_or_empty(df, f"{value} in past 24 hours"), value
    assert is_0_or_1_or_empty(df, "Previously involved in an eye tracking study")

    for value in ("before", "between", "after"):
        assert is_between_or_empty(df, f"Physical comfort {value} sessions", 1, 7), value
        assert is_between_or_empty(df, f"Shoulder fatigue {value} sessions", 1, 5), value
        assert is_between_or_empty(df, f"Neck fatigue {value} sessions", 1, 5), value
        assert is_between_or_empty(df, f"Eye fatigue {value} sessions", 1, 5), value
        if value != "before":
            assert is_between_or_empty(df, f"Physical effort exerted {value} sessions", 1, 5), value
            assert is_between_or_empty(df, f"Mental effort exerted {value} sessions", 1, 5), value
        assert is_between_or_empty(df, f"Sleepiness {value} sessions", 1, 7), value
    
    for value in ("coffee", "tea", "cola"):
        assert is_0_or_1_or_empty(df, f"Drink {value}"), value
    assert is_0_or_1_or_empty(df, "Drink alcohol")
    assert is_0_or_1_or_empty(df, "Currently use recreational or street drugs")

    assert is_0_or_1_or_empty(df, "Glaucoma/ambylopia/eye injury/blindess/strabismus/nystagmus/dry eyes/retinitis pigmentosa/macular degeneration")
    assert is_0_or_1_or_empty(df, "Require vision correction")
    assert is_0_or_1_or_empty(df, "Had corrective vision surgery")
    assert is_0_or_1_or_empty(df, "Had a detached retina")
    assert is_0_or_1_or_empty(df, "Have a prosthetic eye")

    assert is_0_or_1_or_empty(df, "Epilepsy/meningitis/multiple sclerosis/motor neuron disease/brain cancer/tourette syndrome/cerebral palsy/dementia/stroke/parkinsonism")
    assert is_0_or_1_or_empty(df, "Suffer from headaches")
    assert is_0_or_1_or_empty(df, "Had a head injury associated with unconsciousness or memory loss")
    assert is_0_or_1_or_empty(df, "Major depression/anxiety or panic disorder/schizophrenia/bipolar disorder/autism spectrum disorder/asperger/ADHD/ADD/OCD")
    assert is_0_or_1_or_empty(df, "Currently taking any medication")

    assert df["Recording administrator ID"].isin(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]).all()

    assert df.loc[df["Recording round"] != 3, "Had COVID-19"].eq("").all()  # no COVID-19 in Rounds 1 and 2
    assert df.loc[df["Recording round"] != 3, "COVID-19 date"].eq("").all()  # no COVID-19 date in Rounds 1 and 2
    assert is_0_or_1_or_empty(df, "Had COVID-19")
    assert df.loc[df["Had COVID-19"].isin([0, ""]), "COVID-19 date"].eq("").all()  # no COVID-19 date if no COVID-19