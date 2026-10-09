from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "merged"
FIGURE_DIR = ROOT / "figures"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

# ---- Update current_year accordingly ----

START_YEAR = 2019
CURRENT_YEAR = 2026

START_MONTH = 8
START_DAY = 1

END_MONTH = 12
END_DAY = 31

# Plot starts at ISO week 30
PLOT_START_WEEK = 30

# Smoothing window for weekly flow
SMOOTH_WINDOW = 3

# Years to label directly on the graph
LABEL_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]

# ============================================================
# 3. LOAD ALL AEA JOE FILES
# ============================================================

def load_joe_files():

    files = sorted(
        list(RAW_DIR.glob("*.xlsx")) +
        list(RAW_DIR.glob("*.xls"))
    )

    if not files:
        raise FileNotFoundError(
            f"No Excel files found in {RAW_DIR}"
        )

    frames = []

    for file in files:

        print(f"Reading: {file.name}")

        df = pd.read_excel(file)

        df["source_file"] = file.name

        frames.append(df)

    combined = pd.concat(
        frames,
        ignore_index=True
    )

    print()
    print(
        f"Raw observations: {len(combined):,}"
    )

    return combined


# ============================================================
# 4. CLEAN DATA
# ============================================================

def clean_joe(df):

    required_columns = [
        "jp_id",
        "Date_Active"
    ]

    for col in required_columns:

        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' "
                f"is missing."
            )

    # Convert posting date
    df["Date_Active"] = pd.to_datetime(
        df["Date_Active"],
        errors="coerce"
    )

    # Remove observations without posting dates
    df = df.dropna(
        subset=["Date_Active"]
    ).copy()

    # Calendar year
    df["year"] = (
        df["Date_Active"]
        .dt
        .year
    )

    # ISO calendar
    iso_calendar = (
        df["Date_Active"]
        .dt
        .isocalendar()
    )

    df["iso_year"] = (
        iso_calendar["year"]
        .astype(int)
    )

    df["iso_week"] = (
        iso_calendar["week"]
        .astype(int)
    )

    # Keep only desired years
    df = df[
        df["year"].between(
            START_YEAR,
            CURRENT_YEAR
        )
    ].copy()

    # Keep August 1 through December 31
    month = df["Date_Active"].dt.month
    day = df["Date_Active"].dt.day

    after_start = (
        (month > START_MONTH) |
        (
            (month == START_MONTH) &
            (day >= START_DAY)
        )
    )

    before_end = (
        (month < END_MONTH) |
        (
            (month == END_MONTH) &
            (day <= END_DAY)
        )
    )

    df = df[
        after_start & before_end
    ].copy()

    # Deduplicate listings
    df = (
        df
        .sort_values("Date_Active")
        .drop_duplicates(
            subset=["jp_id"],
            keep="last"
        )
    )

    print(
        f"Unique Aug-Dec listings: "
        f"{len(df):,}"
    )

    return df


# ============================================================
# 5. BUILD WEEKLY DATA
# ============================================================

def build_weekly(df):

    weekly = (
        df
        .groupby(
            ["year", "iso_week"]
        )
        ["jp_id"]
        .nunique()
        .reset_index(
            name="new_listings"
        )
    )

    weekly = weekly.sort_values(
        ["year", "iso_week"]
    )

    # Cumulative listings
    weekly["cumulative_listings"] = (
        weekly
        .groupby("year")[
            "new_listings"
        ]
        .cumsum()
    )

    # 3-week centered moving average
    weekly["smoothed_listings"] = (
        weekly
        .groupby("year")["new_listings"]
        .transform(
            lambda x: x.rolling(
                window=SMOOTH_WINDOW,
                center=True,
                min_periods=1
            ).mean()
        )
    )

    return weekly


# ============================================================
# 6. WEEKLY FLOW PLOT
# ============================================================
def plot_weekly(weekly):

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    # Start plot at ISO week 30
    plot_data = weekly[
        weekly["iso_week"] >= PLOT_START_WEEK
    ].copy()

    for year, group in plot_data.groupby("year"):

        group = group.sort_values("iso_week")

        if year == CURRENT_YEAR:

            line, = ax.plot(
                group["iso_week"],
                group["smoothed_listings"],
                linewidth=2.8
            )

        else:

            line, = ax.plot(
                group["iso_week"],
                group["smoothed_listings"],
                linewidth=1.5,
                alpha=0.65
            )

        # Label each line with its year
        if year in LABEL_YEARS:

            last = group.iloc[-1]

            ax.text(
                last["iso_week"] + 0.25,
                last["smoothed_listings"],
                str(year),
                color=line.get_color(),
                fontsize=9,
                va="center"
            )

    ax.set_xlim(
        PLOT_START_WEEK,
        53
    )

    ax.set_xticks(
        range(PLOT_START_WEEK, 54, 2)
    )

    ax.set_xlabel(
        "ISO week of year"
    )

    ax.set_ylabel(
        "New JOE listings"
    )

    ax.set_title(
        f"AEA JOE Listings by Week "
        f"({SMOOTH_WINDOW}-Week Moving Average)"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "joe_weekly_smoothed.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

# ============================================================
# 7. CUMULATIVE PLOT
# ============================================================

def plot_cumulative(weekly):

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    plot_data = weekly[
        weekly["iso_week"] >= PLOT_START_WEEK
    ].copy()

    for year, group in plot_data.groupby("year"):

        group = group.sort_values("iso_week")

        if year == CURRENT_YEAR:

            line, = ax.plot(
                group["iso_week"],
                group["cumulative_listings"],
                linewidth=2.8
            )

        else:

            line, = ax.plot(
                group["iso_week"],
                group["cumulative_listings"],
                linewidth=1.5,
                alpha=0.65
            )

        # Label year at end of line
        if year in LABEL_YEARS:

            last = group.iloc[-1]

            ax.text(
                last["iso_week"] + 0.25,
                last["cumulative_listings"],
                str(year),
                color=line.get_color(),
                fontsize=9,
                va="center"
            )

    ax.set_xlim(
        PLOT_START_WEEK,
        53
    )

    ax.set_xticks(
        range(PLOT_START_WEEK, 54, 2)
    )

    ax.set_xlabel(
        "ISO week of year"
    )

    ax.set_ylabel(
        "Cumulative JOE listings"
    )

    ax.set_title(
        f"Cumulative AEA JOE Listings: "
        f"{CURRENT_YEAR} vs. Previous Years"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "joe_cumulative.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

# ============================================================
# 8. CURRENT-YEAR COMPARISON
# ============================================================

def print_current_comparison(weekly):

    current = weekly[
        weekly["year"] == CURRENT_YEAR
    ]

    if current.empty:
        print(
            f"No observations for {CURRENT_YEAR}."
        )
        return

    latest_week = (
        current["iso_week"].max()
    )

    print()
    print("=" * 60)
    print(
        f"JOE MARKET THROUGH ISO WEEK "
        f"{latest_week}"
    )
    print("=" * 60)

    comparison = []

    for year in sorted(
        weekly["year"].unique()
    ):

        temp = weekly[
            (weekly["year"] == year) &
            (weekly["iso_week"] <= latest_week)
        ]

        total = temp[
            "new_listings"
        ].sum()

        comparison.append({
            "year": year,
            "listings": total
        })

    comparison = pd.DataFrame(
        comparison
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    historical = comparison[
        comparison["year"] < CURRENT_YEAR
    ]

    current_total = comparison.loc[
        comparison["year"] == CURRENT_YEAR,
        "listings"
    ].iloc[0]

    historical_mean = (
        historical["listings"].mean()
    )

    pct_difference = (
        100 *
        (
            current_total /
            historical_mean - 1
        )
    )

    print()
    print(
        f"{CURRENT_YEAR}: "
        f"{current_total:,} listings"
    )

    print(
        f"{START_YEAR}-{CURRENT_YEAR-1} "
        f"average through same week: "
        f"{historical_mean:,.1f}"
    )

    print(
        f"Difference from historical average: "
        f"{pct_difference:+.1f}%"
    )


# ============================================================
# 9. MAIN
# ============================================================


def main():

    print("=" * 60)
    print("AEA JOE JOB MARKET UPDATE")
    print("=" * 60)
    print()

    # Load
    df = load_joe_files()

    # Clean
    df = clean_joe(df)

    # Save master
    df.to_csv(
        PROCESSED_DIR / "joe_master.csv",
        index=False
    )

    # Weekly panel
    weekly = build_weekly(df)

    weekly.to_csv(
        PROCESSED_DIR / "joe_weekly.csv",
        index=False
    )

    # Figures
    plot_weekly(weekly)
    plot_cumulative(weekly)

    # Current status
    print_current_comparison(
        weekly
    )

    print()
    print("=" * 60)
    print("UPDATE COMPLETE")
    print("=" * 60)

    print(
        "Saved:"
    )

    print(
        "  data/processed/joe_master.csv"
    )

    print(
        "  data/processed/joe_weekly.csv"
    )
    
    print(
        "  figures/joe_weekly_smoothed.png"
    )
    
    print(
        "  figures/joe_cumulative.png"
    )

if __name__ == "__main__":
    main()