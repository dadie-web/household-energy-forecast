from pathlib import Path
import pandas as pd
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    PageBreak,
    KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet

ROOT = Path(__file__).resolve().parent


def report(metrics, q, start):
    """Generate Markdown, PDF and Word reports from the measured experiment."""
    if all(
        (ROOT / f"results/monthly_backtest/{month}_metrics.csv").exists()
        for month in ["april", "july", "october"]
    ):
        from monthly_reporting import build_report

        build_report()
        return
    val = pd.read_csv(ROOT / "results/validation_metrics.csv")
    test = metrics.set_index("method")
    h = pd.read_csv(
        ROOT / "data/processed/hourly_load_2009.csv", parse_dates=["timestamp"]
    ).set_index("timestamp")
    monthly = h.load.resample("MS").mean()
    r = pd.read_csv(
        ROOT / "results/forecast_results.csv", parse_dates=["timestamp"]
    ).set_index("timestamp")
    window = r.loc[start : start + pd.Timedelta(hours=167)]
    peak = window.actual.idxmax()
    title = "Machine Learning-Based Household Electricity Load Forecasting: A Case Study Using UCI Data"
    improvement = []
    for baseline in ["yesterday", "weekly"]:
        improvement.append(
            [baseline]
            + [
                100
                * (test.loc[baseline, c] - test.loc["AI", c])
                / test.loc[baseline, c]
                for c in ["MAE_kW", "RMSE_kW"]
            ]
        )
    pages = [
        [
            (
                "Abstract",
                f'Accurate day-ahead load forecasting is challenging for individual households because demand depends on irregular appliance use. This study uses HistGradientBoostingRegressor (HGB) with hourly UCI household electricity data from 2009. The model combines calendar variables and historical loads and is compared with daily and weekly persistence. On the November-December test set of 1464 hours, HGB achieves MAE {test.loc["AI", "MAE_kW"]:.6f} kW and RMSE {test.loc["AI", "RMSE_kW"]:.6f} kW, outperforming both baselines. The results suggest that these features improve average forecast accuracy, although sharp demand peaks remain difficult to predict.',
            ),
            (
                "1. Introduction",
                "Household electricity forecasts help characterize demand patterns and anticipate future consumption. Unlike aggregate demand, a single household can exhibit abrupt changes when appliances are switched on or routines change. Daily and weekly persistence provide simple benchmarks for recurring behavior. This study examines whether a gradient boosting model using calendar and recent-load features improves these benchmarks for a fixed 24-hour forecasting horizon.",
            ),
            (
                "2. Dataset and Preprocessing",
                f'The UCI dataset [1] contains {q["raw_rows"]} minute records from one household. This study retains January 1-December 31, 2009 ({q["rows_2009"]} records). Global_active_power measures minute-average active power in kW. The semicolon-separated file is parsed using dd/mm/yyyy and hh:mm:ss. Timestamps are kept as recorded in the source file.',
            ),
            (
                "2.1 Data Quality",
                f'Missing, nonnumeric and negative power values are treated as missing; unparseable timestamps are removed and duplicate timestamps are averaged. The original file contains {q["unparseable_timestamp"]} unparseable timestamps and {q["negative_power_all_years"]} negative values. In 2009, there are {q["duplicates_2009"]} duplicate timestamps and {q["invalid_power_2009"]} missing or invalid power values. Hourly load is the arithmetic mean of valid minute readings, with at least 54 valid minutes required. All 8760 hourly timestamps are retained, including {q["invalid_hours"]} insufficiently observed hours and {q["valid_hours"]} valid hours. Targets are not interpolated.',
            ),
            (
                "2.2 Sample Availability",
                f'Observation gaps and missing historical features affect sample eligibility differently. Of 5832 training hours, {q["train_invalid_target"]} have invalid targets and another {q["train_history_only_excluded"]} valid-target hours lack complete history, leaving {5832-q["train_excluded"]} training samples. The first seven days lack pre-2009 history, and subsequent gaps affect rolling features. Validation and test each retain all 1464 hours.',
            ),
        ],
        [
            (
                "3. Forecasting Methodology",
                "At midnight, the model predicts the following 24 hours using eight features: target hour, weekday, weekend indicator, month, previous-day and previous-week same-hour load, and mean load over the preceding 24 and 168 hours. Rolling means end before midnight and remain constant across the target day. All historical inputs precede the forecast origin; observations from earlier test days become available for later origins. Samples with incomplete history are excluded consistently across methods.",
            ),
            (
                "3.1 Experimental Design",
                "The initial model is fitted on January-August and evaluated on September-October. A single preset configuration is used without parameter search. The final model is refitted on January-October and evaluated on November-December. Test outcomes do not affect parameters or timestamp selection. Yesterday Naive and Weekly Naive copy the corresponding hour one day and one week earlier, respectively.",
            ),
            (
                "3.2 Model and Metrics",
                "HistGradientBoostingRegressor [2] uses 150 iterations, 15 maximum leaves, learning rate 0.05, L2 regularization 1.0, seed 42 and disabled early stopping. Predictions are clipped to nonnegative power. MAE measures average absolute error; RMSE gives greater weight to large errors. Both are expressed in kW and calculated on identical valid timestamps for all methods.",
            ),
            (
                "4. Experimental Results",
                "On the September-October validation set, HGB has lower MAE and RMSE than both persistence baselines. Weekly Naive also improves on Yesterday Naive, suggesting that weekly routines provide useful predictive information.",
            ),
            ("VALIDATION_TABLE", ""),
            (
                "4.1 Test Performance",
                "HGB again achieves the lowest errors across the full November-December test period. Its lower RMSE also shows an overall reduction in larger errors. Validation and test scores reflect different seasons and fitted models and therefore do not isolate the effect of training-set size.",
            ),
            ("TEST_TABLE", ""),
            ("IMPROVEMENT_TABLE", ""),
        ],
        [
            (
                "5. Load Patterns and Forecast Behavior",
                f'Monthly observed-hour mean power is {monthly.iloc[0]:.3f} kW in January, {monthly.loc["2009-08-01"]:.3f} kW in August and {monthly.iloc[-1]:.3f} kW in December. Figure 1 shows lower summer consumption and higher demand near the beginning and end of the year, alongside substantial daily variation. Monthly statistics use available valid hours; the daily figure requires all 24 hours to be valid.',
            ),
            ("FIGURE1", ""),
            (
                "5.1 Peak-load Prediction",
                f'Figure 2 illustrates November 1-7, the earliest complete seven-day November interval, selected by observation availability. These 168 hours are a display subset of the 1464-hour test set. At {peak.strftime("%Y-%m-%d %H:%M")}, the displayed maximum actual load is {window.loc[peak, "actual"]:.3f} kW, compared with HGB {window.loc[peak, "AI"]:.3f} kW. The HGB curve is generally smoother than the observations and underestimates several sharp excursions. Persistence forecasts also mis-time peaks when appliance activity differs from earlier days.',
            ),
            ("FIGURE2", ""),
            (
                "6. Discussion and Conclusion",
                "The HGB forecasts follow the broad daily pattern but are smoother than the observed loads. Sudden peaks, such as the 4.831 kW peak shown in Figure 2, can be much higher than the forecast. One possible explanation is that past load and calendar features do not show whether an appliance will be switched on at a particular hour.\n\nComplete-history filtering reduces training coverage and may favor periods with fewer missing readings. Some accepted hourly means use only 54-59 valid minute readings, so they may differ from the mean of all 60 minutes. The study uses one household and one year; confidence intervals and performance across multiple years are not evaluated.\n\nOn the full test set, HGB reduces MAE and RMSE by 27.57% and 27.15% relative to Yesterday Naive, and by 18.58% and 20.48% relative to Weekly Naive. These improvements in overall error occur alongside substantial peak underestimation. Results from this single household cannot establish how well the model would perform for other households.",
            ),
            (
                "References",
                '[1] G. Hebrail and A. Berard, "Individual Household Electric Power Consumption," UCI Machine Learning Repository, 2006, doi: 10.24432/C58K54. [Online]. Available: https://archive.ics.uci.edu/dataset/235/individualhouseholdelectricpowerconsumption. Accessed: Oct. 13, 2026. Data license: CC BY 4.0.\n\n[2] Scikit-learn developers, "HistGradientBoostingRegressor," scikit-learn documentation. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html. Accessed: Oct. 13, 2026.',
            ),
        ],
    ]
    styles = getSampleStyleSheet()
    styles["BodyText"].fontSize = 9.5
    styles["BodyText"].leading = 11
    styles["Heading1"].fontSize = 12
    styles["Heading2"].fontSize = 11
    styles["Heading2"].spaceBefore = 7
    story = []
    md = ["# " + title]
    for pi, page in enumerate(pages):
        if pi:
            story.append(PageBreak())
        else:
            story.append(Paragraph(title, styles["Title"]))
        for heading, body in page:
            if heading.endswith("_TABLE"):
                if heading == "IMPROVEMENT_TABLE":
                    rows = [
                        ["Relative error reduction vs baseline", "MAE (%)", "RMSE (%)"]
                    ] + [
                        [
                            {"yesterday": "Yesterday Naive", "weekly": "Weekly Naive"}[
                                a
                            ],
                            f"{b:.2f}",
                            f"{c:.2f}",
                        ]
                        for a, b, c in improvement
                    ]
                    caption = "Table 3. Test error reduction relative to each baseline (%). Reduction = 100 x (baseline error - HGB error) / baseline error."
                else:
                    df = val if heading == "VALIDATION_TABLE" else metrics
                    names = {
                        "yesterday": "Yesterday Naive",
                        "weekly": "Weekly Naive",
                        "AI": "HGB",
                    }
                    rows = [["Method", "MAE (kW)", "RMSE (kW)", "Hours"]] + [
                        [names[a], f"{b:.6f}", f"{c:.6f}", str(int(d))]
                        for a, b, c, d in df.values
                    ]
                    caption = (
                        "Table 1. Validation errors, September-October 2009."
                        if heading == "VALIDATION_TABLE"
                        else "Table 2. Test errors, November-December 2009 (1464 hours)."
                    )
                story.append(
                    KeepTogether(
                        [
                            Paragraph(caption, styles["BodyText"]),
                            Table(
                                rows,
                                style=[
                                    ("GRID", (0, 0), (-1, -1), 0.3, "grey"),
                                    ("BACKGROUND", (0, 0), (-1, 0), "#e6edf4"),
                                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                                ],
                            ),
                            Spacer(1, 8),
                        ]
                    )
                )
                markdown_rows = [rows[0], ["---"] * len(rows[0]), *rows[1:]]
                md.extend(
                    [
                        "\n" + caption,
                        "\n"
                        + "\n".join(
                            "| " + " | ".join(row) + " |" for row in markdown_rows
                        ),
                    ]
                )
                continue
            if heading.startswith("FIGURE"):
                n = int(heading[-1])
                fn = next((ROOT / "figures").glob(f"figure{n}_*.png"))
                cap = (
                    "Figure 1. Daily mean power in 2009 (kW); days lacking 24 valid hours remain gaps."
                    if n == 1
                    else "Figure 2. Actual and three day-ahead forecasts, November 1-7 only; overall test metrics use November-December."
                )
                story.append(
                    KeepTogether(
                        [
                            Image(str(fn), width=450, height=135),
                            Paragraph(cap, styles["BodyText"]),
                            Spacer(1, 6),
                        ]
                    )
                )
                md.append(f"\n![{cap}](../figures/{fn.name})\n\n{cap}")
                continue
            is_subsection = (
                heading[:1].isdigit()
                and len(heading.split()[0].split(".")) == 2
                and heading.split()[0].split(".")[1].isdigit()
            )
            story.extend(
                [
                    Paragraph(
                        heading,
                        styles["Heading2"] if is_subsection else styles["Heading1"],
                    ),
                    Paragraph(body.replace("\n\n", "<br/><br/>"), styles["BodyText"]),
                ]
            )
            md.extend([("\n### " if is_subsection else "\n## ") + heading, body])

    def footer(canvas, doc):
        """Draw a page number in the report footer."""
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(545, 22, f"Page {doc.page}")

    SimpleDocTemplate(
        str(ROOT / "report/Report.pdf"),
        leftMargin=45,
        rightMargin=45,
        topMargin=30,
        bottomMargin=35,
        title=title,
        author="",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    (ROOT / "report/Report.md").write_text("\n\n".join(md), encoding="utf-8")

    from report_word import generate_word

    generate_word()


if __name__ == "__main__":
    metrics = pd.read_csv(ROOT / "results/metrics.csv")
    quality = pd.read_csv(ROOT / "results/data_quality_summary.csv", index_col=0)[
        "value"
    ].to_dict()
    quality = {
        key: int(value) if str(value).isdigit() else value
        for key, value in quality.items()
    }
    forecasts = pd.read_csv(
        ROOT / "results/forecast_results.csv", parse_dates=["timestamp"]
    ).set_index("timestamp")
    start = next(
        day
        for day in pd.date_range("2009-11-01", "2009-11-24")
        if pd.date_range(day, periods=168, freq="h").isin(forecasts.index).all()
    )
    report(metrics, quality, start)
