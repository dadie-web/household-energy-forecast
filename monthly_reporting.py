"""Build the extended research report from saved metrics and diagnostics."""

import json
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Image,
    Table,
    Spacer,
    KeepTogether,
)

from monthly_backtest import OUTPUT, ROOT, verify_baseline

TITLE = (
    "Evaluating Day-Ahead Household Electricity Load Forecasting Across Different "
    "Months Using Expanding Training Windows"
)
PERIODS = ["april", "july", "october", "november_december"]
LABELS = ["April", "July", "October", "November-December"]
NAMES = {"yesterday": "Yesterday Naive", "weekly": "Weekly Naive", "AI": "HGB"}


def report_blocks():
    """Build a data-driven report focused on monthly contrasts and the July case."""
    verify_baseline()
    summary = json.loads(
        (ROOT / "results/july_analysis/july_summary.json").read_text(encoding="utf-8")
    )
    comparison = pd.read_csv(OUTPUT / "period_comparison_metrics.csv")
    plan = pd.read_csv(OUTPUT / "experiment_plan.csv")
    if len(comparison) != 12:
        raise ValueError("Expected three backtests and the original final-test results")
    indexed = comparison.set_index(["period", "method"])

    def value(period, method, metric):
        return f"{indexed.loc[(period, method), metric]:.6f}"

    rows = [["Period", "Yesterday MAE / RMSE", "Weekly MAE / RMSE", "HGB MAE / RMSE"]]
    for period, label in zip(PERIODS, LABELS):
        rows.append(
            [label]
            + [
                f"{value(period,m,'MAE_kW')} / {value(period,m,'RMSE_kW')}"
                for m in NAMES
            ]
        )
    errors = pd.read_csv(
        ROOT / "results/july_analysis/error_distribution.csv"
    ).set_index("method")
    error_rows = [
        ["Method", "Median AE (kW)", "Hours AE > 1 kW", "Max AE (kW)", "SSE (kW²)"]
    ]
    for method in NAMES:
        row = errors.loc[method]
        error_rows.append(
            [
                NAMES[method],
                f"{row.median_absolute_error_kW:.3f}",
                str(int(row.hours_absolute_error_gt_1kW)),
                f"{row.max_absolute_error_kW:.3f}",
                f"{row.sum_squared_error_kW2:.3f}",
            ]
        )
    july = summary
    mae_excess = 100 * (
        indexed.loc[("july", "AI"), "MAE_kW"]
        / indexed.loc[("july", "yesterday"), "MAE_kW"]
        - 1
    )
    rmse_reduction = 100 * (
        1
        - indexed.loc[("july", "AI"), "RMSE_kW"]
        / indexed.loc[("july", "yesterday"), "RMSE_kW"]
    )
    return [
        ("heading", TITLE, 0),
        ("heading", "Abstract", 1),
        (
            "paragraph",
            f'This study evaluates day-ahead household load forecasting with HistGradientBoostingRegressor (HGB), Yesterday Naive and Weekly Naive. Expanding training windows are used to predict April, July and October 2009. HGB has the lowest MAE and RMSE in April and October. In July, its MAE is {value("july","AI","MAE_kW")} kW, above both baselines, while its RMSE of {value("july","AI","RMSE_kW")} kW is slightly lower. A July case analysis finds frequent low-load overestimation alongside fewer large absolute errors. Longer training windows do not consistently correspond to lower forecast error across these months.',
            None,
        ),
        ("heading", "1. Introduction", 1),
        (
            "paragraph",
            "The UCI household dataset [1] offers a case study of irregular demand at a single home. This study compares HGB [2] with two persistence baselines to ask whether a learned model improves forecasts consistently across months. Calendar and lagged demand variables are established inputs in short-term load forecasting [3]. At household level, errors in the timing of sharp changes can affect pointwise scores [4]. Here, the same MAE and RMSE measures are used throughout, with particular attention to their different July rankings.",
            None,
        ),
        ("heading", "2. Dataset and Methodology", 1),
        (
            "paragraph",
            "Global_active_power is minute-average active power in kW. The existing January-December 2009 hourly series is reused. Timestamps are parsed as dd/mm/yyyy and hh:mm:ss and kept as recorded. Invalid and negative readings are missing, duplicate timestamps are averaged, and an hourly mean requires at least 54 valid minutes. Of 8760 hours, 8686 are valid and 74 lack sufficient observations. Targets are not interpolated.",
            None,
        ),
        (
            "paragraph",
            "At midnight, the next 24 hours are predicted from target hour, weekday, weekend, month, previous-day and previous-week same-hour loads, and trailing 24-hour and 168-hour means. All load inputs precede the origin. The rolling means are constant within a target day; observations from earlier days in the prediction month become available for later origins. Fitting requires complete features and a valid target. All three methods use identical eligible evaluation timestamps.",
            None,
        ),
        (
            "paragraph",
            "Yesterday Naive copies the previous day and Weekly Naive the previous week. HGB uses 150 iterations, 15 maximum leaves, learning rate 0.05, L2 regularization 1.0, seed 42 and disabled early stopping. Predictions are clipped at zero. Parameters are fixed across experiments. MAE averages absolute errors; RMSE places greater weight on large errors. No normal observations are removed on the basis of forecast error.",
            None,
        ),
        ("heading", "3. Expanding Window Backtesting", 1),
        (
            "paragraph",
            f"Experiment A fits January-March and forecasts April; B fits January-June and forecasts July; C fits January-September and forecasts October. Each model is fitted once before its prediction month. The usable training counts are {plan.iloc[0].train_valid_hours}, {plan.iloc[1].train_valid_hours} and {plan.iloc[2].train_valid_hours}. Invalid targets account for 3, 59 and 74 excluded training hours; incomplete history excludes another 504, 688 and 856 valid-target hours. All 720 April hours and all 744 hours in July and October are evaluated.",
            None,
        ),
        (
            "paragraph",
            "The November-December comparison comes from the earlier independent test: January-August training, September-October validation and January-October final refitting. The additional months are retrospective backtests, not replacement independent tests. Their training lengths and calendar periods change together, preventing separate estimates of a seasonal effect or a benefit from more training data.",
            None,
        ),
        ("heading", "4. Results and Analysis", 1),
        ("table", "Table 1. MAE / RMSE in kW across four evaluation periods.", rows),
        (
            "paragraph",
            f'HGB improves on both baselines in April and October and also has the lowest errors in the November-December test. Its lowest MAE is in October ({value("october","AI","MAE_kW")} kW), while its lowest RMSE is in July. In July, HGB MAE is {mae_excess:.2f}% higher than Yesterday Naive, but RMSE is {rmse_reduction:.2f}% lower. Neither the training-window length nor a single error measure produces a consistent ranking of forecast quality.',
            None,
        ),
        (
            "figure",
            "Figure 1. Daily mean household power in 2009. Days without 24 valid hours remain gaps.",
            "figure1_consumption_pattern.png",
        ),
        (
            "paragraph",
            f'July has mean actual load {july["actual_mean_kW"]:.3f} kW and median {july["actual_median_kW"]:.3f} kW; its middle half lies between {july["actual_p25_kW"]:.3f} and {july["actual_p75_kW"]:.3f} kW. Of 744 hours, {july["low_hours"]} ({100*july["low_fraction"]:.1f}%) are below 0.6 kW. HGB predicts {july["HGB_mean_kW"]:.3f} kW on average, giving mean signed error +{july["HGB_mean_error_kW"]:.3f} kW. In those low-load hours, it overestimates {july["low_HGB_overestimated_hours"]} times, with mean error +{july["low_HGB_bias_kW"]:.3f} kW. Every July day has a positive mean HGB error within its low-load hours.',
            None,
        ),
        (
            "figure",
            "Figure 2. July 1-7: the earliest complete seven-day window with 168 common valid hours, selected without using forecast errors.",
            "figure2_july_forecast.png",
        ),
        (
            "paragraph",
            f'The July curve is generally too high at low demand and too smooth around peaks. Across the whole month, {july["high_HGB_underestimated_hours"]} of the {july["high_hours"]} hours at or above the actual-load 90th percentile ({july["high_threshold_kW"]:.3f} kW) are underestimated, with mean signed error {july["high_HGB_bias_kW"]:.3f} kW. The month-wide maximum occurs outside the plotted week, on July 8 at 08:00: actual {july["monthly_peak_actual_kW"]:.3f} kW, HGB {july["monthly_peak_HGB_kW"]:.3f} kW. The percentile grouping is descriptive and does not change the evaluation sample.',
            None,
        ),
        (
            "table",
            "Table 2. July absolute-error distribution and sum of squared errors (SSE).",
            error_rows,
        ),
        (
            "paragraph",
            f'HGB has lower daily MAE than Yesterday Naive on {july["days_HGB_better_MAE_than_yesterday"]} days and higher MAE on {july["days_HGB_worse_MAE_than_yesterday"]}, with no ties. Its median absolute error is larger (Table 2), consistent with frequent moderate overestimation. However, it has fewer errors above 1 kW and a smaller maximum error. Its total squared error is also lower. These observed error distributions explain why MAE favors the baselines while RMSE slightly favors HGB; the lower RMSE does not remove the low-load bias or peak underestimation.',
            None,
        ),
        (
            "figure",
            "Figure 3. Comparison of MAE and RMSE across four evaluation periods.",
            "figure3_monthly_performance.png",
        ),
        ("heading", "5. Discussion and Conclusion", 1),
        (
            "paragraph",
            f'The eligible January-June training targets average {july["july_fit_target_mean_kW"]:.3f} kW, compared with {july["actual_mean_kW"]:.3f} kW in July. The fit also contains no July observations. This difference in load level and calendar coverage may contribute to the positive bias, but the experiment does not identify its cause. Calendar and past loads do not directly reveal future appliance switching or occupancy. Such unobserved activity is another possible reason for missed peaks.',
            None,
        ),
        (
            "paragraph",
            "Complete-history filtering reduces training coverage. Some accepted hourly means use 54-59 readings and may differ from a complete 60-minute mean. A single household and year, without uncertainty intervals or independent replication across years, limit generalization. Training size and month cannot be separated in this design.",
            None,
        ),
        (
            "paragraph",
            "HGB improves both errors in April and October, yet loses July MAE on most days. The earlier final test shows MAE/RMSE reductions of 27.57%/27.15% against Yesterday Naive and 18.58%/20.48% against Weekly Naive. These aggregate gains should be read alongside the July low-load bias and underestimated peaks, rather than as evidence that HGB is uniformly more accurate.",
            None,
        ),
        ("heading", "References", 1),
        (
            "paragraph",
            '[1] G. Hebrail and A. Berard, "Individual Household Electric Power Consumption," UCI Machine Learning Repository, 2006, doi: 10.24432/C58K54. Data license: CC BY 4.0.',
            None,
        ),
        (
            "paragraph",
            '[2] Scikit-learn developers, "HistGradientBoostingRegressor," scikit-learn documentation. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html.',
            None,
        ),
        (
            "paragraph",
            '[3] S. Fan and R. J. Hyndman, "Short-term load forecasting based on a semi-parametric additive model," IEEE Trans. Power Syst., vol. 27, no. 1, pp. 134-141, 2012, doi: 10.1109/TPWRS.2011.2162082.',
            None,
        ),
        (
            "paragraph",
            '[4] S. Haben, J. Ward, D. Vukadinovic Greetham, C. Singleton, and P. Grindrod, "A new error measure for forecasts of household-level, high resolution electrical energy consumption," Int. J. Forecast., vol. 30, no. 2, pp. 246-256, 2014, doi: 10.1016/j.ijforecast.2013.08.002.',
            None,
        ),
    ]


def build_report():
    """Render matching Markdown, PDF and editable Word content without fitting."""
    blocks = report_blocks()
    styles = getSampleStyleSheet()
    for name in ["BodyText", "Heading1", "Title"]:
        styles[name].fontName = "Times-Roman" if name == "BodyText" else "Times-Bold"
    styles["BodyText"].fontSize = 11
    styles["BodyText"].leading = 15
    styles["BodyText"].spaceAfter = 4
    styles["Heading1"].fontSize = 12
    styles["Heading1"].keepWithNext = True
    styles["Heading1"].spaceBefore = 7
    styles["Title"].fontSize = 14
    styles["Title"].leading = 18
    styles.add(
        ParagraphStyle(
            "References",
            parent=styles["BodyText"],
            fontSize=10,
            leading=12,
            spaceAfter=3,
        )
    )
    markdown, story = [], []
    for kind, text, extra in blocks:
        if kind == "heading":
            markdown.append(("# " if extra == 0 else "## ") + text)
            story.append(
                Paragraph(
                    escape(text), styles["Title"] if extra == 0 else styles["Heading1"]
                )
            )
        elif kind == "paragraph":
            markdown.append(text)
            story.append(
                Paragraph(
                    escape(text),
                    (
                        styles["References"]
                        if text.startswith("[")
                        else styles["BodyText"]
                    ),
                )
            )
        elif kind == "table":
            markdown.extend(
                [
                    text,
                    "\n".join(
                        "| " + " | ".join(row) + " |"
                        for row in [extra[0], ["---"] * len(extra[0]), *extra[1:]]
                    ),
                ]
            )
            widths = [92, 92, 92, 92, 92] if len(extra[0]) == 5 else [85, 125, 125, 125]
            table = Table(
                extra,
                colWidths=widths,
                style=[
                    ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf0f5")),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ],
            )
            group = [Paragraph(escape(text), styles["BodyText"]), table, Spacer(1, 6)]
            if (
                story
                and isinstance(story[-1], Paragraph)
                and story[-1].style.name == "Heading1"
            ):
                group.insert(0, story.pop())
            story.append(KeepTogether(group))
        else:
            markdown.extend([f"![{text}](../figures/{extra})", text])
            height = 156 if extra.startswith("figure3") else 135
            story.append(
                KeepTogether(
                    [
                        Image(str(ROOT / "figures" / extra), width=450, height=height),
                        Paragraph(escape(text), styles["BodyText"]),
                    ]
                )
            )
    (ROOT / "report/Report.md").write_text(
        "\n\n".join(markdown) + "\n", encoding="utf-8"
    )

    def footer(canvas, doc):
        canvas.setFont("Times-Roman", 9)
        canvas.drawRightString(545, 22, str(doc.page))

    SimpleDocTemplate(
        str(ROOT / "report/Report.pdf"),
        leftMargin=45,
        rightMargin=45,
        topMargin=32,
        bottomMargin=35,
        title=TITLE,
        author="",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    from report_word import generate_word

    generate_word()
    verify_baseline()


if __name__ == "__main__":
    build_report()
