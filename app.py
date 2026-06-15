import datetime
from typing import Optional
import io

import numpy as np
import pandas as pd
import streamlit as st


def safe_pct_change(current: float, baseline: float) -> Optional[float]:
    """Calculate a safe percent change and avoid division by zero."""
    if pd.isna(current) or pd.isna(baseline):
        return np.nan
    if baseline == 0:
        if current == 0:
            return 0.0
        return np.sign(current) * np.inf
    return (current - baseline) / baseline * 100


def get_previous_iso_week(year: int, week: int) -> tuple[int, int]:
    """Return the previous ISO week for a given year and week."""
    if week > 1:
        return year, week - 1

    previous_year = year - 1
    previous_week = datetime.date(previous_year, 12, 28).isocalendar()[1]
    return previous_year, previous_week


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def validate_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Folgende Spalten fehlen in der Datei: {', '.join(missing)}")


def parse_int_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def parse_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)
def fmt_thousands_point(x) -> str:
    """Format number with German thousands separator (.) and no decimals."""
    if pd.isna(x):
        return "N/A"
    try:
        return f"{x:,.0f}".replace(",", ".")
    except Exception:
        return str(x)


def fmt_percent_no_decimal(x) -> str:
    """Format percent without decimal places and with comma as decimal separator if needed."""
    if pd.isna(x):
        return "N/A"
    try:
        s = f"{x:.0f}%"
        return s.replace(".", ",")
    except Exception:
        return str(x)


def build_monitoring_table(df: pd.DataFrame, threshold: float, min_volume: float) -> pd.DataFrame:
    df = normalize_columns(df)
    required_columns = ["Jahr", "KW", "tDM Customer ohne SC", "0"]
    validate_required_columns(df, required_columns)

    df["Jahr"] = parse_int_series(df["Jahr"]).astype(int)
    df["KW"] = parse_int_series(df["KW"]).astype(int)
    df["tDM Customer ohne SC"] = df["tDM Customer ohne SC"].astype(str).str.strip()
    df["volume_0"] = parse_float_series(df["0"]).fillna(0.0)

    df = df[["Jahr", "KW", "tDM Customer ohne SC", "volume_0"]].copy()

    current_year, current_kw = df.loc[df["Jahr"].idxmax(), "Jahr"], None
    latest_years = df[df["Jahr"] == current_year]
    if not latest_years.empty:
        current_kw = int(latest_years["KW"].max())
    else:
        raise ValueError("Konnte die aktuelle Kalenderwoche nicht bestimmen.")

    prev_year_1, prev_kw_1 = get_previous_iso_week(current_year, current_kw)
    prev_year_2, prev_kw_2 = get_previous_iso_week(prev_year_1, prev_kw_1)

    results = []
    grouped = df.groupby("tDM Customer ohne SC", sort=True)

    for customer, group in grouped:
        group = group.sort_values(["Jahr", "KW"]).reset_index(drop=True)
        current_mask = (group["Jahr"] == current_year) & (group["KW"] == current_kw)
        if not current_mask.any():
            continue

        current_value = float(group.loc[current_mask, "volume_0"].iloc[0])
        prev_value = float(group.loc[(group["Jahr"] == prev_year_1) & (group["KW"] == prev_kw_1), "volume_0"].iloc[0]) if ((group["Jahr"] == prev_year_1) & (group["KW"] == prev_kw_1)).any() else np.nan
        prev_prev_value = float(group.loc[(group["Jahr"] == prev_year_2) & (group["KW"] == prev_kw_2), "volume_0"].iloc[0]) if ((group["Jahr"] == prev_year_2) & (group["KW"] == prev_kw_2)).any() else np.nan

        current_index = group.index[current_mask][0]
        prior_rows = group.loc[:current_index - 1, "volume_0"] if current_index > 0 else pd.Series(dtype=float)

        avg_4 = prior_rows.tail(4).mean() if len(prior_rows) > 0 else np.nan
        avg_8 = prior_rows.tail(8).mean() if len(prior_rows) > 0 else np.nan

        diff_prev = safe_pct_change(current_value, prev_value)
        diff_avg4 = safe_pct_change(current_value, avg_4)
        diff_avg8 = safe_pct_change(current_value, avg_8)

        comparison_value = avg_4 if not np.isnan(avg_4) else prev_value if not np.isnan(prev_value) else avg_8
        comparison_diff = safe_pct_change(current_value, comparison_value)

        results.append(
            {
                "tDM Customer ohne SC": customer,
                "Jahr": current_year,
                "KW": current_kw,
                "Aktuelle Woche (0)": current_value,
                "Vorwoche": prev_value,
                "Vor-Vorwoche": prev_prev_value,
                "Durchschnitt 4 Wochen": avg_4,
                "Durchschnitt 8 Wochen": avg_8,
                "% Veränderung vs Vorwoche": diff_prev,
                "% Veränderung vs Ø 4 Wochen": diff_avg4,
                "% Veränderung vs Ø 8 Wochen": diff_avg8,
                "% Veränderung (Hauptvergleich)": comparison_diff,
            }
        )

    result_df = pd.DataFrame(results)
    if result_df.empty:
        return result_df

    filtered = result_df[
        (
            (result_df["Aktuelle Woche (0)"] == 0)
            | (
                (result_df["Aktuelle Woche (0)"] >= float(min_volume))
                & (result_df["% Veränderung (Hauptvergleich)"] < float(threshold))
            )
        )
    ].copy()

    filtered = filtered.sort_values("% Veränderung (Hauptvergleich)", ascending=True).reset_index(drop=True)
    return filtered


def style_drop(row: pd.Series, threshold: float) -> list[str]:
    styles = []
    for col in row.index:
        if col in ["% Veränderung vs Vorwoche", "% Veränderung vs Ø 4 Wochen", "% Veränderung vs Ø 8 Wochen", "% Veränderung (Hauptvergleich)"]:
            value = row[col]
            if pd.notna(value) and value < threshold:
                styles.append("background-color: #ffc6c6")
            else:
                styles.append("")
        else:
            styles.append("")
    return styles


def main() -> None:
    st.set_page_config(page_title="E-Mail Versandmonitoring", layout="wide")
    st.title("Wöchentliches E-Mail Versandmonitoring")
    st.markdown(
        """
        Diese Anwendung analysiert das vertrauenswürdige E-Mail-Versandvolumen aus Excel-Daten und identifiziert
        Kunden mit signifikanten Einbrüchen in der trustedDialog-Zustellung (Spalte `0`).
        """
    )

    with st.sidebar:
        st.header("Einstellungen")
        threshold = st.number_input(
            "Abweichungsschwelle (%)",
            value=-30.0,
            min_value=-100.0,
            max_value=0.0,
            step=1.0,
            help="Zeige nur Kunden mit prozentualem Rückgang unterhalb dieser Schwelle.",
        )
        min_volume = st.number_input(
            "Mindest-Mailvolumen (Spalte 0)",
            value=100,
            min_value=0,
            step=1,
            help="Zeige nur Kunden mit mindestens diesem aktuellen trustedDialog-Volumen oder solchen mit 0.",
        )
        st.markdown(
            "Lade eine Excel-Datei hoch, die die Spalten `Jahr`, `KW`, `tDM Customer ohne SC` und `0` enthält."
        )

    uploaded_file = st.file_uploader("Excel-Datei hochladen", type=["xlsx", "xls"])
    if uploaded_file is None:
        st.info("Bitte laden Sie eine Excel-Datei hoch, um das Monitoring zu starten.")
        return

    try:
        data = pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as exc:
        st.error(f"Fehler beim Laden der Excel-Datei: {exc}")
        return

    try:
        result_df = build_monitoring_table(data, threshold, min_volume)
    except Exception as exc:
        st.error(f"Fehler bei der Datenverarbeitung: {exc}")
        return

    if result_df.empty:
        st.warning("Keine auffälligen Kunden für die aktuellen Einstellungen gefunden.")
        return

    number_of_customers = len(result_df)
    average_drop = result_df["% Veränderung (Hauptvergleich)"].replace([np.inf, -np.inf], np.nan).dropna().mean()

    col1, col2 = st.columns(2)
    col1.metric("Auffällige Kunden", fmt_thousands_point(number_of_customers))
    col2.metric(
        "Durchschnittlicher Drop (%)",
        fmt_percent_no_decimal(average_drop) if pd.notna(average_drop) else "N/A",
    )

    styled = (
        result_df.style
        .apply(lambda row: style_drop(row, threshold), axis=1)
        .format(
            {
                "Aktuelle Woche (0)": fmt_thousands_point,
                "Vorwoche": fmt_thousands_point,
                "Vor-Vorwoche": fmt_thousands_point,
                "Durchschnitt 4 Wochen": fmt_thousands_point,
                "Durchschnitt 8 Wochen": fmt_thousands_point,
                "% Veränderung vs Vorwoche": fmt_percent_no_decimal,
                "% Veränderung vs Ø 4 Wochen": fmt_percent_no_decimal,
                "% Veränderung vs Ø 8 Wochen": fmt_percent_no_decimal,
                "% Veränderung (Hauptvergleich)": fmt_percent_no_decimal,
            },
            na_rep="N/A",
        )
    )

    st.dataframe(styled, use_container_width=True)

    # Export: CSV (German format) and Excel
    csv_str = result_df.to_csv(sep=';', decimal=',', index=False, float_format='%.0f')
    csv_bytes = csv_str.encode('utf-8-sig')

    towrite = io.BytesIO()
    with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
        result_df.to_excel(writer, index=False, sheet_name='Monitoring')
    towrite.seek(0)

    col_csv, col_xlsx = st.columns(2)
    col_csv.download_button(
        label="Download CSV (DE, ; sep, , decimal)",
        data=csv_bytes,
        file_name=f"monitoring_{datetime.date.today().isoformat()}.csv",
        mime="text/csv",
    )

    col_xlsx.download_button(
        label="Download Excel (.xlsx)",
        data=towrite.getvalue(),
        file_name=f"monitoring_{datetime.date.today().isoformat()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown(
        """
        **Hinweise:**
        - Es wird ausschließlich die Spalte `0` analysiert.
        - Kunden mit aktuellem Volumen `0` werden immer gezeigt.
        - Fehlende Vorwochen werden als `NaN` angezeigt.
        - Der Hauptvergleich basiert auf dem Durchschnitt der letzten 4 Wochen, falls vorhanden.
        """
    )


if __name__ == "__main__":
    main()
