# app.py
# Streamlit-App für Monitoring trustedDialog (Spalte "0")
# Features: Upload, Wochenvergleich, Schwellwert-Filter, Charts, Export

import io
import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="trustedDialog Monitoring", page_icon="📉", layout="wide")
st.title("📉 trustedDialog Monitoring – Volumen-Einbrüche finden")
st.caption("Upload wöchentlicher Exporte → automatische Ausreißer-Erkennung, Vergleich und Charts")

# ---------------------------
# Spalten-Aliasse & Hilfsfunktionen
# ---------------------------
EXPECTED_COLS_ALIASES = {
    "Jahr": "year", "Jahr von Date": "year", "Jahr von Datum": "year",
    "Quartal": "quarter", "Quartal von Date": "quarter",
    "Monat": "month", "Monat von Date": "month",
    "KW": "week",
    "Kunde": "customer", "tDM Customer ohne SC": "customer",
    "Gesamt Status": "total_all", "Gesamt": "total_all",
    "0": "td_0", "11": "col_11", "20": "col_20",
}

def _safe_number(x):
    if pd.isna(x):
        return pd.NA
    if isinstance(x, (int, float)):
        return x
    s = str(x).strip().replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return pd.NA

def load_any_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith((".csv", ".txt", ".tsv")):
        df = pd.read_csv(uploaded_file, sep="\t", dtype=str, encoding="utf-8-sig", engine="python")
    elif name.endswith((".xlsx", ".xls")):
        try:
            xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
            sheet = "Rohdaten" if "Rohdaten" in xls.sheet_names else xls.sheet_names[0]
            df = xls.parse(sheet_name=sheet, dtype=str)
        except Exception:
            df = pd.read_excel(uploaded_file, dtype=str, engine="openpyxl")
    else:
        st.error("Nur CSV/TSV oder Excel werden unterstützt.")
        return pd.DataFrame()

    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={c: EXPECTED_COLS_ALIASES.get(c, c) for c in df.columns})

    if "customer" not in df.columns:
        customer_like = [c for c in df.columns if "customer" in c.lower() or "kunde" in c.lower()]
        if customer_like: df = df.rename(columns={customer_like[0]: "customer"})
    if "year" not in df.columns:
        year_like = [c for c in df.columns if c.lower().startswith("jahr")]
        if year_like: df = df.rename(columns={year_like[0]: "year"})
    if "week" not in df.columns:
        week_like = [c for c in df.columns if c.upper() in ("KW", "WEEK")]
        if week_like: df = df.rename(columns={week_like[0]: "week"})
    if "td_0" not in df.columns:
        if "0" in df.columns: df = df.rename(columns={"0": "td_0"})

    keep_cols = [c for c in ["year", "quarter", "month", "week", "customer", "td_0", "total_all"] if c in df.columns]
    df = df[keep_cols].copy()

    if "year" in df.columns: df["year"] = df["year"].map(_safe_number).astype("Int64")
    if "week" in df.columns: df["week"] = df["week"].map(_safe_number).astype("Int64")
    for num_col in ["td_0", "total_all"]:
        if num_col in df.columns: df[num_col] = df[num_col].map(_safe_number).astype("Float64")

    df = df.dropna(subset=["year", "week", "customer", "td_0"])
    df["customer"] = df["customer"].astype(str).str.strip()
    return df

@st.cache_data(show_spinner=False)
def concat_and_prepare(files) -> pd.DataFrame:
    frames = [load_any_file(f) for f in files]
    frames = [f for f in frames if not f.empty]
    if not frames: return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.groupby(["year", "week", "customer"], as_index=False, dropna=False)[["td_0"]].sum()
    df["year_week_key"] = df["year"].astype(int) * 100 + df["week"].astype(int)
    df["kw_label"] = df["year"].astype(int).astype(str) + "-KW" + df["week"].astype(int).astype(str).str.zfill(2)
    return df.sort_values(["year", "week", "customer"]).reset_index(drop=True)

def compute_baselines(df: pd.DataFrame, current_key: int, avg4_excl=True, avg8_excl=True) -> pd.DataFrame:
    def pct_change_safe(curr, base):
        if pd.isna(curr) or pd.isna(base): return pd.NA
        if base == 0 and curr == 0: return 0.0
        if base == 0 and curr > 0: return float("inf")
        return (curr - base) / base

    pvt = df.pivot_table(index="customer", columns="year_week_key", values="td_0", aggfunc="sum")
    keys = sorted(pvt.columns.tolist())
    if current_key not in keys: raise ValueError("Aktuelle Woche existiert nicht im Datensatz.")
    idx = {k: i for i, k in enumerate(keys)}; cur = idx[current_key]
    k1 = keys[cur-1] if cur-1 >= 0 else None
    k2 = keys[cur-2] if cur-2 >= 0 else None

    def prev_window(keys, end_pos, length, excl):
        end = end_pos - 1 if excl else end_pos
        start = max(0, end - length + 1)
        return keys[start:end+1] if end >= 0 else []

    keys4 = prev_window(keys, cur, 4, avg4_excl)
    keys8 = prev_window(keys, cur, 8, avg8_excl)

    res = pd.DataFrame(index=pvt.index).reset_index()
    res["curr"] = pvt.get(current_key)
    res["w_1"] = pvt.get(k1) if k1 is not None else pd.NA
    res["w_2"] = pvt.get(k2) if k2 is not None else pd.NA
    res["avg_4"] = pvt[keys4].mean(axis=1) if keys4 else pd.NA
    res["avg_8"] = pvt[keys8].mean(axis=1) if keys8 else pd.NA
    res["dev_vs_w1"] = [pct_change_safe(c, b) for c, b in zip(res["curr"], res["w_1"])]
    res["dev_vs_w2"] = [pct_change_safe(c, b) for c, b in zip(res["curr"], res["w_2"])]
    res["dev_vs_avg4"] = [pct_change_safe(c, b) for c, b in zip(res["curr"], res["avg_4"])]
    res["dev_vs_avg8"] = [pct_change_safe(c, b) for c, b in zip(res["curr"], res["avg_8"])]
    return res

def format_pct(x):
    if pd.isna(x): return "–"
    if x == float("inf"): return "+∞"
    return f"{x:.1%}"

def df_to_excel_bytes(df: pd.DataFrame, sheet_name="Auffällige Kunden") -> bytes:
    with io.BytesIO() as buffer:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
        return buffer.getvalue()

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.header("⚙️ Einstellungen")
    uploaded_files = st.file_uploader(
        "Wöchentlichen Export hochladen (CSV/TSV oder Excel). Mehrfach-Upload möglich.",
        type=["csv", "tsv", "txt", "xlsx", "xls"],
        accept_multiple_files=True
    )
    avg_excl_current = st.toggle("Ø von 4/8 Wochen **ohne** aktuelle Woche", value=True)
    threshold_pct = st.slider("Schwellwert für Negativabweichung", min_value=-0.95, max_value=0.0,
                              value=-0.30, step=0.01,
                              help="Beispiel: -0.30 = -30%")
    focus_last_n_weeks = st.number_input("Trend (N letzte Wochen)", min_value=8, max_value=52, value=12, step=1)
    st.caption("Hinweis: Spalte **„0“** (trustedDialog) wird als Volumen verwendet.")

# ---------------------------
# Hauptlogik
# ---------------------------
if not uploaded_files:
    st.info("Bitte lade mindestens eine Datei hoch, um zu starten.")
    st.stop()

df = concat_and_prepare(uploaded_files)
if df.empty:
    st.warning("Keine verwertbaren Daten gefunden. Bitte Spalten prüfen.")
    st.stop()

keys_sorted = sorted(df["year_week_key"].unique().tolist())
current_key = st.selectbox(
    "Aktuelle Woche",
    options=keys_sorted,
    index=len(keys_sorted)-1,
    format_func=lambda k: df.drop_duplicates("year_week_key").set_index("year_week_key")["kw_label"].to_dict().get(k, str(k))
)

baseline_choice = st.selectbox(
    "Vergleichs-Baseline",
    options=[
        ("dev_vs_w1", "Vorwoche"),
        ("dev_vs_w2", "Vor‑Vorwoche"),
        ("dev_vs_avg4", "Ø letzte 4 Wochen"),
        ("dev_vs_avg8", "Ø letzte 8 Wochen"),
    ],
    format_func=lambda t: t[1]
)

res = compute_baselines(df, current_key, avg4_excl=avg_excl_current, avg8_excl=avg_excl_current)

metric_col = baseline_choice[0]
res_filtered = res[(res[metric_col] < threshold_pct) | (res["curr"] == 0)].sort_values([metric_col, "curr"], ascending=[True, True])

pretty_cols = {
    "customer": "Kunde", "curr": "Aktuelle Woche", "w_1": "Vorwoche", "w_2": "Vor‑Vorwoche",
    "avg_4": "Ø letzte 4W", "avg_8": "Ø letzte 8W",
    "dev_vs_w1": "∆ vs Vorwoche", "dev_vs_w2": "∆ vs Vor‑Vorwoche",
    "dev_vs_avg4": "∆ vs Ø4W", "dev_vs_avg8": "∆ vs Ø8W",
}
show_cols = ["customer", "curr", "w_1", "w_2", "avg_4", "avg_8", "dev_vs_w1", "dev_vs_w2", "dev_vs_avg4", "dev_vs_avg8"]
res_display = res_filtered[show_cols].rename(columns=pretty_cols).copy()
for c in ["∆ vs Vorwoche", "∆ vs Vor‑Vorwoche", "∆ vs Ø4W", "∆ vs Ø8W"]:
    if c in res_display.columns: res_display[c] = res_display[c].map(format_pct)

st.subheader("🔎 Auffällige Kunden")
st.caption("Gefiltert nach Schwellwert und Kunden mit 0 Volumen in der aktuellen Woche")
st.dataframe(res_display, use_container_width=True, hide_index=True)

col_dl1, col_dl2, _ = st.columns([1, 1, 2])
with col_dl1:
    st.download_button("⬇️ CSV exportieren", data=res_filtered.to_csv(index=False).encode("utf-8"),
                       file_name="auffaellige_kunden.csv", mime="text/csv")
with col_dl2:
    st.download_button("⬇️ Excel exportieren",
                       data=df_to_excel_bytes(res_filtered),
                       file_name="auffaellige_kunden.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.subheader("📊 Visualisierungen")

# Top-Einbrüche
k = 15
topn = res_filtered[["customer", "curr", metric_col]].copy().head(k)
if not topn.empty:
    topn["Abweichung (%)"] = topn[metric_col] * 100
    bar = (alt.Chart(topn)
           .mark_bar(color="#d62728")
           .encode(
               y=alt.Y("customer:N", sort="-x", title="Kunde"),
               x=alt.X("Abweichung (%):Q", title="Abweichung in %"),
               tooltip=["customer", alt.Tooltip("Abweichung (%):Q", format=".1f"),
                        alt.Tooltip("curr:Q", title="Aktuelle Woche", format=",.0f")]
           )
           .properties(height=28*len(topn), width="container"))
    st.altair_chart(bar, use_container_width=True)
else:
    st.info("Keine Kunden erfüllen aktuell den Filter.")

# Trend für ausgewählte Kunden
st.markdown("### 📈 Verlauf je Kunde")
candidates = res_filtered["customer"].unique().tolist()
if len(candidates) == 0:
    candidates = res["customer"].head(10).tolist()
selected_customers = st.multiselect("Kunden auswählen", options=candidates,
                                    default=candidates[:min(5, len(candidates))])

if selected_customers:
    cur_pos = keys_sorted.index(current_key)
    start_pos = max(0, cur_pos - (focus_last_n_weeks - 1))
    keys_window = keys_sorted[start_pos:cur_pos + 1]
    trend = (df[df["customer"].isin(selected_customers) & df["year_week_key"].isin(keys_window)]
             .groupby(["customer", "kw_label"], as_index=False)["td_0"].sum())
    line = (alt.Chart(trend)
            .mark_line(point=True)
            .encode(
                x=alt.X("kw_label:N", sort=keys_window, title="Kalenderwoche"),
                y=alt.Y("td_0:Q", title="trustedDialog Volumen (Spalte 0)", axis=alt.Axis(format=",.0f")),
                color=alt.Color("customer:N", legend=alt.Legend(title="Kunde")),
                tooltip=["customer", "kw_label", alt.Tooltip("td_0:Q", title="Volumen", format=",.0f")]
            ).properties(height=400))
    st.altair_chart(line, use_container_width=True)

with st.expander("🧾 Alle Kunden – aktuelle Woche & Baselines (ungefiltert)"):
    all_display = res[show_cols].rename(columns=pretty_cols).copy()
    for c in ["∆ vs Vorwoche", "∆ vs Vor‑Vorwoche", "∆ vs Ø4W", "∆ vs Ø8W"]:
        if c in all_display.columns: all_display[c] = all_display[c].map(format_pct)
    st.dataframe(all_display, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Definition Abweichung: (Aktuelle Woche − Baseline) ÷ Baseline. Baseline=0 & aktuell>0 → +∞; 0→0 → 0 %.")
st.caption("Ø4W/Ø8W beziehen sich standardmäßig auf die vorangegangenen Wochen (ohne aktuelle Woche).")