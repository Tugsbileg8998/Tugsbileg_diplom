import subprocess
import sys

packages = ["streamlit", "pandas", "numpy", "plotly", "openpyxl", "scipy"]

for package in packages:
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="Өгөгдлийн танилцуулга", layout="wide")

FILE_PATH = "dashboard.xlsx"

@st.cache_data
def load_data():
    df = pd.read_excel(FILE_PATH, sheet_name=0, header=1)
    cols = df.columns.tolist()

    df = df.rename(columns={
        cols[0]: "Он",
        cols[1]: "Өрхийн дугаар",
        cols[2]: "Хүүхэд (14-с доош)",
        cols[3]: "Тэргүүлэх насанд хүрсэн",
        cols[4]: "Бусад насанд хүрсэн",
        cols[5]: "Нийт гишүүд",
        cols[6]: "Өрхийн жин",
        cols[7]: "Цалин, хөлс",
        cols[8]: "Тэтгэвэр, тэтгэмж болон бусад",
        cols[9]: "Үйлдвэрлэл үйлчилгээ",
        cols[10]: "Бэлэг тусламж болон өөрийн аж ахуйгаас хэрэглэсэн",
        cols[11]: "Нэрлэсэн  орлого",
        cols[12]: "Бодит орлого (2015 үнэ, ₮)"
    })

    numeric_cols = [
        "Он", "Хүүхэд (14-с доош)", "Тэргүүлэх насанд хүрсэн", "Бусад насанд хүрсэн", "Нийт гишүүд",
        "Өрхийн жин", "Цалин, хөлс", "Тэтгэвэр, тэтгэмж болон бусад", "Үйлдвэрлэл үйлчилгээ",
        "Бэлэг тусламж болон өөрийн аж ахуйгаас хэрэглэсэн", "Нэрлэсэн  орлого", "Бодит орлого (2015 үнэ, ₮)"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Он", "Өрхийн жин", "Нэрлэсэн  орлого"])
    df["Он"] = df["Он"].astype(int)

    df["Тэнцвэржүүлсэн масштаб"] = (
        df["Тэргүүлэх насанд хүрсэн"].fillna(0) * 1.0 +
        df["Бусад насанд хүрсэн"].fillna(0) * 0.5 +
        df["Хүүхэд (14-с доош)"].fillna(0) * 0.3
    )

    df = df[df["Тэнцвэржүүлсэн масштаб"] > 0].copy()
    df["Тэнцвэржүүлсэн орлого"] = df["Нэрлэсэн  орлого"] / df["Тэнцвэржүүлсэн масштаб"]

    return df


def weighted_median(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = ~(np.isnan(values) | np.isnan(weights))
    values = values[mask]
    weights = weights[mask]

    if len(values) == 0 or weights.sum() == 0:
        return np.nan

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cum_weights = np.cumsum(weights)
    cutoff = weights.sum() / 2

    return values[np.searchsorted(cum_weights, cutoff)]


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = ~(np.isnan(values) | np.isnan(weights))
    values = values[mask]
    weights = weights[mask]

    if len(values) == 0 or weights.sum() == 0:
        return np.nan

    return np.average(values, weights=weights)


def weighted_variance(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = ~(np.isnan(values) | np.isnan(weights))
    values = values[mask]
    weights = weights[mask]

    if len(values) == 0 or weights.sum() == 0:
        return np.nan

    mean = np.average(values, weights=weights)
    variance = np.average((values - mean) ** 2, weights=weights)
    return variance


def weighted_std(values, weights):
    var = weighted_variance(values, weights)
    if np.isnan(var):
        return np.nan
    return np.sqrt(var)


def weighted_percentile(values, weights, percentile):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = ~(np.isnan(values) | np.isnan(weights))
    values = values[mask]
    weights = weights[mask]

    if len(values) == 0 or weights.sum() == 0:
        return np.nan

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cum_weights = np.cumsum(weights)
    cutoff = weights.sum() * percentile / 100

    return values[np.searchsorted(cum_weights, cutoff)]


def weighted_gini(values, weights):
    """Жини коэффициентийг жинээр тооцно"""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = ~(np.isnan(values) | np.isnan(weights))
    values = values[mask]
    weights = weights[mask]

    if len(values) == 0 or weights.sum() == 0:
        return np.nan

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cum_values = np.cumsum(values * weights)
    total_weight = weights.sum()
    total_value = cum_values[-1]

    if total_value <= 0:
        return np.nan

    cum_values_prev = np.concatenate(([0.0], cum_values[:-1]))
    gini_numerator = 2 * np.sum(weights * cum_values_prev) + np.sum(weights * weights * values)
    gini = 1 - gini_numerator / (total_weight * total_value)
    return max(0.0, min(1.0, gini))


def weighted_lorenz(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = ~(np.isnan(values) | np.isnan(weights))
    values = values[mask]
    weights = weights[mask]

    if len(values) == 0 or weights.sum() == 0:
        return np.array([0.0, 100.0]), np.array([0.0, 100.0])

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cum_weights = np.cumsum(weights)
    cum_values = np.cumsum(values * weights)

    total_value = cum_values[-1]
    total_weight = weights.sum()

    lorenz_x = np.concatenate([[0.0], cum_weights / total_weight * 100])
    lorenz_y = np.concatenate([[0.0], cum_values / total_value * 100])
    return lorenz_x, lorenz_y


def weighted_skewness(values, weights):
    """Compute skewness with weights"""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = ~(np.isnan(values) | np.isnan(weights))
    values = values[mask]
    weights = weights[mask]

    if len(values) == 0 or weights.sum() == 0:
        return np.nan

    mean = np.average(values, weights=weights)
    std = weighted_std(values, weights)
    
    if std == 0 or np.isnan(std):
        return np.nan

    skew = np.average(((values - mean) / std) ** 3, weights=weights)
    return skew


def weighted_kurtosis(values, weights):
    """Compute excess kurtosis with weights"""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mask = ~(np.isnan(values) | np.isnan(weights))
    values = values[mask]
    weights = weights[mask]

    if len(values) == 0 or weights.sum() == 0:
        return np.nan

    mean = np.average(values, weights=weights)
    std = weighted_std(values, weights)
    
    if std == 0 or np.isnan(std):
        return np.nan

    kurt = np.average(((values - mean) / std) ** 4, weights=weights) - 3
    return kurt



df = load_data()

st.title("Монгол Улсын дундаж давхаргын тооцоолол")
st.caption("Жинлэсэн статистик, тэнцвэржүүлсэн орлогын арга")

years = sorted(df["Он"].dropna().unique())

selected_year = st.sidebar.selectbox(
    "Он сонгох",
    years,
    index=len(years) - 1
)

tab1, tab2, tab3 = st.tabs(["Жилүүдийн харьцуулалт", "Дундаж давхарга", "Дэлгэрэнгүй өгөгдлийн танилцуулга"]) 

with tab2:
    st.header("Дундаж давхарга")
    
    data = df[df["Он"] == selected_year].copy()
    data = data.dropna(subset=["Нэрлэсэн  орлого", "Тэнцвэржүүлсэн орлого", "Өрхийн жин"])
    
    median_eq = weighted_median(data["Тэнцвэржүүлсэн орлого"], data["Өрхийн жин"])
    lower = median_eq * 0.75
    upper = median_eq * 2.00
    
    def classify_income(x):
        if x < lower:
            return "Бага орлоготой"
        elif x <= upper:
            return "Дундаж орлоготой"
        else:
            return "Өндөр орлоготой"
    
    data["Орлогын бүлэг"] = data["Тэнцвэржүүлсэн орлого"].apply(classify_income)
    
    weighted_households = data["Өрхийн жин"].sum()
    mean_income = weighted_mean(data["Нэрлэсэн  орлого"], data["Өрхийн жин"])
    median_income = weighted_median(data["Нэрлэсэн  орлого"], data["Өрхийн жин"])
    gini_index = weighted_gini(data["Тэнцвэржүүлсэн орлого"], data["Өрхийн жин"])
    
    middle_households = data.loc[
        data["Орлогын бүлэг"] == "Дундаж орлоготой",
        "Өрхийн жин"
    ].sum()
    
    middle_share = middle_households / weighted_households * 100
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Он", f"{selected_year}")
    col2.metric("Жинлэсэн өрхийн тоо", f"{weighted_households:,.0f}")
    col3.metric("Жинлэсэн дундаж орлого", f"{mean_income:,.0f} ₮")
    col4.metric("Дундаж давхарга", f"{middle_share:.2f}%")
    col5.metric("Орлогын тэгш бус байдал", f"{gini_index:.4f}")
    
    st.divider()
    
    st.subheader("Тэнцвэржүүлсэн орлогын босго утгууд")
    
    threshold_df = pd.DataFrame({
        "Үзүүлэлт": [
            "Жинлэсэн медиан тэнцвэржүүлсэн орлого",
            "Доод босго (75%)",
            "Дээд босго (200%)"
        ],
        "Утга (₮)": [
            round(median_eq, 2),
            round(lower, 2),
            round(upper, 2)
        ]
    })
    
    st.dataframe(threshold_df, use_container_width=True)
    
    st.subheader("Орлогын бүлгийн бүтэц")
    
    group_df = (
        data
        .groupby("Орлогын бүлэг")
        ["Өрхийн жин"]
        .sum()
        .reset_index()
    )
    
    group_df.columns = ["Орлогын бүлэг", "Жинлэсэн өрхийн тоо"]
    
    group_df["Хувь (%)"] = (
        group_df["Жинлэсэн өрхийн тоо"] /
        group_df["Жинлэсэн өрхийн тоо"].sum() * 100
    )
    
    order = ["Бага орлоготой", "Дундаж орлоготой", "Өндөр орлоготой"]
    
    group_df["Орлогын бүлэг"] = pd.Categorical(
        group_df["Орлогын бүлэг"],
        categories=order,
        ordered=True
    )
    
    group_df = group_df.sort_values("Орлогын бүлэг")
    
    st.dataframe(group_df, use_container_width=True)
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        fig1 = px.bar(
            group_df,
            x="Орлогын бүлэг",
            y="Хувь (%)",
            text="Хувь (%)",
            title="Орлогын бүлгийн хувь (%)"
        )
        fig1.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col_chart2:
        fig2 = px.pie(
            group_df,
            names="Орлогын бүлэг",
            values="Жинлэсэн өрхийн тоо",
            title="Орлогын бүлгийн эзлэх хувь"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Орлогын хуваарилалтын Лоренцын муруй")
    lorenz_x, lorenz_y = weighted_lorenz(data["Тэнцвэржүүлсэн орлого"], data["Өрхийн жин"])
    fig_lorenz = go.Figure()
    fig_lorenz.add_trace(go.Scatter(
        x=lorenz_x,
        y=lorenz_y,
        mode="lines",
        name="Lorenz curve",
        line=dict(color="#636efa", width=3),
        hoverinfo="skip"
    ))
    fig_lorenz.add_trace(go.Scatter(
        x=[0, 100],
        y=[0, 100],
        mode="lines",
        name="Тэгш байдал",
        line=dict(color="#EF553B", dash="dash"),
        hoverinfo="skip"
    ))
    fig_lorenz.update_layout(
        xaxis_title="Хуримтлагдсан өрхийн эзлэх хувь (%)",
        yaxis_title="Хуримтлагдсан орлогын эзлэх хувь (%)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig_lorenz, use_container_width=True)

with tab1:
    st.header("Жилүүдийн харьцуулалт")
    
    yearly_stats = []
    
    for year in years:
        year_data = df[df["Он"] == year].copy()
        year_data = year_data.dropna(subset=["Нэрлэсэн  орлого", "Тэнцвэржүүлсэн орлого", "Өрхийн жин"])
        
        if len(year_data) > 0:
            yearly_stats.append({
                "Он": year,
                "Өрхийн тоо": int(year_data["Өрхийн жин"].sum()),
                "Дундаж орлого (₮)": round(weighted_mean(year_data["Нэрлэсэн  орлого"], year_data["Өрхийн жин"]), 0),
                "Медиан орлого (₮)": round(weighted_median(year_data["Нэрлэсэн  орлого"], year_data["Өрхийн жин"]), 0),
                "Стандарт хазайлт": round(weighted_std(year_data["Нэрлэсэн  орлого"], year_data["Өрхийн жин"]), 0),
                "Дундаж тэнцвэржүүлсэн (₮)": round(weighted_mean(year_data["Тэнцвэржүүлсэн орлого"], year_data["Өрхийн жин"]), 0),
                "Медиан тэнцвэржүүлсэн (₮)": round(weighted_median(year_data["Тэнцвэржүүлсэн орлого"], year_data["Өрхийн жин"]), 0),
                "Жини коэффициент": round(weighted_gini(year_data["Тэнцвэржүүлсэн орлого"], year_data["Өрхийн жин"]), 4)
            })
    
    yearly_df = pd.DataFrame(yearly_stats)
    st.dataframe(yearly_df, use_container_width=True)
    
    col_trend1, col_trend2 = st.columns(2)
    
    with col_trend1:
        fig_trend1 = px.line(
            yearly_df,
            x="Он",
            y="Дундаж орлого (₮)",
            title="Өрхийн жилийн тэнцвэржүүлсэн орлого",
            markers=True
        )
        fig_trend1.update_yaxes(tickformat=",.0f")
        st.plotly_chart(fig_trend1, use_container_width=True)
    
    with col_trend2:
        fig_trend2 = px.line(
            yearly_df,
            x="Он",
            y="Жини коэффициент",
            title="Орлогын тэгш бус байдал",
            markers=False
        )
        fig_trend2.update_traces(hoverinfo='skip')
        st.plotly_chart(fig_trend2, use_container_width=True)

    st.subheader("Жилүүдийн өрхийн орлогын эх үүсвэрийн бүтэц")
    income_sources = [
        "Цалин, хөлс",
        "Тэтгэвэр, тэтгэмж болон бусад",
        "Үйлдвэрлэл үйлчилгээ",
        "Бэлэг тусламж болон өөрийн аж ахуйгаас хэрэглэсэн"
    ]

    source_rows = []
    for year in years:
        year_data = df[df["Он"] == year].copy()
        year_data = year_data.dropna(subset=income_sources + ["Өрхийн жин"])

        if len(year_data) > 0:
            total_by_source = {
                source: (year_data[source] * year_data["Өрхийн жин"]).sum()
                for source in income_sources
            }
            total_income = sum(total_by_source.values())

            if total_income > 0:
                for source, value in total_by_source.items():
                    source_rows.append({
                        "Он": year,
                        "Эх үүсвэр": source,
                        "Хувь (%)": value / total_income * 100
                    })

    source_df = pd.DataFrame(source_rows)

    if not source_df.empty:
        fig_source = px.bar(
            source_df,
            x="Он",
            y="Хувь (%)",
            color="Эх үүсвэр",
            title="Жилүүдийн өрхийн орлогын эх үүсвэрийн бүтэц",
            labels={"Хувь (%)": "Хувь (%)"}
        )
        fig_source.update_layout(barmode="stack")
        fig_source.update_yaxes(tickformat=".0f%%", range=[0, 100])
        st.plotly_chart(fig_source, use_container_width=True)

with tab3:
    st.header("Сонгосон онын дэлгэрэнгүй өгөгдлийн танилцуулга")
    
    data = df[df["Он"] == selected_year].copy()
    data = data.dropna(subset=["Нэрлэсэн  орлого", "Тэнцвэржүүлсэн орлого", "Өрхийн жин"])
    
    desc_vars = [
        "Нэрлэсэн  орлого",
        "Бодит орлого (2015 үнэ, ₮)",
        "Тэнцвэржүүлсэн орлого",
        "Нийт гишүүд",
        "Хүүхэд (14-с доош)",
        "Тэргүүлэх насанд хүрсэн",
        "Бусад насанд хүрсэн",
        "Цалин, хөлс",
        "Тэтгэвэр, тэтгэмж болон бусад",
        "Үйлдвэрлэл үйлчилгээ",
        "Бэлэг тусламж болон өөрийн аж ахуйгаас хэрэглэсэн"
    ]
    
    stats_data = []
    
    for var in desc_vars:
        values = data[var].dropna()
        weights = data.loc[values.index, "Өрхийн жин"]
        
        stats_data.append({
            "Хувьсагч": var,
            "Тоо": len(values),
            "Дундаж": round(weighted_mean(values, weights), 2),
            "Медиан": round(weighted_median(values, weights), 2),
            "Стандарт хазайлт": round(weighted_std(values, weights), 2),
            "Хамгийн бага": round(values.min(), 2),
            "Q1 (25%)": round(weighted_percentile(values, weights, 25), 2),
            "Q3 (75%)": round(weighted_percentile(values, weights, 75), 2),
            "Хамгийн их": round(values.max(), 2),
            "Skewness": round(weighted_skewness(values, weights), 4),
            "Kurtosis": round(weighted_kurtosis(values, weights), 4)
        })
    
    desc_df = pd.DataFrame(stats_data)
    st.dataframe(desc_df, use_container_width=True)
    
    st.subheader("Орлогын эх үүсвэрийн статистик")
    
    data = df[df["Он"] == selected_year].copy()
    data = data.dropna(subset=["Нэрлэсэн  орлого", "Тэнцвэржүүлсэн орлого", "Өрхийн жин"])
    
    income_sources = {
        "Цалин, хөлс": data[["Цалин, хөлс", "Өрхийн жин"]].dropna(),
        "Тэтгэвэр, тэтгэмж болон бусад": data[["Тэтгэвэр, тэтгэмж болон бусад", "Өрхийн жин"]].dropna(),
        "Үйлдвэрлэл үйлчилгээ": data[["Үйлдвэрлэл үйлчилгээ", "Өрхийн жин"]].dropna(),
        "Бэлэг тусламж болон өөрийн аж ахуйгаас хэрэглэсэн": data[["Бэлэг тусламж болон өөрийн аж ахуйгаас хэрэглэсэн", "Өрхийн жин"]].dropna()
    }
    
    income_stats = []
    
    for source_name, source_data in income_sources.items():
        if len(source_data) > 0:
            values = source_data.iloc[:, 0]
            weights = source_data.iloc[:, 1]
            
            income_stats.append({
                "Эх үүсвэр": source_name,
                "Дундаж (₮)": round(weighted_mean(values, weights), 0),
                "Медиан (₮)": round(weighted_median(values, weights), 0),
                "Стандарт хазайлт": round(weighted_std(values, weights), 0),
                "Хамгийн их (₮)": round(values.max(), 0)
            })
    
    income_source_df = pd.DataFrame(income_stats)
    st.dataframe(income_source_df, use_container_width=True)
    
    fig_income = px.bar(
        income_source_df,
        x="Эх үүсвэр",
        y="Дундаж (₮)",
        title="Орлогын эх үүсвэрийн жинлэсэн дундаж"
    )
    fig_income.update_yaxes(tickformat=",.0f")
    st.plotly_chart(fig_income, use_container_width=True)

st.divider()

st.subheader("📥 Өгөгдлийг татах")

download_data = df[df["Он"] == selected_year].copy()
csv = download_data.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label=f"📊 CSV татах ({selected_year})",
    data=csv,
    file_name=f"өрхийн орлогын өгөгдөл_{selected_year}.csv",
    mime="text/csv"
)