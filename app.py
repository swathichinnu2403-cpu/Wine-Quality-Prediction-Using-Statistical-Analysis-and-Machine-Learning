import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from pathlib import Path

# ------------------------------------------------------------
# Wine Quality Prediction and Statistical Analysis Dashboard
# ------------------------------------------------------------
st.set_page_config(
    page_title="Wine Quality Analysis Dashboard",
    page_icon="🍷",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

st.title("🍷 Wine Quality Analysis Dashboard")
st.caption("Interactive dashboard for exploring wine chemistry, quality, and model results.")

# ---------- Helper functions ----------
def find_file(filename):
    """Find a project CSV beside app.py."""
    path = BASE_DIR / filename
    return path if path.exists() else None


def load_csv(filename):
    path = find_file(filename)
    if path is None:
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.warning(f"Could not read {filename}: {exc}")
        return None


def find_column(df, candidates):
    """Find a column without depending on capitalization or spaces."""
    if df is None:
        return None
    normalized = {str(c).strip().lower().replace("_", " "): c for c in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower().replace("_", " ")
        if key in normalized:
            return normalized[key]
    return None


def show_missing_file(filename, purpose):
    st.info(
        f"`{filename}` was not found. Add it to the same folder as `app.py` "
        f"to display {purpose}."
    )


# ---------- Load cleaned data ----------
data_path = find_file("wine_quality_cleaned.csv")
if data_path is None:
    st.error(
        "The file `wine_quality_cleaned.csv` was not found. "
        "Place it in the same folder as app.py and run the app again."
    )
    st.stop()

try:
    df = pd.read_csv(data_path)
except Exception as exc:
    st.error(f"Unable to load the cleaned dataset: {exc}")
    st.stop()

if "quality" not in df.columns:
    st.error("The cleaned dataset must contain a `quality` column.")
    st.stop()

# Convert columns to numeric where possible
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")

numeric_df = df.select_dtypes(include=np.number)
if "quality" not in numeric_df.columns:
    st.error("The `quality` column must contain numeric values.")
    st.stop()

# ---------- Sidebar controls ----------
st.sidebar.header("Dashboard filters")
st.sidebar.write("Use these controls to explore different groups of wines.")

quality_values = sorted(df["quality"].dropna().unique().tolist())
if quality_values:
    selected_quality = st.sidebar.multiselect(
        "Select quality scores",
        options=quality_values,
        default=quality_values
    )
else:
    selected_quality = []

filtered = df[df["quality"].isin(selected_quality)].copy()

numeric_columns = numeric_df.columns.tolist()
if numeric_columns:
    property_to_view = st.sidebar.selectbox(
        "Choose a chemical property",
        options=[c for c in numeric_columns if c != "quality"] or numeric_columns,
        index=0
    )
else:
    property_to_view = None

# Optional quality group filter
if "quality" in filtered.columns and not filtered.empty:
    group_options = ["All", "Good (quality ≥ 7)", "Not Good (quality < 7)"]
    quality_group = st.sidebar.selectbox("Quality group", group_options)
    if quality_group == "Good (quality ≥ 7)":
        filtered = filtered[filtered["quality"] >= 7]
    elif quality_group == "Not Good (quality < 7)":
        filtered = filtered[filtered["quality"] < 7]

if filtered.empty:
    st.warning("No rows match the selected filters. Change the filters in the sidebar.")
    st.stop()

# ---------- KPI cards ----------
total_observations = len(filtered)
average_quality = filtered["quality"].mean()
median_quality = filtered["quality"].median()
unique_scores = filtered["quality"].nunique()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total observations", f"{total_observations:,}")
k2.metric("Average quality", f"{average_quality:.2f}")
k3.metric("Median quality", f"{median_quality:.1f}")
k4.metric("Distinct quality scores", f"{unique_scores}")

st.divider()

# ---------- Quality distribution ----------
st.subheader("1. Distribution of wine quality scores")
quality_counts = (
    filtered["quality"].value_counts()
    .sort_index()
    .rename_axis("Quality score")
    .reset_index(name="Number of wines")
)
fig_quality = px.bar(
    quality_counts,
    x="Quality score",
    y="Number of wines",
    text="Number of wines",
    title="Number of wines at each quality score",
    color="Quality score",
    color_continuous_scale="Blues"
)
fig_quality.update_layout(coloraxis_showscale=False)
st.plotly_chart(fig_quality, use_container_width=True)

# ---------- Chemical property exploration ----------
st.subheader("2. Chemical-property exploration")
left, right = st.columns(2)

with left:
    if property_to_view:
        fig_hist = px.histogram(
            filtered,
            x=property_to_view,
            nbins=30,
            title=f"Distribution of {property_to_view}",
            marginal="box",
            color_discrete_sequence=["#3977A8"]
        )
        st.plotly_chart(fig_hist, use_container_width=True)

with right:
    if property_to_view:
        fig_scatter = px.scatter(
            filtered,
            x=property_to_view,
            y="quality",
            opacity=0.65,
            trendline="ols",
            title=f"{property_to_view} vs wine quality",
            color="quality",
            color_continuous_scale="RdYlGn"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# ---------- Correlation heatmap ----------
st.subheader("3. Chemical-property correlation")
corr = numeric_df.corr(numeric_only=True)
fig_corr = px.imshow(
    corr,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Correlation heatmap of numeric variables"
)
st.plotly_chart(fig_corr, use_container_width=True)
st.caption("Correlation ranges from -1 (strong negative relationship) to +1 (strong positive relationship). Correlation does not prove causation.")

# ---------- Model performance comparison ----------
st.divider()
st.subheader("4. Model performance comparison")

regression_results = load_csv("regression_model_comparison.csv")
classification_results = load_csv("classification_model_comparison.csv")
roc_results = load_csv("classification_roc_auc_comparison.csv")

tab_reg, tab_cls, tab_roc = st.tabs(
    ["Regression models", "Classification models", "ROC-AUC"]
)

with tab_reg:
    if regression_results is not None:
        st.dataframe(regression_results, use_container_width=True)
        model_col = find_column(regression_results, ["Model"])
        rmse_col = find_column(regression_results, ["RMSE"])
        if model_col and rmse_col:
            plot_df = regression_results.copy()
            plot_df[rmse_col] = pd.to_numeric(plot_df[rmse_col], errors="coerce")
            plot_df = plot_df.dropna(subset=[rmse_col])
            if not plot_df.empty:
                fig = px.bar(
                    plot_df, x=model_col, y=rmse_col, color=model_col,
                    title="Regression model comparison (lower RMSE is better)",
                    text_auto=".3f"
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        show_missing_file("regression_model_comparison.csv", "regression model comparison")

with tab_cls:
    if classification_results is not None:
        st.dataframe(classification_results, use_container_width=True)
        model_col = find_column(classification_results, ["Model"])
        metric_cols = [
            find_column(classification_results, [m])
            for m in ["Accuracy", "Precision", "Recall", "F1-Score", "F1 Score"]
        ]
        metric_cols = list(dict.fromkeys([c for c in metric_cols if c]))
        if model_col and metric_cols:
            long_df = classification_results.melt(
                id_vars=[model_col], value_vars=metric_cols,
                var_name="Metric", value_name="Score"
            )
            long_df["Score"] = pd.to_numeric(long_df["Score"], errors="coerce")
            fig = px.bar(
                long_df, x=model_col, y="Score", color="Metric",
                barmode="group", title="Classification metrics by model",
                text_auto=".2f"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        show_missing_file("classification_model_comparison.csv", "classification metrics")

with tab_roc:
    if roc_results is not None:
        st.dataframe(roc_results, use_container_width=True)
        model_col = find_column(roc_results, ["Model"])
        auc_col = find_column(roc_results, ["ROC-AUC", "ROC AUC", "AUC"])
        if model_col and auc_col:
            plot_df = roc_results.copy()
            plot_df[auc_col] = pd.to_numeric(plot_df[auc_col], errors="coerce")
            fig = px.bar(
                plot_df.dropna(subset=[auc_col]), x=model_col, y=auc_col,
                color=model_col, title="ROC-AUC comparison (higher is better)",
                text_auto=".3f"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        show_missing_file("classification_roc_auc_comparison.csv", "ROC-AUC comparison")

# ---------- Actual vs predicted ----------
st.divider()
st.subheader("5. Regression: actual vs predicted values")
prediction_results = load_csv("regression_prediction_results.csv")

if prediction_results is not None:
    actual_col = find_column(prediction_results, ["Actual", "Actual Quality", "y_test", "y_true"])
    if actual_col:
        prediction_cols = [
            c for c in prediction_results.columns
            if c != actual_col and pd.api.types.is_numeric_dtype(
                pd.to_numeric(prediction_results[c], errors="coerce")
            )
        ]
        if prediction_cols:
            chosen_prediction = st.selectbox(
                "Choose a regression model prediction",
                prediction_cols
            )
            plot_df = prediction_results[[actual_col, chosen_prediction]].copy()
            plot_df[actual_col] = pd.to_numeric(plot_df[actual_col], errors="coerce")
            plot_df[chosen_prediction] = pd.to_numeric(plot_df[chosen_prediction], errors="coerce")
            plot_df = plot_df.dropna()
            fig = px.scatter(
                plot_df, x=actual_col, y=chosen_prediction,
                title=f"Actual vs predicted values — {chosen_prediction}",
                opacity=0.65
            )
            low = min(plot_df[actual_col].min(), plot_df[chosen_prediction].min())
            high = max(plot_df[actual_col].max(), plot_df[chosen_prediction].max())
            fig.add_shape(
                type="line", x0=low, y0=low, x1=high, y1=high,
                line=dict(dash="dash", color="red")
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Points closer to the dashed diagonal line indicate more accurate predictions.")
        else:
            st.warning("No numeric prediction columns were found in regression_prediction_results.csv.")
    else:
        st.warning("Could not identify the actual-value column. Expected a column such as `Actual`.")
else:
    show_missing_file("regression_prediction_results.csv", "actual-versus-predicted visualization")

# ---------- Classification metrics and confusion matrix ----------
st.divider()
st.subheader("6. Classification metrics and confusion matrix")
classification_predictions = load_csv("classification_prediction_results.csv")

if classification_predictions is not None:
    actual_class_col = find_column(
        classification_predictions,
        ["Actual", "Actual Class", "y_test", "y_true", "Actual Quality Category"]
    )
    predicted_class_col = find_column(
        classification_predictions,
        ["Predicted", "Predicted Class", "y_pred", "Prediction", "Predicted Quality Category"]
    )
    if actual_class_col and predicted_class_col:
        actual_labels = classification_predictions[actual_class_col].astype(str)
        predicted_labels = classification_predictions[predicted_class_col].astype(str)
        labels = sorted(set(actual_labels.dropna()) | set(predicted_labels.dropna()))
        cm = pd.crosstab(
            pd.Categorical(actual_labels, categories=labels),
            pd.Categorical(predicted_labels, categories=labels),
            rownames=["Actual"], colnames=["Predicted"],
            dropna=False
        )
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Confusion matrix"
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.warning(
            "The prediction file was found, but actual/predicted class columns could not be identified. "
            "Expected columns such as `Actual` and `Predicted`."
        )
else:
    show_missing_file(
        "classification_prediction_results.csv",
        "the confusion matrix (actual and predicted class labels are needed)"
    )

# ---------- Important variables ----------
st.divider()
st.subheader("7. Important variables")

importance_results = load_csv("feature_importance_report.csv")
if importance_results is not None:
    feature_col = find_column(importance_results, ["Feature", "Variable", "Feature Name"])
    importance_col = find_column(
        importance_results,
        ["Importance", "Permutation Importance", "Mean Importance", "Importance Mean"]
    )
    if feature_col and importance_col:
        importance_plot = importance_results.copy()
        importance_plot[importance_col] = pd.to_numeric(
            importance_plot[importance_col], errors="coerce"
        )
        importance_plot = importance_plot.dropna(subset=[importance_col]).sort_values(
            importance_col, ascending=True
        )
        fig_imp = px.bar(
            importance_plot, x=importance_col, y=feature_col,
            orientation="h", title="Feature importance",
            color=importance_col, color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        st.dataframe(importance_results, use_container_width=True)
    else:
        st.warning("Could not identify feature and importance columns in feature_importance_report.csv.")
else:
    show_missing_file("feature_importance_report.csv", "the feature-importance chart")

# ---------- Evidence-based findings ----------
st.divider()
st.subheader("8. Evidence-based findings")
st.write("The findings below are calculated from the cleaned dataset and update when the filters change.")

findings = []

quality_corr = numeric_df.corr(numeric_only=True)["quality"].drop("quality").dropna()
if not quality_corr.empty:
    strongest_feature = quality_corr.abs().idxmax()
    strongest_value = quality_corr[strongest_feature]
    direction = "positive" if strongest_value > 0 else "negative"
    findings.append(
        f"**Strongest linear association with quality:** `{strongest_feature}` "
        f"(correlation = {strongest_value:.3f}), showing a {direction} relationship."
    )

if "alcohol" in filtered.columns:
    high = filtered.loc[filtered["quality"] >= 7, "alcohol"].dropna()
    lower = filtered.loc[filtered["quality"] < 7, "alcohol"].dropna()
    if len(high) and len(lower):
        findings.append(
            f"**Alcohol comparison:** average alcohol is {high.mean():.2f} for wines with "
            f"quality ≥ 7 and {lower.mean():.2f} for wines with quality < 7."
        )

mode_quality = filtered["quality"].mode()
if not mode_quality.empty:
    mode_count = int((filtered["quality"] == mode_quality.iloc[0]).sum())
    findings.append(
        f"**Most common quality score:** {mode_quality.iloc[0]:g}, appearing in "
        f"{mode_count:,} of the {len(filtered):,} filtered observations."
    )

if "volatile acidity" in filtered.columns:
    va_corr = filtered[["volatile acidity", "quality"]].corr().iloc[0, 1]
    if pd.notna(va_corr):
        findings.append(
            f"**Volatile acidity relationship:** its correlation with quality is {va_corr:.3f} "
            f"for the currently filtered data."
        )

if property_to_view and property_to_view in filtered.columns:
    findings.append(
        f"**Selected chemical property:** `{property_to_view}` ranges from "
        f"{filtered[property_to_view].min():.3f} to {filtered[property_to_view].max():.3f} "
        f"in the filtered observations."
    )

for item in findings:
    st.markdown(f"- {item}")

st.caption(
    "Note: These are descriptive findings from this dataset, not proof that a chemical property causes wine quality."
)

# ---------- Data preview and download ----------
with st.expander("View filtered dataset"):
    st.dataframe(filtered, use_container_width=True)
    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data as CSV",
        data=csv_bytes,
        file_name="wine_quality_filtered.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("Wine Quality Prediction and Statistical Analysis | Built with Streamlit")
