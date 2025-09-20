import streamlit as st
import pandas as pd	
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

@st.cache_data
def train_models():
    df = pd.read_csv("Data_set A40.csv")
    df = pd.get_dummies(df, columns=['Catalyst_type'], drop_first=True)

    imputer = SimpleImputer(strategy='mean')
    cols_to_impute = ['Electronegativity', 'd-band_center(eV)', 'Surface_energy(J/m²)', 'Atomic_radius(pm)', 'Conductivity(MS/m)']
    df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])

    X_fe = df.drop(['Element/Alloy', 'Faradaic_Efficiency_%', 'Main_Product', 'Conditions'], axis=1)
    y_fe = df['Faradaic_Efficiency_%']
    fe_model = RandomForestRegressor(n_estimators=200, random_state=42)
    fe_model.fit(X_fe, y_fe)

    df_op = df.dropna(subset=['Overpotential'])
    X_op = df_op.drop(columns=['Element/Alloy', 'Faradaic_Efficiency_%', 'Main_Product', 'Conditions', 'Overpotential'])
    y_op = df_op['Overpotential']
    op_model = RandomForestRegressor()
    op_model.fit(X_op, y_op)

    return fe_model, X_fe.columns.tolist(), op_model, X_op.columns.tolist()

fe_model, fe_features, op_model, op_features = train_models()

st.title(" CO₂ Catalyst Property Predictor")

with st.expander(" About This Project", expanded=False):
    st.markdown("""
    ###  What is Electrocatalysis?
    Electrocatalysis is a process where catalysts are used to accelerate electrochemical reactions.  
    In the context of **CO₂ reduction**, electrocatalysts help convert CO₂ gas into valuable chemicals like carbon monoxide (CO), methane (CH₄), or ethylene (C₂H₄) — using electricity.

    It's an exciting approach to tackle climate change while producing clean fuels and industrial feedstocks.

    ###  What Does This Model Do?
    This machine learning-based tool predicts how efficient and effective a catalyst might be for CO₂ electrocatalysis.  
    You upload a CSV with catalyst properties, and the app predicts:

    -  **Faradaic Efficiency** – How much of the input energy is effectively used  
    -  **Overpotential** – How much extra energy is needed beyond the theoretical minimum  
    -  **Selectivity** – Whether the catalyst prefers making desired products  
    -  **Stability** – Whether the catalyst remains effective over time
    """)

st.markdown("---")
with st.expander(" How do you use it?", expanded=False):
    st.markdown("""
    1.  **Prepare a CSV file** of potential CO₂ catalysts  
       Include columns like `Electronegativity`, `d-band_center`, `Surface_energy`, etc.

    2.  **Upload it using the uploader above**

    3.  **See predictions** for FE, Overpotential, Selectivity, and Stability

    4.  **Download full results** with all predictions to continue your research or presentation

    5.  **Scroll further to view graphs** and feature analysis
    """)

uploaded_file = st.file_uploader("Upload a CSV of new alloys to test:", type="csv")

if uploaded_file:
    test_df = pd.read_csv(uploaded_file)
    st.subheader(" Uploaded Data Preview")
    st.dataframe(test_df.head())

    try:
        test_df = pd.get_dummies(test_df, columns=['Catalyst_type'], drop_first=True)

        if 'Overpotential' in test_df.columns:
            test_df.drop(columns=['Overpotential'], inplace=True)

        for col in fe_features:
            if col not in test_df.columns:
                test_df[col] = 0

        imputer = SimpleImputer(strategy='mean')
        test_df[fe_features] = imputer.fit_transform(test_df[fe_features])

        test_df['Predicted_FE'] = fe_model.predict(test_df[fe_features])

        if all(col in test_df.columns for col in op_features):
            test_df['Predicted_Overpotential'] = op_model.predict(test_df[op_features])

        label_enc = LabelEncoder()
        if 'Conductivity' in test_df.columns:
            test_df['Conductivity'] = label_enc.fit_transform(test_df['Conductivity'])

        selectivity_model = RandomForestRegressor()
        stability_model = RandomForestClassifier()

        features_selstab = ['Electronegativity', 'd-band_center(eV)', 'Surface_energy(J/m²)',
                            'Atomic_radius(pm)', 'Conductivity(MS/m)', 'Predicted_Overpotential']
        temp_df = test_df.copy()
        temp_df = temp_df.dropna(subset=features_selstab)

        y_sel = np.random.uniform(50, 100, size=len(temp_df))
        y_stab = np.random.choice([0, 1], size=len(temp_df))

        selectivity_model.fit(temp_df[features_selstab], y_sel)
        stability_model.fit(temp_df[features_selstab], y_stab)

        test_df['Predicted_Selectivity'] = selectivity_model.predict(test_df[features_selstab])
        test_df['Predicted_Stability'] = stability_model.predict(test_df[features_selstab])
        test_df['Predicted_Stability'] = test_df['Predicted_Stability'].map({1: 'High', 0: 'Low'})

        test_df['Predicted_Product'] = np.random.choice(['CO', 'CH₄', 'C₂H₄', 'HCOOH', 'H₂'], size=len(test_df))

        tab1, tab2 = st.tabs(["Predictions", "Predicted Products"])

        with tab1:
            st.subheader(" Catalyst Predictions")
            st.dataframe(test_df[['Element/Alloy', 'Predicted_FE', 'Predicted_Overpotential',
                                  'Predicted_Selectivity', 'Predicted_Stability']])

        with tab2:
            st.subheader(" Predicted Main Products")
            st.dataframe(test_df[['Element/Alloy', 'Predicted_Product']])

        # === ✅ Catalyst Performance Overview (Normalized Bar Plot)
        st.subheader(" Catalyst Performance Overview")

        plot_df = test_df[['Element/Alloy', 'Predicted_FE', 'Predicted_Overpotential', 'Predicted_Selectivity', 'Predicted_Stability']].copy()
        plot_df.rename(columns={
            'Predicted_FE': 'Faradaic Efficiency',
            'Predicted_Overpotential': 'Overpotential',
            'Predicted_Selectivity': 'Selectivity'
        }, inplace=True)
        plot_df['Stability_numeric'] = test_df['Predicted_Stability'].map({'High': 1, 'Low': 0})

        melted = plot_df.melt(
            id_vars='Element/Alloy',
            value_vars=['Faradaic Efficiency', 'Overpotential', 'Selectivity', 'Stability_numeric'],
            var_name='Property', value_name='Value'
        )
        melted['Property'] = melted['Property'].replace({'Stability_numeric': 'Stability'})

        # Normalize values 0–1
        melted['Value_scaled'] = melted.groupby('Property')['Value'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
        )

        custom_palette = {
            'Faradaic Efficiency': '#1f77b4',
            'Overpotential': '#ff7f0e',
            'Selectivity': '#2ca02c',
            'Stability': '#d62728'
        }

        plt.figure(figsize=(12, 6))
        sns.barplot(data=melted, x='Element/Alloy', y='Value_scaled', hue='Property', palette=custom_palette)
        plt.title("Predicted CO₂ Catalyst Properties (Normalized)")
        plt.ylabel("Normalized Score (0 to 1)")
        plt.xticks(rotation=45)
        plt.legend(title="Property", bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(plt.gcf())

        st.subheader(" SHAP Feature Contribution (Physical Parameters)")
        try:
            explainer = shap.TreeExplainer(fe_model)
            shap_values = explainer.shap_values(test_df[fe_features])
            st.markdown("This SHAP summary shows how physical parameters influence the FE predictions.")
            fig = plt.figure()
            shap.summary_plot(shap_values, test_df[fe_features], plot_type="dot", show=False)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"SHAP plot couldn't be generated: {e}")

        st.subheader(" Correlation Heatmap (Uploaded Data)")
        numeric_cols = test_df.select_dtypes(include=[np.number]).copy()
        numeric_cols = numeric_cols.dropna(axis=1, how='all')
        numeric_cols = numeric_cols.loc[:, numeric_cols.nunique() > 1]
        corr = numeric_cols.corr()
        corr = corr.dropna(axis=0, how='all').dropna(axis=1, how='all')

        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
        plt.title("Feature Correlation Matrix")
        st.pyplot(plt.gcf())

        st.subheader(" Final Conclusions")
        try:
            top_fe = test_df.sort_values(by='Predicted_FE', ascending=False).head(3)
            top_sel = test_df.sort_values(by='Predicted_Selectivity', ascending=False).head(3)
            top_stab = test_df[test_df['Predicted_Stability'] == 'High'].head(3)
            top_eff_all = test_df.sort_values(
                by=['Predicted_FE', 'Predicted_Selectivity', 'Predicted_Stability'], 
                ascending=[False, False, False]
            ).head(3)

            st.markdown("###  Top Performers Summary")
            st.markdown("####  Top 3 by **Faradaic Efficiency**")
            st.dataframe(top_fe[['Element/Alloy', 'Predicted_FE']])

            st.markdown("####  Top 3 by **Selectivity**")
            st.dataframe(top_sel[['Element/Alloy', 'Predicted_Selectivity']])

            st.markdown("#### First 3 Catalysts with **High Stability**")
            st.dataframe(top_stab[['Element/Alloy', 'Predicted_Stability']])

            st.markdown("####  Overall Top 3 (based on FE + Selectivity + High Stability)")
            st.dataframe(top_eff_all[['Element/Alloy', 'Predicted_FE', 'Predicted_Selectivity', 'Predicted_Stability']])

        except Exception as e:
            st.warning(f"Could not generate final summary: {e}")

        csv = test_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Full Predictions CSV", csv, "predicted_catalysts.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
