# app.py - Fichier principal de l'application Streamlit
import streamlit as st
from pathlib import Path
from data_loader import load_sp500_composition, load_stocks_performance, merge_sp500_data
from reg_analysis import extract_reg_info
from analyze_reg_impact_enhanced import analyze_reg_impact_enhanced
import plotly.express as px
import pandas as pd
from collections import Counter

st.set_page_config(page_title="RegAI Portfolio Analyzer", layout="wide")
st.title("Team 37")
# Menu latéral pour les étapes
st.sidebar.title("Étapes de l'Application")
steps = [
    "1. Chargement des Données S&P 500",
    "2. Upload et Extraction du Texte Réglementaire",
    "3. Modélisation de l'Impact",
    "4. Évaluation Globale du Portefeuille",
    "5. Visualisations et Recommandations"
]
selected_step = st.sidebar.selectbox("Sélectionnez une étape", steps)

# Uploaders dans le sidebar pour persistance
DEFAULT_DATA_DIR = Path(__file__).parent / "jeu_de_donnees"
if 'data_dir' not in st.session_state:
    st.session_state['data_dir'] = str(DEFAULT_DATA_DIR)

st.sidebar.header("Jeu de Données")
data_dir_input = st.sidebar.text_input("Répertoire jeu_de_donnees", st.session_state['data_dir'])
st.session_state['data_dir'] = data_dir_input.strip() or st.session_state['data_dir']
data_dir_path = Path(st.session_state['data_dir'])

if st.sidebar.button("Charger les fichiers par défaut"):
    comp_path = data_dir_path / "2025-08-15_composition_sp500.csv"
    perf_path = data_dir_path / "2025-09-26_stocks-performance.csv"
    if comp_path.exists() and perf_path.exists():
        df_comp = load_sp500_composition(comp_path)
        df_perf = load_stocks_performance(perf_path)
        portfolio_df = merge_sp500_data(df_comp, df_perf)
        if portfolio_df is not None:
            st.session_state['portfolio_df'] = portfolio_df
            st.sidebar.success("Données chargées depuis le jeu de données.")
        else:
            st.sidebar.error("Fusion impossible : vérifiez les fichiers.")
    else:
        st.sidebar.error("Impossible de trouver les CSV dans le répertoire indiqué.")

st.sidebar.header("Chargement des Fichiers CSV (upload manuel)")
comp_file = st.sidebar.file_uploader("composition_sp500.csv", type=['csv'], key="comp_uploader")
perf_file = st.sidebar.file_uploader("stocks-performance.csv", type=['csv'], key="perf_uploader")

# Chargement automatique si fichiers uploadés et DF pas en session
if comp_file and perf_file and 'portfolio_df' not in st.session_state:
    df_comp = load_sp500_composition(comp_file)
    df_perf = load_stocks_performance(perf_file)
    portfolio_df = merge_sp500_data(df_comp, df_perf)
    if portfolio_df is not None:
        st.session_state['portfolio_df'] = portfolio_df
        st.sidebar.success("Données chargées avec succès !")

# Récupération du DF depuis session
portfolio_df = st.session_state.get('portfolio_df', None)

# Bouton pour recharger si besoin
if st.sidebar.button("Recharger les Données CSV"):
    if comp_file and perf_file:
        df_comp = load_sp500_composition(comp_file)
        df_perf = load_stocks_performance(perf_file)
        portfolio_df = merge_sp500_data(df_comp, df_perf)
        st.session_state['portfolio_df'] = portfolio_df
        st.sidebar.success("Données rechargées !")
    else:
        st.sidebar.error("Veuillez uploader les fichiers d'abord.")

# Étape 1: Chargement des Données S&P 500
if selected_step == "1. Chargement des Données S&P 500":
    st.header("Étape 1: Chargement et Aperçu des Données S&P 500")
    st.write("Utilisez les uploaders dans le sidebar pour charger les fichiers. Les données persistent via session_state.")
    if portfolio_df is not None:
        st.dataframe(portfolio_df.head(10))  # Afficher top 10
        st.write(f"Nombre total d'actions: {len(portfolio_df)}")
        st.write("Exemple de métriques dérivées:")
        st.dataframe(portfolio_df[['Symbol', 'Op. Margin', 'Net Margin']].head())
    else:
        st.warning("Veuillez uploader les fichiers dans le sidebar.")

# Étape 2: Upload et Extraction du Texte Réglementaire (updated for multiple files)
if selected_step == "2. Upload et Extraction du Texte Réglementaire":
    st.header("Étape 2: Upload du Document Réglementaire et Extraction des Informations")
    if portfolio_df is None:
        st.warning("Veuillez d'abord charger les données CSV via le sidebar.")
    else:
        uploaded_files = st.file_uploader("Téléchargez des fichiers texte réglementaires (TXT, HTML, XML)", type=['txt', 'html', 'xml', 'htm'], accept_multiple_files=True)
        
        reg_texts = []
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_extension = uploaded_file.name.split('.')[-1]
                reg_text = uploaded_file.read().decode('utf-8', errors='ignore')
                reg_texts.append((reg_text, file_extension, uploaded_file.name))
        
        # Optional text area for manual input
        manual_text = st.text_area("Ou collez le texte réglementaire ici (exemple par défaut fourni)", 
                                   value="mettre le json par exemple")
        if manual_text:
            reg_texts.append((manual_text, 'txt', 'Manual Input'))
        
        if st.button("Extraire les Informations"):
            all_extracted = []
            combined_text = ''
            for reg_text, ext, name in reg_texts:
                extracted = extract_reg_info(reg_text, ext)
                all_extracted.append((name, extracted))
                combined_text += reg_text + ' '  # For combined analysis
            
            st.session_state['all_extracted'] = all_extracted
            st.session_state['reg_texts'] = reg_texts  # Store list
            
            # Display per file
            # --- Affichage Amélioré des Résultats ---


            st.markdown("## 📑 Résumé des Textes Réglementaires Analysés")

            for name, extracted in all_extracted:
                with st.expander(f"📘 {name}", expanded=False):
                    # --- Ligne 1 : résumé rapide ---
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Entités détectées", len(extracted['entities']))
                    col2.metric("Périodes mentionnées", len(extracted['dates']))
                    col3.metric("Thèmes / Mesures", len(extracted['measures']))

                    st.markdown(f"**Type de Réglementation :** `{extracted['type_reg']}`")

                    # --- Entités principales ---
                    st.markdown("### 🏛️ Entités Principales")
                    entities_preview = sorted(list(extracted['entities']))[:15]
                    st.write(", ".join(entities_preview) + (" ..." if len(extracted['entities']) > 15 else ""))

                    # --- Thèmes clés / mots-clés ---
                    st.markdown("### 🧩 Thèmes / Mots-clés Dominants")
                    common_measures = Counter(extracted["measures"]).most_common(10)
                    if common_measures:
                        df_common = pd.DataFrame(common_measures, columns=["Mot-clé", "Occurrences"])
                        fig = px.bar(df_common, x="Mot-clé", y="Occurrences", title="Top 10 Thèmes Détectés", height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Aucun mot-clé dominant détecté.")

                    # --- Périodes clés ---
                    st.markdown("### 📅 Périodes Mentionnées")
                    dates_preview = sorted(list(extracted['dates']))[:10]
                    st.write(", ".join(dates_preview) + (" ..." if len(extracted['dates']) > 10 else ""))

                    # --- Synthèse automatique ---
                    st.markdown("### 🧠 Synthèse")
                    st.info(
                        f"La directive contient **{len(extracted['entities'])} entités**, "
                        f"**{len(extracted['dates'])} références temporelles** "
                        f"et **{len(extracted['measures'])} thèmes économiques**."
                    )

# Étape 3: Modélisation de l'Impact
if selected_step == "3. Modélisation de l'Impact":
    st.header("Étape 3: Modélisation de l'Impact Réglementaire sur les Actions")
    if portfolio_df is None:
        st.warning("Veuillez d'abord charger les données CSV via le sidebar.")
    elif 'reg_texts' not in st.session_state:
        st.warning("Veuillez d'abord extraire les informations à l'étape 2.")
    else:
        default_fillings = st.session_state.get(
            'fillings_dir',
            str((data_dir_path / "fillings").resolve())
        )
        fillings_dir = st.text_input("Dossier des rapports 10-K (fillings)", value=default_fillings)
        use_bedrock = st.checkbox("Activer l'analyse Bedrock (si disponible)", value=st.session_state.get('use_bedrock', False))

        if st.button("Modéliser l'Impact"):
            # Combined text for analysis
            combined_text = ' '.join([text for text, _, _ in st.session_state['reg_texts']])
            st.session_state['fillings_dir'] = fillings_dir
            st.session_state['use_bedrock'] = use_bedrock

            with st.spinner("Analyse réglementaire avancée en cours..."):
                analyzed_df, extracted, portfolio_risk, concentration, recommendations = analyze_reg_impact_enhanced(
                    portfolio_df.copy(),
                    combined_text,
                    fillings_dir=fillings_dir,
                    file_extension='txt',
                    use_bedrock=use_bedrock
                )

            st.session_state['analyzed_df'] = analyzed_df
            st.session_state['extracted_combined'] = extracted  # For later use
            st.session_state['portfolio_risk'] = portfolio_risk
            st.session_state['concentration'] = concentration
            st.session_state['recommendations'] = recommendations
            bedrock_status = extracted.get('bedrock_status', {})
            st.session_state['bedrock_status'] = bedrock_status

            if bedrock_status and not bedrock_status.get('enabled', False) and bedrock_status.get('error'):
                st.warning(f"Analyse Bedrock indisponible: {bedrock_status['error']}. Basculé en mode heuristique.")

            st.subheader("Scores de Risque par Action (Top 10 impactées):")
            high_risk = analyzed_df[analyzed_df['Risk Score'] > 0].sort_values('Risk Score', ascending=False).head(10)
            cols = ['Symbol', 'Company', 'Risk Score', 'Direct Risk', 'Supply Chain Risk', 'Geographic Risk', 'Impact Est. Loss %', 'Impact Est. Loss']
            existing_cols = [c for c in cols if c in high_risk.columns]
            st.dataframe(high_risk[existing_cols])

# Étape 4: Évaluation Globale du Portefeuille
if selected_step == "4. Évaluation Globale du Portefeuille":
    st.header("Étape 4: Évaluation de l'Effet Global sur le Portefeuille S&P 500")
    if portfolio_df is None:
        st.warning("Veuillez d'abord charger les données CSV via le sidebar.")
    elif 'analyzed_df' not in st.session_state:
        st.warning("Veuillez d'abord modéliser l'impact à l'étape 3.")
    else:
        analyzed_df = st.session_state['analyzed_df']
        portfolio_risk = st.session_state.get('portfolio_risk')
        concentration = st.session_state.get('concentration')
        if portfolio_risk is None or concentration is None:
            st.warning("Aucun résultat stocké. Relancez la modélisation à l'étape 3.")
        else:
            st.subheader("Risque Global du Portefeuille:")
            st.write(f"**Score de Risque Agrégé:** {portfolio_risk:.4f}")
            st.subheader("Concentrations de Risque par Secteur:")
            st.dataframe(concentration)

# Étape 5: Visualisations et Recommandations
if selected_step == "5. Visualisations et Recommandations":
    st.header("Étape 5: Visualisations, Simulations et Recommandations")
    if portfolio_df is None:
        st.warning("Veuillez d'abord charger les données CSV via le sidebar.")
    elif 'analyzed_df' not in st.session_state:
        st.warning("Veuillez d'abord modéliser l'impact à l'étape 3.")
    else:
        analyzed_df = st.session_state['analyzed_df']
        recommendations = st.session_state.get('recommendations', [])
        
        # Visualisation
        st.subheader("Visualisation des Risques")
        fig = px.bar(analyzed_df[analyzed_df['Risk Score'] > 0].sort_values('Risk Score', ascending=False).head(20),
                     x='Symbol', y='Risk Score', title="Top 20 Actions à Risque")
        st.plotly_chart(fig)
        
        # Simulations
        st.subheader("Simulations de Scénarios")
        scenario = st.selectbox("Choisissez un scénario", ["Base", "High Impact (+20%)", "Low Impact (-20%)"])
        impact_loss = analyzed_df['Impact Est. Loss'].sum() / 1e12
        if scenario == "High Impact (+20%)":
            impact_loss *= 1.2
        elif scenario == "Low Impact (-20%)":
            impact_loss *= 0.8
        st.write(f"Perte Estimée Totale ({scenario}): {impact_loss:.2f} T$")
        
        # Recommandations
        st.subheader("Recommandations Stratégiques")
        if not recommendations:
            st.info("Aucune recommandation disponible. Relancez l'analyse pour en générer.")
        elif isinstance(recommendations[0], dict):
            for rec in recommendations:
                header = f"{rec.get('action', 'ACTION')} - {rec.get('ticker', 'Portefeuille')}"
                with st.expander(header):
                    st.write(f"**Entreprise:** {rec.get('company', 'N/A')}")
                    st.write(f"**Score de risque:** {rec.get('risk_score', 'N/A')}")
                    st.write(f"**Recommandation:** {rec.get('recommendation', rec.get('reason', 'N/A'))}")
                    if rec.get('current_weight'):
                        st.write(f"**Poids actuel:** {rec['current_weight']}")
                    if rec.get('estimated_loss'):
                        st.write(f"**Perte estimée:** {rec['estimated_loss']}")
                    if rec.get('urgency') or rec.get('urgence'):
                        st.write(f"**Urgence:** {rec.get('urgency', rec.get('urgence'))}")
                    extras = {k: v for k, v in rec.items() if k not in {'action', 'ticker', 'company', 'risk_score', 'recommendation', 'reason', 'current_weight', 'estimated_loss', 'urgency', 'urgence'}}
                    if extras:
                        st.json(extras)
        else:
            for rec in recommendations:
                st.write(f"- {rec}")
