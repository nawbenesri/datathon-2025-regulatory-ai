

# 🧠 Datathon 2025 – Analyse IA de l’Impact Réglementaire

## 🎯 Objectif du Projet

Ce projet a été développé dans le cadre du **Datathon POLYFINANCES 2025**.
Il vise à concevoir un outil d’analyse basé sur l’**IA générative** pour évaluer l’impact des réglementations financières sur les portefeuilles d’actions, notamment le **S&P 500**.

---

## 📊 Contexte

Les marchés financiers sont aujourd’hui profondément influencés par :

- Un cadre réglementaire de plus en plus complexe et changeant
- Des politiques économiques protectionnistes
- Des sanctions et restrictions internationales

Ces facteurs redéfinissent la gestion d’actifs et exigent des **outils d’aide à la décision intelligents, rapides et explicables**.

---

## ✨ Fonctionnalités Principales

### 1. 🧾 Analyse Automatique de Textes Réglementaires

- Extraction automatique des éléments clés : entités, secteurs, dates, mesures, lois citées.
- Combinaison de **NLP** et d’**IA générative**.
- Compatibilité avec plusieurs formats : lois, rapports, documents 10-K, PDF ou HTML.

### 2. 📈 Évaluation de l’Impact

- Calcul de **scores de risque** par entreprise.
- Analyse des **expositions sectorielles et géographiques**.
- Estimation des **pertes potentielles** (% et valeur).
- Explication détaillée du raisonnement sous-jacent.

### 3. 🧩 Recommandations Stratégiques

- Simulation de **scénarios alternatifs**.
- Identification des zones de **concentration du risque**.
- Suggestions d’ajustements concrets :

  - Réallocation ou rotation sectorielle
  - Remplacement de titres
  - Ajustement géographique des expositions

### 4. 💻 Interface Web Interactive

- Visualisation intuitive de l’exposition du portefeuille.
- Tableaux et graphiques dynamiques.
- Présentation claire et pédagogique des ajustements proposés.

---

## 🛠️ Technologies Utilisées

- **IA Générative & NLP** : Analyse et extraction d’informations réglementaires
- **Python** : Langage principal de développement
- **Streamlit** : Interface web interactive
- **AWS Services** : Traitement et hébergement cloud
- **Pandas / Plotly** : Analyse et visualisation des données

---

## 📂 Données

### Données Fournies

- `sp500_composition_2025-08-15.csv` : Composition du S&P 500 (tickers, poids, prix)
- `stocks-performance_2025-09-26.csv` : Performances financières (market cap, EPS, FCF, etc.)

### Sources Externes Autorisées

- [SEC EDGAR](https://www.sec.gov/edgar/search/) — Rapports 10-K / 10-Q
- [Yahoo Finance](https://finance.yahoo.com/) — Données de marché
- [Morningstar](https://www.morningstar.com/) — Analyses financières

---

## 📁 Structure du Projet

```
datathon-2025-regulatory-ai/
│
├── data/                # Données brutes et traitées
├── notebooks/           # Notebooks d'analyse exploratoire
├── src/                 # Code source principal
│   ├── extraction/      # Modules d'extraction de texte
│   ├── analysis/        # Modules d'analyse et de scoring
│   ├── recommendations/ # Génération de recommandations
│   └── web/             # Interface web (Streamlit)
├── tests/               # Tests unitaires
├── docs/                # Documentation technique
├── requirements.txt     # Dépendances Python
└── README.md            # Ce fichier
```

---

## 🚀 Installation

```bash
# Cloner le dépôt
git clone https://github.com/Omar-Zed/datathon-2025-regulatory-ai.git
cd datathon-2025-regulatory-ai

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

---

## 🧠 Utilisation

```python
from src.analysis import RegulatoryAnalyzer

# Initialiser l’analyseur
analyzer = RegulatoryAnalyzer()

# Analyser un document réglementaire
results = analyzer.analyze_document("path/to/document.pdf")

# Générer des recommandations
recommendations = analyzer.generate_recommendations(results)
```

---

## 🏆 Critères d’Évaluation

| Catégorie                   | Pondération | Détails                            |
| --------------------------- | ----------- | ---------------------------------- |
| Extraction d’informations   | 20%         | Pertinence et précision du NLP     |
| Scoring & impact            | 20%         | Cohérence des scores et calculs    |
| Recommandations             | 20%         | Qualité et valeur ajoutée          |
| Interface utilisateur       | 15%         | UX, lisibilité, interactivité      |
| Storytelling & présentation | 25%         | Clarté du message et démonstration |

---

## 📅 Chronologie du Datathon

| Étape                   | Description                                           |
| ----------------------- | ----------------------------------------------------- |
| **Vendredi / Samedi**   | Exploration des données, conception de l’architecture |
| **Dimanche matin**      | Réception du document réglementaire complémentaire    |
| **Dimanche après-midi** | Finalisation, test et préparation de la présentation  |

---

## ⚠️ Points Clés

- **Optimisation AWS** : Tester d’abord sur un échantillon réduit.
- **Cache des résultats** : Minimiser les appels API répétés.
- **Flexibilité** : Support de formats variés (PDF, HTML, DOCX).
- **Transparence** : Justifier chaque recommandation avec des explications claires.

---

## 👥 Équipe 13

> Benesrighe Nawal
> Zedek Mohammed Omar
> Jaafri Hayani Rita
> Talbe Sara

---

## 📝 Licence

Projet développé dans le cadre du **Datathon POLYFINANCES 2025**.
Usage académique et démonstratif uniquement.

---

## 🔗 Liens Utiles

- [Site officiel POLYFINANCES](https://polyfinances.ca)
- [SEC EDGAR Database](https://www.sec.gov/edgar/search/)
- [S&P 500 Overview](https://www.spglobal.com/spdji/en/indices/equity/sp-500/)

---

**Datathon POLYFINANCES 2025** — Transformer la complexité réglementaire en opportunités d’analyse et de décision.

---

Souhaites-tu que je t’en fasse une **version markdown stylisée** (avec emojis, encadrés de code colorés et tableau de résumé du pipeline IA) pour le GitHub final ?
