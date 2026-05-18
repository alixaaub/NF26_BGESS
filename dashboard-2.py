from pathlib import Path
import dash
from dash import Dash, Input, Output, html, dcc, dash_table
import pandas as pd
import plotly.graph_objects as go

# On importe la classe Spark de requêtes
from req import DataWarehouseQueries

# ── Chemins & Constantes ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
WAREHOUSE_DIR = ROOT / "export" / "warehouse"

# Liste des questions disponibles
QUESTIONS_DICT = {
    f"q{i}": f"Question {i}" for i in range(1, 21)
}

# ── Fonctions Utilitaires ────────────────────────────────────────────────────
def get_available_dates():
    """Liste les dossiers disponibles dans l'entrepôt."""
    if not WAREHOUSE_DIR.exists():
        return ["latest"]
    folders = [p.name for p in WAREHOUSE_DIR.iterdir() if p.is_dir()]
    # S'assurer que 'latest' est en haut de liste
    if "latest" in folders:
        folders.remove("latest")
        return ["latest"] + sorted(folders, reverse=True)
    return sorted(folders, reverse=True)

# ── Application Dash ─────────────────────────────────────────────────────────
app = Dash(__name__, title="DataWarehouse BGES - Requêtes Spark")

app.layout = html.Div([
    html.H1("Explorateur de Requêtes DataWarehouse (Spark)", style={"textAlign": "center"}),
    
    # Zone de contrôle (Filtres)
    html.Div(style={"display": "flex", "gap": "20px", "padding": "20px", "background": "#f8f9fa", "borderRadius": "8px"}, children=[
        html.Div(style={"flex": 1}, children=[
            html.Label("Sélectionnez la date (Dossier Warehouse) :", style={"fontWeight": "bold"}),
            dcc.Dropdown(
                id="dropdown-date",
                options=[{"label": d, "value": d} for d in get_available_dates()],
                value="latest",
                clearable=False
            )
        ]),
        html.Div(style={"flex": 3}, children=[
            html.Label("Sélectionnez la/les question(s) à afficher :", style={"fontWeight": "bold"}),
            dcc.Dropdown(
                id="dropdown-questions",
                options=[{"label": f"{k} - {v}", "value": k} for k, v in QUESTIONS_DICT.items()],
                multi=True,
                placeholder="Choisissez une ou plusieurs requêtes..."
            )
        ])
    ]),
    
    # Zone d'affichage des résultats dynamiques
    html.Div(id="results-container", style={"padding": "20px", "marginTop": "20px"})
])

# ── Callbacks ────────────────────────────────────────────────────────────────
@app.callback(
    Output("results-container", "children"),
    Input("dropdown-date", "value"),
    Input("dropdown-questions", "value")
)
def update_results(selected_date, selected_questions):
    if not selected_questions:
        return html.Div("Veuillez sélectionner au moins une question.", style={"color": "gray", "fontStyle": "italic"})

    # 1. Initialisation de la session Spark sur le dossier choisi
    try:
        # On instancie la classe de req.py
        queries_engine = DataWarehouseQueries(date_folder=selected_date)
    except Exception as e:
        return html.Div(f"Erreur lors du chargement des données Spark : {str(e)}", style={"color": "red"})

    # 2. Construction dynamique de la page
    components = []
    for q_name in selected_questions:
        # On récupère dynamiquement la méthode (q1, q2, etc.) de l'objet
        if hasattr(queries_engine, q_name):
            func = getattr(queries_engine, q_name)
            
            try:
                result = func() # Exécution de la requête Spark
                
                # En-tête de la question
                components.append(html.H3(f"Résultat : {q_name.upper()}", style={"borderBottom": "2px solid #007bff"}))
                
                # A. Si le résultat est un DataFrame Pandas (Tableau/Valeur)
                if isinstance(result, pd.DataFrame):
                    components.append(dash_table.DataTable(
                        data=result.to_dict('records'),
                        columns=[{"name": i, "id": i} for i in result.columns],
                        style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white', 'fontWeight': 'bold'},
                        style_data={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'},
                        style_table={'overflowX': 'auto', 'marginBottom': '40px'}
                    ))
                    
                # B. Si le résultat est un Graphique Plotly (Figure)
                elif isinstance(result, go.Figure):
                    components.append(dcc.Graph(
                        figure=result,
                        style={'marginBottom': '40px'}
                    ))
                    
                else:
                    components.append(html.Div("Format de réponse non supporté.", style={"color": "red"}))
                    
            except Exception as e:
                components.append(html.Div(f"Erreur lors de l'exécution de {q_name} : {str(e)}", style={"color": "red"}))
                
    return components

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)