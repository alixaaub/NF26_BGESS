import pandas as pd
import pyspark.pandas as ps
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, date_format, lower, datediff
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

class DataWarehouseQueries():

    def __init__(self, date_folder="latest"):
        self.spark = SparkSession.builder.getOrCreate()
        self.spark.conf.set("spark.sql.session.timeZone", "CET")  

        # Rendre le chemin dynamique basé sur le répertoire du script
        base_path = Path(__file__).resolve().parent / "export" / "warehouse" / date_folder

        # Chargement des fichiers
        psdf_dim_materiel = ps.read_csv(str(base_path / "DIM_MATERIEL.csv"))
        psdf_dim_mission = ps.read_csv(str(base_path / "DIM_MISSION.csv"))
        psdf_fait_materiel = ps.read_csv(str(base_path / "FAIT_MATERIEL.csv"))
        psdf_fait_mission = ps.read_csv(str(base_path / "FAIT_MISSION.csv"))

        if date_folder == "latest":
            psdf_dim_personnel = ps.read_csv(str(base_path / "DIM_PERSONNEL.csv"))
        else:
            psdf_dim_personnel = ps.read_csv(str(Path(__file__).resolve().parent / "export" / "warehouse" / "init" / "DIM_PERSONNEL.csv"))

        # Création des vues SQL temporaires
        psdf_dim_materiel.to_spark().createOrReplaceTempView("DIM_MATERIEL")
        psdf_dim_personnel.to_spark().createOrReplaceTempView("DIM_PERSONNEL")
        psdf_dim_mission.to_spark().createOrReplaceTempView("DIM_MISSION")
        psdf_fait_materiel.to_spark().createOrReplaceTempView("FAIT_MATERIEL")
        psdf_fait_mission.to_spark().createOrReplaceTempView("FAIT_MISSION")

    def _sauvegarder_graphique(self, fig, nom_fichier):
        """
        Sauvegarde un graphique Plotly au format PDF. 
        """
        export_dir = Path(__file__).resolve().parent / "export" / "figures"
        export_dir.mkdir(parents=True, exist_ok=True) # Crée le dossier s'il n'existe pas
        
        chemin_complet = export_dir / f"{nom_fichier}.pdf"
        fig.write_image(str(chemin_complet), format="pdf")
        print(f"[INFO] Graphique enregistré : {chemin_complet}")

    def q1(self):
        """1. Combien de cadres travaillent sur le site de Paris ?"""
        return self.spark.sql("""
            SELECT COUNT(DISTINCT ID_PERSONNEL) as nb_cadres
            FROM DIM_PERSONNEL
            WHERE FONCTION_PERSONNEL ='Cadre'
              AND LOWER(ID_SITE) = 'paris'
        """).toPandas()

    def q2(self):
        """2. Combien d'ingénieurs Data travaillent sur les sites aux États-Unis ?"""
        return self.spark.sql("""
            SELECT COUNT(DISTINCT ID_PERSONNEL) as nb_data_engineers
            FROM DIM_PERSONNEL
            WHERE LOWER(FONCTION_PERSONNEL) = 'ingénieur data'
              AND LOWER(ID_SITE) IN ('newyork', 'la') 
        """).toPandas()

    def q3(self):
        """3. Combien d'ingénieurs informaticiens travaillent dans l'organisation ?"""
        return self.spark.sql("""
            SELECT COUNT(DISTINCT ID_PERSONNEL) as nb_ingenieurs_info
            FROM DIM_PERSONNEL
            WHERE LOWER(FONCTION_PERSONNEL) = 'ingénieur informaticien'
        """).toPandas()

    def q4(self):
        """4. Combien de PC fixes ont été achetés entre juin et septembre 2026 ?"""
        return self.spark.sql("""
            SELECT COUNT(f.ID_MATERIEL) as nb_pc_fixes
            FROM FAIT_MATERIEL f
            JOIN DIM_MATERIEL m ON f.ID_MATERIEL = m.ID_MATERIEL
            WHERE LOWER(m.TYPE) LIKE 'pc fixe%'
              AND f.ID_DATE_ACHAT BETWEEN '2026-06-01' AND '2026-09-30'
        """).toPandas()

    def q5(self):
        """5. Impact carbone des PC fixes (sans écran) entre mai et octobre 2026 ?"""
        return self.spark.sql("""
            SELECT (SUM(m.impact) / 1000) as impact_total_pc_fixes_tonnes
            FROM FAIT_MATERIEL f
            JOIN DIM_MATERIEL m ON f.ID_MATERIEL = m.ID_MATERIEL
            WHERE LOWER(m.TYPE) = 'pc fixe sans ecran' 
              AND f.ID_DATE_ACHAT BETWEEN '2026-05-01' AND '2026-10-31'
        """).toPandas()

    def q6(self):
        """6. Impact des PC portables des Data Engineers (mai-oct 2026, Londres/NY)."""
        return self.spark.sql("""
            SELECT (SUM(m.impact) / 1000) as impact_pc_portables_tonnes
            FROM FAIT_MATERIEL f
            JOIN DIM_MATERIEL m ON f.ID_MATERIEL = m.ID_MATERIEL
            JOIN DIM_PERSONNEL p ON f.ID_PERSONNEL = p.ID_PERSONNEL
            WHERE LOWER(m.TYPE) = 'pc portable'
              AND LOWER(p.FONCTION_PERSONNEL) = 'ingénieur data'
              AND LOWER(f.ID_SITE) IN ('london', 'newyork')
              AND f.ID_DATE_ACHAT BETWEEN '2026-05-01' AND '2026-10-31'
        """).toPandas()

    def q7(self):
        """7. Impact des écrans achetés par les cadres (juil-sept 2026)."""
        return self.spark.sql("""
            SELECT (SUM(m.impact) / 1000) as impact_ecrans_tonnes
            FROM FAIT_MATERIEL f
            JOIN DIM_MATERIEL m ON f.ID_MATERIEL = m.ID_MATERIEL
            JOIN DIM_PERSONNEL p ON f.ID_PERSONNEL = p.ID_PERSONNEL
            WHERE LOWER(m.TYPE) = 'ecran'
              AND LOWER(p.FONCTION_PERSONNEL) = 'cadre'
              AND f.ID_DATE_ACHAT BETWEEN '2026-07-01' AND '2026-09-30'
        """).toPandas()

    def q8(self):
        """8. Impact carbone des missions sur les sites Européens (mai-oct 2026)."""
        return self.spark.sql("""
            SELECT (SUM(m.impact) / 1000) as impact_missions_europe_tonnes
            FROM FAIT_MISSION f
            JOIN DIM_MISSION m ON f.id_mission = m.ID_MISSION
            WHERE LOWER(f.ID_SITE) IN ('paris', 'london', 'berlin')
              AND f.ID_DATE_MISSION BETWEEN '2026-05-01' AND '2026-10-31'
        """).toPandas()

    def q9(self):
        """9. Les 5 jours les plus impactants en avion pour les sites Européens."""
        return self.spark.sql("""
            SELECT f.ID_DATE_MISSION, (SUM(m.impact) / 1000) as impact_journalier_tonnes
            FROM FAIT_MISSION f
            JOIN DIM_MISSION m ON f.id_mission = m.ID_MISSION
            WHERE LOWER(m.TRANSPORT) = 'avion'
              AND LOWER(f.ID_SITE) IN ('paris', 'london', 'berlin')
            GROUP BY f.ID_DATE_MISSION
            ORDER BY impact_journalier_tonnes DESC
            LIMIT 5
        """).toPandas()

    def q10(self):
        """10. Secteur d'activité le plus impactant (Missions + Matériel)."""
        return self.spark.sql("""
            WITH Emissions_Globales AS (
                SELECT p.FONCTION_PERSONNEL, m.impact
                FROM FAIT_MISSION f
                JOIN DIM_MISSION m ON f.id_mission = m.ID_MISSION
                JOIN DIM_PERSONNEL p ON f.ID_PERSONNEL = p.ID_PERSONNEL
                UNION ALL
                SELECT p.FONCTION_PERSONNEL, mat.impact
                FROM FAIT_MATERIEL f2
                JOIN DIM_MATERIEL mat ON f2.ID_MATERIEL = mat.ID_MATERIEL
                JOIN DIM_PERSONNEL p ON f2.ID_PERSONNEL = p.ID_PERSONNEL
            )
            SELECT FONCTION_PERSONNEL, (SUM(impact) / 1000) as impact_total_tonnes
            FROM Emissions_Globales
            GROUP BY FONCTION_PERSONNEL
            ORDER BY impact_total_tonnes DESC
            LIMIT 1
        """).toPandas()

    def q11(self):
        """11. Quel site a eu le plus d'impact (Missions + Matériel) ?"""
        return self.spark.sql("""
            WITH Impact_Sites AS (
                SELECT f.ID_SITE, m.impact FROM FAIT_MISSION f
                JOIN DIM_MISSION m ON f.id_mission = m.ID_MISSION
                UNION ALL
                SELECT f2.ID_SITE, mat.impact FROM FAIT_MATERIEL f2
                JOIN DIM_MATERIEL mat ON f2.ID_MATERIEL = mat.ID_MATERIEL
            )
            SELECT ID_SITE, (SUM(impact) / 1000) as impact_total_tonnes
            FROM Impact_Sites
            GROUP BY ID_SITE
            ORDER BY impact_total_tonnes DESC
            LIMIT 1
        """).toPandas()

    def q12(self):
        """12. Impact carbone des missions reliant deux sites de l'organisation en sept 2026."""
        return self.spark.sql("""
            SELECT (SUM(m.impact) / 1000) as impact_inter_sites_tonnes
            FROM FAIT_MISSION f
            JOIN DIM_MISSION m ON f.id_mission = m.ID_MISSION
            WHERE f.ID_DATE_MISSION BETWEEN '2026-09-01' AND '2026-09-30'
              AND LOWER(m.VILLE_DESTINATION) IN (
                  SELECT DISTINCT LOWER(ID_SITE) FROM DIM_PERSONNEL
              )
        """).toPandas()

    def q13(self):
        """13. Impact des conférences en juillet 2026 pour les employés de Los Angeles."""
        return self.spark.sql("""
            SELECT (SUM(m.impact) / 1000) as impact_conferences_tonnes
            FROM FAIT_MISSION f
            JOIN DIM_MISSION m ON f.id_mission = m.ID_MISSION
            WHERE LOWER(m.TYPE_MISSION) = 'conférence'
              AND LOWER(f.ID_SITE) = 'la'
              AND f.ID_DATE_MISSION BETWEEN '2026-07-01' AND '2026-07-31'
        """).toPandas()

    def q14(self):
        """14. Secteur d'activité le plus impactant pour les conférences (mai-sept)."""
        return self.spark.sql("""
            SELECT p.FONCTION_PERSONNEL, (SUM(m.impact) / 1000) as impact_total_tonnes
            FROM FAIT_MISSION f
            JOIN DIM_MISSION m ON f.id_mission = m.ID_MISSION
            JOIN DIM_PERSONNEL p ON f.ID_PERSONNEL = p.ID_PERSONNEL
            WHERE LOWER(m.TYPE_MISSION) = 'conférence'
              AND f.ID_DATE_MISSION BETWEEN '2026-05-01' AND '2026-09-30'
            GROUP BY p.FONCTION_PERSONNEL
            ORDER BY impact_total_tonnes DESC
            LIMIT 1
        """).toPandas()

    def q15(self):
        """15. Âge moyen des Ingénieurs Data partis en formation (juil-sept)."""

        #datediff permet de passer de la date de naissance à l'age
        return self.spark.sql("""
            SELECT AVG(datediff(f.ID_DATE_MISSION, p.DT_NAISS) / 365.25) as age_moyen
            FROM FAIT_MISSION f
            JOIN DIM_MISSION m ON f.id_mission = m.ID_MISSION
            JOIN DIM_PERSONNEL p ON f.ID_PERSONNEL = p.ID_PERSONNEL
            WHERE LOWER(p.FONCTION_PERSONNEL) = 'ingénieur data'
              AND LOWER(m.TYPE_MISSION) = 'formation'
              AND f.ID_DATE_MISSION BETWEEN '2026-07-01' AND '2026-09-30'
        """).toPandas()

    def q16(self):
        """16. Destination la plus impactante en cumulé (mai-oct 2026)."""
        return self.spark.sql("""
            SELECT m.VILLE_DESTINATION, (SUM(m.impact) / 1000) as impact_cumule_tonnes
            FROM FAIT_MISSION f
            JOIN DIM_MISSION m ON f.id_mission = m.ID_MISSION
            WHERE f.ID_DATE_MISSION BETWEEN '2026-05-01' AND '2026-10-31'
            GROUP BY m.VILLE_DESTINATION
            ORDER BY impact_cumule_tonnes DESC
            LIMIT 1
        """).toPandas()

    def q17(self):
        """17. Les 3 catégories de missions les plus impactantes pour les cadres en Europe (mai 2026)."""
        return self.spark.sql("""
            SELECT m.TYPE_MISSION, (SUM(m.impact) / 1000) as impact_total_tonnes
            FROM FAIT_MISSION f
            JOIN DIM_MISSION m ON f.id_mission = m.ID_MISSION
            JOIN DIM_PERSONNEL p ON f.ID_PERSONNEL = p.ID_PERSONNEL
            WHERE LOWER(p.FONCTION_PERSONNEL) = 'cadre'
              AND LOWER(f.ID_SITE) IN ('paris', 'london', 'berlin')
              AND f.ID_DATE_MISSION BETWEEN '2026-05-01' AND '2026-05-31'
            GROUP BY m.TYPE_MISSION
            ORDER BY impact_total_tonnes DESC
            LIMIT 3
        """).toPandas()

    def q18(self):
        """18. Quelles ont été les 5 missions les plus impactantes sur le site de Paris ?"""
        df = self.spark.sql("""
            SELECT m.ID_MISSION, m.TYPE_MISSION, m.VILLE_DESTINATION, (SUM(m.impact) / 1000) as impact_total_tonnes
            FROM FAIT_MISSION f
            JOIN DIM_MISSION m ON f.id_mission = m.ID_MISSION
            WHERE LOWER(f.id_site) = 'paris'
            GROUP BY m.ID_MISSION, m.TYPE_MISSION, m.VILLE_DESTINATION
            ORDER BY impact_total_tonnes DESC
            LIMIT 5
        """).toPandas()
        
        df['Label'] = df['ID_MISSION'].astype(str) + " (" + df['TYPE_MISSION'] + " -> " + df['VILLE_DESTINATION'] + ")"
        
        fig = px.bar(df, x="impact_total_tonnes", y="Label", orientation='h', 
                     title="Top 5 des missions les plus impactantes - Site de Paris",
                     labels={"impact_total_tonnes": "Impact (Tonnes CO2)", "Label": "Mission"})
                     
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        
        # Ajustement des axes si les valeurs sont très proches
        min_val, max_val = df['impact_total_tonnes'].min(), df['impact_total_tonnes'].max()
        if (max_val - min_val) < (max_val * 0.1):
            fig.update_xaxes(range=[min_val * 0.9, max_val * 1.05])

        self._sauvegarder_graphique(fig, "q18_top_5_missions_paris")
        return fig

    def q19(self):
        """19. Figure comparant l'impact carbone mensuel des missions par transport et site."""
        df = self.spark.sql("""
            SELECT 
                DATE_FORMAT(f.ID_DATE_MISSION, 'yyyy-MM') as Mois,
                m.TRANSPORT,
                f.id_site as Site,
                (SUM(m.impact) / 1000) as impact_total_tonnes
            FROM FAIT_MISSION f
            JOIN DIM_MISSION m ON f.id_mission = m.ID_MISSION
            GROUP BY Mois, TRANSPORT, Site
            ORDER BY Mois
        """).toPandas()

        # Création d'une grille
        fig = px.line(df, x="Mois", y="impact_total_tonnes", color="TRANSPORT", facet_row="Site",
                      title="Impact carbone mensuel par transport et par site",
                      markers=True)
        
        # Details d'affichage
        fig.update_yaxes(matches=None, title_text="")
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

        # Ajout d'un seul titre global centré
        fig.add_annotation(
            x=-0.1,                  
            y=0.5,                    
            text="Impact (Tonnes CO2)", 
            textangle=-90,            
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=14)
        )
        
        self._sauvegarder_graphique(fig, "q19_impact_mensuel_transport_site")
        return fig

    def q20(self):
        """20. Figure illustrant l'impact carbone global mensuel de l'organisation."""
        df = self.spark.sql("""
            WITH Missions AS (
                SELECT DATE_FORMAT(f.ID_DATE_MISSION, 'yyyy-MM') as Mois, 'Missions' as Categorie, (SUM(m.impact) / 1000) as impact_tonnes
                FROM FAIT_MISSION f 
                JOIN DIM_MISSION m ON f.id_mission = m.ID_MISSION
                GROUP BY Mois
            ),
            Materiel AS (
                SELECT DATE_FORMAT(f2.ID_DATE_ACHAT, 'yyyy-MM') as Mois, 'Matériel Informatique' as Categorie, (SUM(mat.impact) / 1000) as impact_tonnes
                FROM FAIT_MATERIEL f2 
                JOIN DIM_MATERIEL mat ON f2.ID_MATERIEL = mat.ID_MATERIEL
                GROUP BY Mois
            )
            SELECT * FROM Missions
            UNION ALL
            SELECT * FROM Materiel
            ORDER BY Mois
        """).toPandas()

        # Diagramme en barres empilées
        fig = px.bar(df, x="Mois", y="impact_tonnes", color="Categorie",
                     title="Impact Carbone Global Mensuel de l'Organisation",
                     labels={"impact_tonnes": "Impact Global (Tonnes CO2)"},
                     barmode="stack")
                     
        self._sauvegarder_graphique(fig, "q20_impact_carbone_global_mensuel")
        return fig