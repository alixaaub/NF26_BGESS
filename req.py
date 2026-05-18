import pandas as pd
import pyspark.pandas as ps
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, date_format, lower
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
        psdf_dim_personnel = ps.read_csv(str(base_path / "DIM_PERSONNEL.csv"))
        psdf_dim_mission = ps.read_csv(str(base_path / "DIM_MISSION.csv"))
        psdf_fait_materiel = ps.read_csv(str(base_path / "FAIT_MATERIEL.csv"))
        psdf_fait_mission = ps.read_csv(str(base_path / "FAIT_MISSION.csv"))

        # Création des vues SQL temporaires
        psdf_dim_materiel.to_spark().createOrReplaceTempView("DIM_MATERIEL")
        psdf_dim_personnel.to_spark().createOrReplaceTempView("DIM_PERSONNEL")
        psdf_dim_mission.to_spark().createOrReplaceTempView("DIM_MISSION")
        psdf_fait_materiel.to_spark().createOrReplaceTempView("FAIT_MATERIEL")
        psdf_fait_mission.to_spark().createOrReplaceTempView("FAIT_MISSION")


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
            SELECT SUM(m.impact) as impact_total_pc_fixes
            FROM FAIT_MATERIEL f
            JOIN DIM_MATERIEL m ON f.ID_MATERIEL = m.ID_MATERIEL
            WHERE LOWER(m.TYPE) = 'pc fixe sans ecran' 
              AND f.ID_DATE_ACHAT BETWEEN '2026-05-01' AND '2026-10-31'
        """).toPandas()

    def q6(self):
        """6. Impact des PC portables des Data Engineers (mai-oct 2026, Londres/NY)."""
        return self.spark.sql("""
            SELECT SUM(m.impact) as impact_pc_portables
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
            SELECT SUM(m.impact) as impact_ecrans
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
            SELECT SUM(m.impact) as impact_missions_europe
            FROM FAIT_MISSION f
            JOIN DIM_MISSION m ON f.id_mission = m.ID_MISSION
            WHERE LOWER(f.ID_SITE) IN ('paris', 'london', 'berlin')
              AND f.ID_DATE_MISSION BETWEEN '2026-05-01' AND '2026-10-31'
        """).toPandas()

    def q9(self):
        """9. Les 5 jours les plus impactants en avion pour les sites Européens."""
        return self.spark.sql("""
            SELECT f.ID_DATE_MISSION, SUM(m.impact) as impact_journalier
            FROM FAIT_MISSION f
            JOIN DIM_MISSION m ON f.id_mission = m.ID_MISSION
            WHERE LOWER(m.TRANSPORT) = 'avion'
              AND LOWER(f.ID_SITE) IN ('paris', 'london', 'berlin')
            GROUP BY f.ID_DATE_MISSION
            ORDER BY impact_journalier DESC
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
            SELECT FONCTION_PERSONNEL, SUM(impact) as impact_total
            FROM Emissions_Globales
            GROUP BY FONCTION_PERSONNEL
            ORDER BY impact_total DESC
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
            SELECT ID_SITE, SUM(impact) as impact_total
            FROM Impact_Sites
            GROUP BY ID_SITE
            ORDER BY impact_total DESC
            LIMIT 1
        """).toPandas()

    def q12(self):
        """12. Impact carbone des missions reliant deux sites de l'organisation en sept 2026."""
        return self.spark.sql("""
            SELECT SUM(m.impact) as impact_inter_sites
            FROM FAIT_MISSION f
            JOIN DIM_MISSION m ON f.id_mission = m.ID_MISSION
            WHERE f.ID_DATE_MISSION BETWEEN '2026-09-01' AND '2026-09-30'
              -- Le départ est le site de l'employé (vérifié par f.ID_SITE ou m.VILLE_DEPART)
              -- L'arrivée doit correspondre à un site existant dans l'entreprise
              AND LOWER(m.VILLE_DESTINATION) IN (
                  SELECT DISTINCT LOWER(ID_SITE) FROM DIM_PERSONNEL
              )
        """).toPandas()

    def q13(self):
        """13. Impact des conférences en juillet 2026 pour les employés de Los Angeles."""
        return self.spark.sql("""
            SELECT SUM(m.impact) as impact_conferences
            FROM FAIT_MISSION f
            JOIN DIM_MISSION m ON f.id_mission = m.ID_MISSION
            WHERE LOWER(m.TYPE_MISSION) = 'conférence'
              AND LOWER(f.ID_SITE) = 'la'
              AND f.ID_DATE_MISSION BETWEEN '2026-07-01' AND '2026-07-31'
        """).toPandas()

    def q14(self):
        """14. Secteur d'activité le plus impactant pour les conférences (mai-sept)."""
        return self.spark.sql("""
            SELECT p.SECTEUR_ACTIVITE, SUM(m.impact) as impact_total
            FROM FAIT_MISSION f
            JOIN DIM_MISSION m ON f.id_mission = m.ID_MISSION
            JOIN DIM_PERSONNEL p ON f.ID_PERSONNEL = p.ID_PERSONNEL
            WHERE LOWER(m.TYPE_MISSION) = 'conférence'
              AND f.ID_DATE_MISSION BETWEEN '2026-05-01' AND '2026-09-30'
            GROUP BY p.SECTEUR_ACTIVITE
            ORDER BY impact_total DESC
            LIMIT 1
        """).toPandas()

    def q15(self):
        """15. Âge moyen des Ingénieurs Data partis en formation (juil-sept)."""
        return self.spark.sql("""
            SELECT AVG(p.AGE) as age_moyen
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
            SELECT m.VILLE_DESTINATION, SUM(m.impact) as impact_cumule
            FROM FAIT_MISSION f
            JOIN DIM_MISSION m ON f.id_mission = m.ID_MISSION
            WHERE f.ID_DATE_MISSION BETWEEN '2026-05-01' AND '2026-10-31'
            GROUP BY m.VILLE_DESTINATION
            ORDER BY impact_cumule DESC
            LIMIT 1
        """).toPandas()

    def q17(self):
        """17. Les 3 catégories de missions les plus impactantes pour les cadres en Europe (mai 2026)."""
        return self.spark.sql("""
            SELECT m.TYPE_MISSION, SUM(m.impact) as impact_total
            FROM FAIT_MISSION f
            JOIN DIM_MISSION m ON f.id_mission = m.ID_MISSION
            JOIN DIM_PERSONNEL p ON f.ID_PERSONNEL = p.ID_PERSONNEL
            WHERE LOWER(p.FONCTION_PERSONNEL) = 'cadre'
              AND LOWER(f.ID_SITE) IN ('paris', 'london', 'berlin')
              AND f.ID_DATE_MISSION BETWEEN '2026-05-01' AND '2026-05-31'
            GROUP BY m.TYPE_MISSION
            ORDER BY impact_total DESC
            LIMIT 3
        """).toPandas()

    def q18(self):
        """
        18. Quelles ont été les 5 missions les plus impactantes sur le site de Paris ?
        """
        # Requête via l'API DataFrame
        df = self.spark.sql("""
            SELECT m.ID_MISSION, m.TYPE_MISSION, m.VILLE_DESTINATION, SUM(f.impact) as impact_total
            FROM FAIT_MISSION f
            JOIN DIM_MISSION m ON f.id_mission = m.ID_MISSION
            WHERE LOWER(f.id_site) = 'paris'
            GROUP BY m.ID_MISSION, m.TYPE_MISSION, m.VILLE_DESTINATION
            ORDER BY impact_total DESC
            LIMIT 5
        """).toPandas()
        
        # Création du format d'étiquette
        df['Label'] = df['ID_MISSION'].astype(str) + " (" + df['TYPE_MISSION'] + " -> " + df['VILLE_DESTINATION'] + ")"
        
        # Figure Plotly renvoyée directement à Dash
        fig = px.bar(df, x="impact_total", y="Label", orientation='h', 
                     title="Top 5 des missions les plus impactantes - Site de Paris",
                     labels={"impact_total": "Impact Carbone (kg CO2)", "Label": "Mission"})
        fig.update_layout(yaxis={'categoryorder':'total ascending'}) # Impact le plus haut en haut
        return fig

    """
    def q19_impact_transport_site(self, sdf_fait_mission, sdf_dim_mission):
        
        19. Proposer une figure comparant l’impact carbone mensuel des missions 
        en fonction du type de transport et sur chaque site.
        

        # Agrégation PySpark : Extraction Année-Mois, regroupement par Mois, Transport et Site
        result = sdf_fait_mission.join(sdf_dim_mission, sdf_fait_mission.id_mission == sdf_dim_mission.ID_MISSION) \
            .withColumn("Mois", date_format(col("ID_DATE_MISSION"), "yyyy-MM")) \
            .groupBy("Mois", "TRANSPORT", "id_site") \
            .agg(sum("impact").alias("impact_total")) \
            .orderBy("Mois") \
            .collect()
            
        # Extraction des entités uniques pour structurer la figure dynamique
        sites = sorted(list(set([r['id_site'] for r in result])))
        
        # Création d'une figure multi-courbes (1 sous-graphique vertical par Site)
        fig, axes = plt.subplots(len(sites), 1, figsize=(12, 4 * len(sites)), sharex=True)
        if len(sites) == 1: 
            axes = [axes]
            
        for i, site in enumerate(sites):
            ax = axes[i]
            # Filtrer les lignes collectées pour le site concerné
            data_site = [r for r in result if r['id_site'] == site]
            transports_site = list(set([r['TRANSPORT'] for r in data_site]))
            
            # Tracer une ligne chronologique par type de transport
            for transport in transports_site:
                mois_t = [r['Mois'] for r in data_site if r['TRANSPORT'] == transport]
                impact_t = [r['impact_total'] for r in data_site if r['TRANSPORT'] == transport]
                ax.plot(mois_t, impact_t, marker='o', linewidth=2, label=transport)
                
            ax.set_title(f"Impact Carbone Mensuel - Site : {site.upper()}", fontsize=12, fontweight='bold')
            ax.set_ylabel("CO2 Émis")
            ax.legend(title="Transport")
            ax.grid(True, linestyle=':', alpha=0.6)
            
        plt.xlabel("Période (Mois)")
        fig.suptitle("Comparaison de l'impact carbone mensuel par transport et par site", y=1.01, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Sauvegarde PDF et affichage
        self._afficher_et_sauvegarder(fig, "q19_impact_mensuel_transport_site")


    def q20_impact_global_mensuel(self, sdf_fait_mission, sdf_dim_mission, sdf_fait_materiel, sdf_dim_materiel):
        
        20. Proposer une figure illustrant l’impact carbone global mensuel de l’organisation.
        
        # 1. Total mensuel pour les Missions
        df_missions = sdf_fait_mission.join(sdf_dim_mission, sdf_fait_mission.id_mission == sdf_dim_mission.ID_MISSION) \
            .withColumn("Mois", date_format(col("ID_DATE_MISSION"), "yyyy-MM")) \
            .groupBy("Mois") \
            .agg(sum("impact").alias("impact_missions"))
            
        # 2. Total mensuel pour le Matériel Informatique
        df_materiel = sdf_fait_materiel.join(sdf_dim_materiel, sdf_fait_materiel.ID_MATERIEL == sdf_dim_materiel.ID_MATERIEL) \
            .withColumn("Mois", date_format(col("ID_DATE_ACHAT"), "yyyy-MM")) \
            .groupBy("Mois") \
            .agg(sum("impact").alias("impact_materiel"))
            
        # 3. Jointure externe (Outer Join) pour coupler Missions et Matériels sur tous les mois disponibles
        df_global = df_missions.join(df_materiel, on="Mois", how="outer") \
            .na.fill(0) \
            .orderBy("Mois") \
            .collect()
            
        # Extraction
        liste_mois = [r['Mois'] for r in df_global]
        vals_missions = [r['impact_missions'] for r in df_global]
        vals_materiel = [r['impact_materiel'] for r in df_global]
        
        # Construction de la figure (Histogramme empilé pour visualiser la part de chaque pôle)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Barre de base : Missions
        ax.bar(liste_mois, vals_missions, label="Missions Professionnelles", color='#2ecc71')
        # Barre empilée : Matériels (commence là où s'arrêtent les missions)
        ax.bar(liste_mois, vals_materiel, bottom=vals_missions, label="Achat Matériel Informatique", color='#e74c3c')
        
        ax.set_xlabel("Mois")
        ax.set_ylabel("Impact Carbone Global (kg CO2)")
        ax.set_title("Impact Carbone Global Mensuel de l'Organisation (Missions + Matériels)", fontsize=14, fontweight='bold')
        ax.legend(loc="upper left")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Sauvegarde PDF et affichage
        self._afficher_et_sauvegarder(fig, "q20_impact_carbone_global_mensuel")
    
    """

