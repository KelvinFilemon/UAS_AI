"""
================================================================================
PROYEK UAS ARTIFICIAL INTELLIGENCE (AI)
Judul  : Analisis Segmentasi & Klasifikasi Akun Top Instagram Menggunakan
         K-Means Clustering dan Random Forest Classifier
Dataset: Top Instagram Accounts (latest.csv & daily.csv)
Nama   : Kelvin Filemon 
NIM    : 32250077
Nama   : Vio Lefrans
NIM    : 322500
================================================================================
"""

import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats

# Library Scikit-Learn untuk Preprocessing, Clustering, Klasifikasi & Evaluasi
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

def load_and_preprocess_data(latest_path="latest.csv", daily_path="daily.csv"):
    """
    Memuat dataset latest.csv dan daily.csv, melakukan pembersihan anomali,
    penanganan missing value, penggabungan metrik pertumbuhan (growth),
    serta feature engineering.
    """
    print("[1/5] Memuat dataset dan melakukan Data Preprocessing...")

    try:
        df_latest = pd.read_csv(latest_path)
        df_daily = pd.read_csv(daily_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"File dataset tidak ditemukan: {e}. Pastikan latest.csv dan daily.csv berada di folder yang sama."
        )

    initial_rows = len(df_latest)
    print(f"      - Jumlah baris awal (latest.csv): {initial_rows:,} akun")


    id_col = "username" if "username" in df_latest.columns else df_latest.columns[0]
    df_latest = df_latest.drop_duplicates(subset=[id_col]).copy()

    df_clean = df_latest[df_latest["followers"] > 0].copy()
    removed_anomalies = initial_rows - len(df_clean)
    print(f"      - Anomali 0 followers dihapus   : {removed_anomalies} akun")
    print(f"      - Total akun valid bersih       : {len(df_clean):,} akun")

    for col in ["category", "country", "type"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna("Other").astype(str).str.strip()
            df_clean[col] = df_clean[col].replace({"": "Other", "nan": "Other", "None": "Other"})

    if {"username", "followers"}.issubset(df_daily.columns):
        daily_agg = (
            df_daily.groupby("username")["followers"]
            .agg(min_f="first", max_f="last")
            .reset_index()
        )
        daily_agg["growth_30d_pct"] = np.where(
            daily_agg["min_f"] > 0,
            ((daily_agg["max_f"] - daily_agg["min_f"]) / daily_agg["min_f"]) * 100.0,
            0.0,
        )
        df_clean = pd.merge(
            df_clean, daily_agg[["username", "growth_30d_pct"]], on="username", how="left"
        )
        df_clean["growth_30d_pct"] = df_clean["growth_30d_pct"].fillna(0.0)
    elif "growth_30d_pct" not in df_clean.columns:
        df_clean["growth_30d_pct"] = 0.0

    # 1. Transformasi Log10 pada followers untuk meredam skewness distribusi ekstrem
    df_clean["log_followers"] = np.log10(df_clean["followers"])

    # 2. Rank berdasarkan jumlah followers
    df_clean["rank"] = df_clean["followers"].rank(ascending=False, method="min").astype(int)

    # 3. Panjang karakter nama akun & jumlah negara target (jika dipisah koma)
    name_col = "name" if "name" in df_clean.columns else id_col
    df_clean["name_length"] = df_clean[name_col].fillna("").astype(str).str.len()
    df_clean["country_count"] = (
        df_clean["country"]
        .astype(str)
        .apply(lambda x: 0 if x in ["Other", "Unknown"] else len([c for c in x.split(",") if c.strip()]))
    )

    # 4. Encoding binary untuk tipe akun (Person = 1, Organization = 0)
    if "type" in df_clean.columns:
        df_clean["is_person"] = (
            df_clean["type"].str.lower().isin(["person", "individual", "creator"]).astype(int)
        )
    else:
        df_clean["is_person"] = 0

    return df_clean


def run_correlation_analysis(df):
    """
    Melakukan uji korelasi parametrik (Pearson) dan non-parametrik (Spearman)
    untuk menguji hubungan antar variabel utama.
    """
    print("\n[2/5] Menjalankan Uji Korelasi Statistik (Pearson & Spearman)...")

    pairs = [
        ("log_followers", "rank", "Log10 Followers vs Rank"),
        ("followers", "growth_30d_pct", "Followers vs 30-Day Growth (%)"),
        ("is_person", "followers", "Account Type (Person=1) vs Followers"),
        ("country_count", "log_followers", "Jumlah Negara Target vs Log10 Followers"),
    ]

    results = []
    for col_x, col_y, label in pairs:
        if col_x in df.columns and col_y in df.columns:
            valid = df[[col_x, col_y]].dropna()
            r_pearson, p_pearson = stats.pearsonr(valid[col_x], valid[col_y])
            rho_spearman, p_spearman = stats.spearmanr(valid[col_x], valid[col_y])

            results.append(
                {
                    "pair": label,
                    "pearson_r": round(float(r_pearson), 4),
                    "pearson_p": float(p_pearson),
                    "spearman_rho": round(float(rho_spearman), 4),
                    "spearman_p": float(p_spearman),
                }
            )
            print(
                f"      * {label:40s} | Pearson r = {r_pearson:+.4f} | Spearman rho = {rho_spearman:+.4f} (p={p_spearman:.2e})"
            )

    return results


def run_kmeans_clustering(df, n_clusters=4):
    """
    Melakukan segmentasi akun top Instagram menggunakan K-Means Clustering (k=4)
    dan mengevaluasi kualitas klaster dengan Silhouette Score.
    """
    print(f"\n[3/5] Melatih AI Model 1: K-Means Clustering (k={n_clusters})...")

    cluster_features = ["log_followers", "growth_30d_pct", "country_count", "name_length"]
    X_cluster = df[cluster_features].fillna(0)

    # Standarisasi fitur menggunakan Z-Score (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # Melatih model K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # Menghitung Silhouette Score (menggunakan sampel jika data > 10.000 agar cepat)
    sil_score = silhouette_score(X_scaled, df["cluster"], sample_size=min(5000, len(df)), random_state=42)
    print(f"      - Silhouette Score (k={n_clusters}): {sil_score:.4f}")

    # Ringkasan profil tiap klaster
    cluster_summary = (
        df.groupby("cluster")
        .agg(
            total_accounts=("followers", "count"),
            avg_followers=("followers", "mean"),
            avg_growth_pct=("growth_30d_pct", "mean"),
            avg_countries=("country_count", "mean"),
        )
        .reset_index()
    )

    for _, row in cluster_summary.iterrows():
        print(
            f"      - Cluster {int(row['cluster'])}: {int(row['total_accounts']):,} akun | "
            f"Avg Followers: {row['avg_followers']:,.0f} | Avg Growth: {row['avg_growth_pct']:+.2f}%"
        )

    return df, kmeans, scaler, round(float(sil_score), 4), cluster_summary


# ==============================================================================
# 4. AI MODEL 2: RANDOM FOREST CLASSIFIER (SUPERVISED LEARNING)
# ==============================================================================
def run_random_forest_classifier(df):
    """
    Melatih Random Forest Classifier untuk memprediksi tipe akun
    (Person vs Organization) berdasarkan metrik popularitas dan profil.
    """
    print("\n[4/5] Melatih AI Model 2: Random Forest Classifier (Person vs Organization)...")

    le_cat = LabelEncoder()
    df["category_encoded"] = le_cat.fit_transform(df["category"].astype(str))

    feature_cols = ["log_followers", "rank", "category_encoded", "name_length", "country_count"]
    X = df[feature_cols].fillna(0)
    y = df["is_person"]

    # Split data 80% Training, 20% Testing dengan stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"      - Akurasi Testing Random Forest: {acc * 100:.2f}%")
    print("\n      Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Organization (0)", "Person (1)"]))

    # Feature Importance
    importances = dict(zip(feature_cols, [round(float(val) * 100, 2) for val in rf_model.feature_importances_]))
    print("      Feature Importance (%):", importances)

    return rf_model, round(float(acc) * 100, 2), importances


# ==============================================================================
# 5. EKSPOR HASIL KE JSON UNTUK INTEGRASI DASHBOARD UI/UX
# ==============================================================================
def export_dashboard_summary(df, corr_results, sil_score, cluster_summary, rf_acc, importances):
    print("\n[5/5] Menyimpan ringkasan hasil analisis ke 'dashboard_insights.json'...")

    summary = {
        "total_valid_accounts": int(len(df)),
        "kmeans_silhouette": sil_score,
        "random_forest_accuracy": rf_acc,
        "correlations": corr_results,
        "clusters": cluster_summary.to_dict(orient="records"),
        "feature_importance": importances,
    }

    with open("dashboard_insights.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("      - Selesai! File 'dashboard_insights.json' siap digunakan.")


if __name__ == "__main__":
    print("=" * 72)
    print("MEMULAI PIPELINE DATA SCIENCE & AI MODELING — TOP INSTAGRAM ACCOUNTS")
    print("=" * 72)

    df_processed = load_and_preprocess_data("latest.csv", "daily.csv")
    corr_res = run_correlation_analysis(df_processed)
    df_clustered, kmeans_model, scaler_model, silhouette, cluster_df = run_kmeans_clustering(df_processed, n_clusters=4)
    rf_clf, rf_accuracy, feat_imp = run_random_forest_classifier(df_clustered)
    export_dashboard_summary(df_clustered, corr_res, silhouette, cluster_df, rf_accuracy, feat_imp)