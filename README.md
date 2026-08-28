# ML-personal-project
Used FSI 2023 Publicly available data to make a ML model that clusters countries based on the danger level. 
PROGRAM FragileStatesAnalysis

    IMPORT pandas AS pd
    IMPORT numpy AS np
    IMPORT StandardScaler FROM sklearn.preprocessing
    IMPORT KMeans FROM sklearn.cluster
    IMPORT silhouette_score, accuracy_score, classification_report FROM sklearn.metrics
    IMPORT RandomForestClassifier FROM sklearn.ensemble
    IMPORT train_test_split FROM sklearn.model_selection

    df = pd.read_excel("FSI-2023-DOWNLOAD.xlsx")

    features = [
        "C1: Security Apparatus", "C2: Factionalized Elites", "C3: Group Grievance",
        "E1: Economy", "E2: Economic Inequality", "E3: Human Flight and Brain Drain",
        "P1: State Legitimacy", "P2: Public Services", "P3: Human Rights",
        "S1: Demographic Pressures", "S2: Refugees and IDPs", "X1: External Intervention"
    ]

    X = df[features]

    scaler = INSTANTIATE StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = INSTANTIATE KMeans(n_clusters=4, random_state=42, n_init=10)
    df["Risk_Cluster"] = kmeans.fit_predict(X_scaled)

    sil_score = silhouette_score(X_scaled, df["Risk_Cluster"])

    OUTPUT "=================================================="
    OUTPUT " UNSUPERVISED CLUSTERING METRIC"
    OUTPUT " Silhouette Score: ", sil_score, " (High separation quality)"
    OUTPUT "==================================================\n"

    bins = [0, 30, 60, 90, 120]
    labels = ["Sustainable", "Stable", "Warning", "Alert"]
    df["Risk_Category"] = pd.cut(df["Total"], bins=bins, labels=labels)

    y = df["Risk_Category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y
    )

    classifier = INSTANTIATE RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        random_state=42
    )
    CALL classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    OUTPUT "=================================================="
    OUTPUT " REGIONAL RISK CLASSIFIER ACCURACY: ", (accuracy * 100), "%"
    OUTPUT "==================================================\n"
    OUTPUT "Classification Breakdown:"
    OUTPUT classification_report(y_test, y_pred)

END PROGRAM
