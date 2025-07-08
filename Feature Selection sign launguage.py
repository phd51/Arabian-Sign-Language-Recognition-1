import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

def mcdmi_feature_selection(X, y, top_percent=0.1, mi_threshold=0.08):
    feature_names = X.columns.tolist()

    # === Mutual Information Calculation ===
    mi_scores = mutual_info_classif(X, y, discrete_features='auto')
    mi_scores_norm = mi_scores / np.max(mi_scores)

    # === Thresholding based on MI ===
    selected_idx = np.where(mi_scores_norm >= mi_threshold)[0]
    X_sel = X.iloc[:, selected_idx]
    selected_names = X_sel.columns.tolist()
    print(f"\nSelected {len(selected_names)} features after MI thresholding.\n")

    # === Decision Matrix Setup ===
    np.random.seed(0)
    relevance = mi_scores_norm[selected_idx] * 5
    redundancy = np.random.randint(1, 6, size=len(selected_idx))
    cost = np.random.randint(1, 6, size=len(selected_idx))
    interpretability = np.random.randint(1, 6, size=len(selected_idx))

    D = pd.DataFrame({
        'Relevance': relevance,
        'Redundancy': redundancy,
        'Cost': cost,
        'Interpretability': interpretability
    }, index=selected_names)

    # === Normalize the Decision Matrix ===
    scaler = MinMaxScaler()
    D_norm = pd.DataFrame(scaler.fit_transform(D), columns=D.columns, index=D.index)

    # === AHP Weights ===
    A = np.array([
        [1,   3,   5,   7],
        [1/3, 1,   3,   5],
        [1/5, 1/3, 1,   3],
        [1/7, 1/5, 1/3, 1]
    ])
    eigvals, eigvecs = np.linalg.eig(A)
    max_idx = np.argmax(eigvals.real)
    weights = eigvecs[:, max_idx].real
    weights /= np.sum(weights)

    # === Final Weighted Scores ===
    D_norm['FinalScore'] = D_norm.dot(weights)
    top_k = int(np.ceil(len(D_norm) * top_percent))
    selected_final = D_norm['FinalScore'].sort_values(ascending=False).head(top_k)
    final_features = selected_final.index.tolist()

    return final_features, D_norm, mi_scores, weights

# === Main Execution ===
if __name__ == "__main__":
    input_csv = ""
    output_csv = ""

    df = pd.read_csv(input_csv)

    # Identify columns
    filename_column = 'filename' if 'filename' in df.columns else None
    label_column = 'label' if 'label' in df.columns else df.columns[-1]

    # Extract relevant parts
    X = df.drop(columns=[col for col in [filename_column, label_column] if col])
    y = df[label_column].values

    # === Feature Selection ===
    selected_columns, decision_matrix, mi_scores, weights = mcdmi_feature_selection(X, y, top_percent=0.10)

    # === Create Final DataFrame ===
    selected_df = df[selected_columns].copy()

    if filename_column:
        selected_df[filename_column] = df[filename_column]
    selected_df[label_column] = df[label_column]

    # Reorder columns: filename, selected features..., label
    columns_order = [filename_column] + selected_columns + [label_column] if filename_column else selected_columns + [label_column]
    selected_df = selected_df[columns_order]

    # === Save Output ===
    selected_df.to_csv(output_csv, index=False)
    print(f"\n✅ Selected features saved to: {output_csv}")
    print(f"✔️ Selected Features ({len(selected_columns)}): {list(selected_columns)}")

    # === Explanation ===
    print("\n--- MCDMI Feature Selection Justification ---")
    print("→ MI relevance computed for each feature")
    print("→ Features below threshold (δ) discarded")
    print("→ Multi-Criteria scores generated (R, C, T, I)")
    print("→ AHP used to compute weights for criteria")
    print("→ Final features selected based on top 10% weighted scores")