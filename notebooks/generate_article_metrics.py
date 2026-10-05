import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def load_dataset(csv_path: Path):
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c.startswith("f_")]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = (df["label"] == "positive").astype(int).to_numpy()
    return X, y, feature_cols

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = Path(os.path.join(base_dir, "data", "csv", "mfcc_features.csv"))
    model_path = Path(os.path.join(base_dir, "models", "wakeword_model.joblib"))
    
    # 1. Load Data
    print(f"[{csv_path}] dosyası okunuyor...")
    X, y, feature_cols = load_dataset(csv_path)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Tablo Raporu
    print("\n=============================================")
    print("        DATASET SPLIT TABLE (ARTICLES)       ")
    print("=============================================")
    print(f"{'Class':<15} | {'Train (80%)':<15} | {'Test (20%)':<15} | {'Total':<15}")
    print("-" * 65)
    
    train_pos = np.sum(y_train == 1)
    train_neg = np.sum(y_train == 0)
    test_pos = np.sum(y_test == 1)
    test_neg = np.sum(y_test == 0)
    
    print(f"{'Positive (1)':<15} | {train_pos:<15} | {test_pos:<15} | {train_pos+test_pos:<15}")
    print(f"{'Negative (0)':<15} | {train_neg:<15} | {test_neg:<15} | {train_neg+test_neg:<15}")
    print(f"{'Total':<15} | {len(y_train):<15} | {len(y_test):<15} | {len(y):<15}")
    print("=============================================\n")

    # 2. Load Model
    print(f"Model yükleniyor: {model_path}")
    if model_path.exists():
        model_data = joblib.load(model_path)
        best_model = model_data["model"]
    else:
        print("Model bulunamadı, script varsayılan bir pipeline ile devam edecek.")
        best_model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, C=10.0, gamma='scale'))
        ])

    print("Test verisi üzerinde tahminler alınıyor...")
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # 3. Confusion Matrix
    print("Confusion Matrix görseli oluşturuluyor...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    cm_path = os.path.join(base_dir, "notebooks", "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"--> Kaydedildi: {cm_path}")

    # 4. ROC and PR Curves
    print("ROC ve Precision-Recall Eğrileri oluşturuluyor...")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    precisions, recalls, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recalls, precisions)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0].set_xlim([-0.01, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Receiver Operating Characteristic (ROC)')
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # PR
    axes[1].plot(recalls, precisions, color='purple', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)

    roc_pr_path = os.path.join(base_dir, "notebooks", "roc_pr_curves.png")
    plt.savefig(roc_pr_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"--> Kaydedildi: {roc_pr_path}")

    # 5. Learning Curve
    print("Learning Curve (Öğrenme Eğrisi) hesaplanıyor (Bu işlem uzun sürebilir)...")
    
    # Hızlı olması için probability=False ile yeni bir model kopyası oluşturuyoruz
    lc_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=False, C=10.0, gamma='scale'))
    ])
    
    # Daha az nokta deneyerek işlemi hızlandırıyoruz
    train_sizes_fraction = np.linspace(0.3, 1.0, 4)
    train_sizes, train_scores, test_scores = learning_curve(
        lc_model, X, y, cv=3, n_jobs=1, 
        train_sizes=train_sizes_fraction, scoring='accuracy', random_state=42
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.title('Learning Curve - SVM')
    plt.xlabel('Training Examples')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    lc_path = os.path.join(base_dir, "notebooks", "learning_curve.png")
    plt.savefig(lc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"--> Kaydedildi: {lc_path}")
    print("=============================================")
    print("Tüm metrikler ve makale grafikleri tamamlandı!")

if __name__ == "__main__":
    main()
