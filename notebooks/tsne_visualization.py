import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def main():
    # Dosya yollarını tanımla
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", "csv", "mfcc_features.csv")
    output_path = os.path.join(base_dir, "notebooks", "tsne_plot.png")

    print(f"[{csv_path}] dosyası okunuyor...")
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Hata: {csv_path} bulunamadı! Önce veri setini oluşturduğunuzdan emin olun.")
        return

    # Etiketleri ve özellikleri ayır
    # 'label' ve 'filepath' (eğer varsa) sütunları dışındaki her şey özelliktir
    columns_to_drop = ['label', 'filepath']
    features_df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    labels = df['label']

    print(f"Toplam {len(df)} örnek ve {features_df.shape[1]} özellik bulundu.")

    # Veriyi ölçeklendir (StandardScaler)
    print("Özellikler standartlaştırılıyor (Z-score normalization)...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)

    # t-SNE Uygula
    print("t-SNE işlemi başlatılıyor (bu işlem birkaç dakika sürebilir)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(scaled_features)

    # DataFrame'e at
    tsne_df = pd.DataFrame(data=tsne_results, columns=['tsne_1', 'tsne_2'])
    tsne_df['label'] = labels

    # Görselleştir
    print("Görsel oluşturuluyor...")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='tsne_1', 
        y='tsne_2',
        hue='label',
        palette=sns.color_palette("husl", len(labels.unique())),
        data=tsne_df,
        legend="full",
        alpha=0.7
    )
    plt.title('t-SNE Visualization of MFCC Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Kaydet
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Görsel başarıyla kaydedildi: {output_path}")

if __name__ == "__main__":
    main()
