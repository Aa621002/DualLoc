#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import os
from tqdm import tqdm
import torch

class Visualizer:
    """視覺化工具類別"""
    
    def __init__(self, config):
        self.config = config
        
    def extract_features(self, model, data_loader):
        """提取特徵"""
        model.eval()
        features_list = []
        labels_list = []

        with torch.no_grad():
            for data in tqdm(data_loader, desc="提取特徵"):
                ids = data['input_ids'].to(self.config.device, dtype=torch.long)
                mask = data['attention_mask'].to(self.config.device, dtype=torch.long)
                targets = data['targets'].to(self.config.device, dtype=torch.float)

                outputs = model(ids, mask)
                features = outputs.cpu().numpy()
                labels = targets.cpu().numpy()

                features_list.append(features)
                labels_list.append(labels)

        features = np.vstack(features_list)
        labels = np.vstack(labels_list)

        return features, labels

    def plot_training_history(self, history, save_path=None):
        """繪製訓練歷史"""
        plt.figure(figsize=(12, 5))
        
        # 準確率圖
        plt.subplot(1, 2, 1)
        plt.plot(history['train_acc'], label='訓練準確率')
        plt.plot(history['test_acc'], label='驗證準確率')
        plt.title('訓練歷史 - 準確率')
        plt.ylabel('準確率')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        
        # 損失圖
        plt.subplot(1, 2, 2)
        plt.plot(history['train_loss'], label='訓練損失')
        plt.plot(history['test_loss'], label='驗證損失')
        plt.title('訓練歷史 - 損失')
        plt.ylabel('損失')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_confidence_distribution(self, df_confidence, epoch, save_path=None):
        """繪製置信度分布"""
        bins = np.arange(0, 1.1, 0.1)
        confidence_counts, _ = np.histogram(df_confidence['confidence'], bins=bins, density=False)
        
        print("每個區間的數量:", confidence_counts)
        
        # 計算每個區間的中點
        bin_mids = (bins[:-1] + bins[1:]) / 2

        # 計算正態分布曲線
        mean = df_confidence['confidence'].mean()
        std = df_confidence['confidence'].std()
        x = np.linspace(0, 1, 100)
        y = norm.pdf(x, mean, std) * len(df_confidence) * np.diff(bins).mean()

        # 繪製分布圖
        plt.figure(figsize=(10, 6))
        plt.bar(bin_mids, confidence_counts, width=0.1, color='b', alpha=0.5, label='置信度分布')
        plt.plot(x, y, color='r', label='正態分布')
        plt.title(f'置信度分布 vs 正態分布 - Epoch {epoch}')
        plt.xlabel('置信度分數')
        plt.ylabel('樣本數量')
        plt.xticks(bins)
        plt.grid(True)
        plt.legend()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_pca(self, features, labels, title, epoch, data_dir):
        """繪製PCA結果"""
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(features)

        custom_labels = ['CH', ' GPI', ' MT', ' NES', ' NLS', ' PTS', ' SP', ' TH', ' TM']

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                             c=labels.argmax(axis=1), cmap='viridis', s=5)
        plt.title(title)

        # 添加自定義顏色條
        cbar = plt.colorbar(scatter)
        cbar.set_ticks(range(len(custom_labels)))
        cbar.set_ticklabels(custom_labels)
        cbar.set_label('蛋白質定位類別')

        # 保存圖片
        filename = f"{title}_epoch_{epoch}.png"
        save_path = os.path.join(data_dir, filename)
        plt.savefig(save_path)
        plt.close()

    def plot_umap(self, features, labels, title, epoch, data_dir):
        """繪製UMAP結果"""
        reducer = umap.UMAP()
        umap_result = reducer.fit_transform(features)

        custom_labels = ['CH', ' GPI', ' MT', ' NES', ' NLS', ' PTS', ' SP', ' TH', ' TM']

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1], 
                             c=labels.argmax(axis=1), cmap='viridis', s=5)
        plt.title(title)

        cbar = plt.colorbar(scatter)
        cbar.set_ticks(range(len(custom_labels)))
        cbar.set_ticklabels(custom_labels)
        cbar.set_label('蛋白質定位類別')

        # 保存圖片
        filename = f"{title}_epoch_{epoch}.png"
        save_path = os.path.join(data_dir, filename)
        plt.savefig(save_path)
        plt.close()

    def plot_tsne(self, features, labels, title, epoch, data_dir):
        """繪製t-SNE結果"""
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
        tsne_result = tsne.fit_transform(features)

        custom_labels = ['CH', ' GPI', ' MT', ' NES', ' NLS', ' PTS', ' SP', ' TH', ' TM']

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                             c=labels.argmax(axis=1), cmap='viridis', s=5)
        plt.title(title)

        cbar = plt.colorbar(scatter)
        cbar.set_ticks(range(len(custom_labels)))
        cbar.set_ticklabels(custom_labels)
        cbar.set_label('蛋白質定位類別')

        # 保存圖片
        filename = f"{title}_epoch_{epoch}.png"
        save_path = os.path.join(data_dir, filename)
        plt.savefig(save_path)
        plt.close()

    def plot_roc_curves(self, y_true, y_pred_proba, target_list, save_path=None):
        """繪製ROC曲線"""
        from sklearn.metrics import roc_curve, auc
        
        n_classes = y_true.shape[1]
        
        # 計算每個類別的ROC曲線和AUC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # 計算微平均ROC曲線和AUC
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # 繪製所有ROC曲線
        plt.figure(figsize=(12, 8))
        plt.plot(fpr["micro"], tpr["micro"], 
                label=f'微平均 ROC 曲線 (面積 = {roc_auc["micro"]:0.4f})')
        
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], 
                    label=f'ROC 曲線 {target_list[i]} (面積 = {roc_auc[i]:0.4f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('假陽性率')
        plt.ylabel('真陽性率')
        plt.title('接受者操作特徵 (ROC) 曲線')
        plt.legend(loc="lower right", prop={'size': 7})
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_confusion_matrices(self, y_true, y_pred, target_list, save_path=None):
        """繪製混淆矩陣"""
        from sklearn.metrics import multilabel_confusion_matrix
        import seaborn as sns
        
        cm = multilabel_confusion_matrix(y_true, y_pred)
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.ravel()
        
        for i, (matrix, label) in enumerate(zip(cm, target_list)):
            sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=axes[i])
            axes[i].set_title(f'混淆矩陣 - {label}')
            axes[i].set_xlabel('預測標籤')
            axes[i].set_ylabel('真實標籤')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()