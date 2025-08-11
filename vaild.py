#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    jaccard_score, f1_score, matthews_corrcoef, 
    classification_report, accuracy_score,
    hamming_loss, label_ranking_loss, coverage_error,
    label_ranking_average_precision_score
)
import os

class Evaluator:
    """評估和指標計算類別"""
    
    def __init__(self, config):
        self.config = config
        
    def get_predictions(self, model, df, data_loader, epoch, data_dir):
        """獲取模型預測結果"""
        model.eval()
        
        titles = []
        predictions = []
        prediction_probs = []
        target_values = []
        confidence_scores = []
        indices = []
        incorrect_predictions = []

        with torch.no_grad():
            for data in tqdm(data_loader, desc="Predicting"):
                title = data["title"]
                ids = data["input_ids"].to(self.config.device, dtype=torch.long)
                mask = data["attention_mask"].to(self.config.device, dtype=torch.long)
                targets = data["targets"].to(self.config.device, dtype=torch.float)
                index = data["index"].to(self.config.device, dtype=torch.long)

                outputs = model(ids, mask)
                logits = outputs

                outputs_prob = torch.sigmoid(outputs).detach().cpu()
                preds = (outputs_prob >= self.config.THRESHOLD).float()
                targets_cpu = targets.detach().cpu()

                titles.extend(title)
                predictions.extend(preds)
                prediction_probs.extend(outputs_prob)
                target_values.extend(targets_cpu)
                indices.extend(index.cpu().numpy())

                # 計算置信度
                probs = torch.softmax(logits, dim=1)
                confidence = probs.max(dim=1).values.cpu().numpy()
                confidence_scores.extend(confidence)

                # 追蹤錯誤預測
                first_label_index = 0
                incorrect_indices = (preds[:, first_label_index] != targets_cpu[:, first_label_index]).nonzero(as_tuple=True)[0]
                for idx in incorrect_indices:
                    incorrect_predictions.append({
                        'title': title[idx],
                        'true_label': targets_cpu[idx, first_label_index].item(),
                        'predicted_label': preds[idx, first_label_index].item(),
                        'confidence': confidence[idx]
                    })

        # 根據索引恢復原始順序
        indices = np.array(indices)
        sorted_indices = np.argsort(indices)

        titles = np.array(titles)[sorted_indices]
        predictions = torch.stack(predictions)[sorted_indices]
        prediction_probs = torch.stack(prediction_probs)[sorted_indices]
        target_values = torch.stack(target_values)[sorted_indices]
        confidence_scores = np.array(confidence_scores)[sorted_indices]

        # 保存置信度數據
        df_confidence = df.copy()
        df_confidence['confidence'] = confidence_scores

        # 保存錯誤預測
        incorrect_predictions_df = pd.DataFrame(incorrect_predictions)
        
        # 保存到檔案
        incorrect_predictions_df.to_csv(
            os.path.join(data_dir, f"incorrect_predictions_epoch_{epoch}.csv"), 
            index=False
        )
        df_confidence.to_csv(
            os.path.join(data_dir, f"df_train_adjusted_confidence_epoch_{epoch}.csv"), 
            index=False
        )

        return titles, predictions, prediction_probs, target_values, df_confidence, incorrect_predictions_df

    def calculate_metrics(self, y_true, y_pred, y_prob, target_list):
        """計算各種評估指標"""
        metrics = {}
        
        # 轉換為numpy數組
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        if torch.is_tensor(y_prob):
            y_prob = y_prob.cpu().numpy()
        
        # Jaccard Score (IoU)
        jaccard_micro = jaccard_score(y_true, y_pred, average='micro')
        jaccard_macro = jaccard_score(y_true, y_pred, average='macro')
        
        # F1 Scores
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        # Hamming Loss
        hamming = hamming_loss(y_true, y_pred)
        
        # Subset Accuracy
        subset_accuracy = accuracy_score(y_true, y_pred)
        
        # 每個標籤的MCC和F1
        label_mccs = []
        label_f1s = []
        label_stats = []
        
        for i, label_name in enumerate(target_list):
            # MCC for each label
            try:
                mcc = matthews_corrcoef(y_true[:, i], y_pred[:, i])
                label_mccs.append(mcc)
            except:
                label_mccs.append(0.0)
            
            # F1 for each label
            f1_label = f1_score(y_true[:, i], y_pred[:, i], average='binary')
            label_f1s.append(f1_label)
            
            # 統計每個標籤的正樣本數量
            true_positives = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
            predicted_positives = np.sum(y_pred[:, i] == 1)
            actual_positives = np.sum(y_true[:, i] == 1)
            
            label_stats.append({
                'label': label_name,
                'true_positives': int(true_positives),
                'predicted_positives': int(predicted_positives),
                'actual_positives': int(actual_positives),
                'mcc': mcc if 'mcc' in locals() else 0.0,
                'f1': f1_label
            })
        
        # 計算多少蛋白質被預測為有正標籤
        proteins_with_positive_pred = np.sum(np.any(y_pred == 1, axis=1))
        total_proteins = y_pred.shape[0]
        
        metrics.update({
            'jaccard_micro': jaccard_micro,
            'jaccard_macro': jaccard_macro,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'hamming_loss': hamming,
            'subset_accuracy': subset_accuracy,
            'label_mccs': label_mccs,
            'label_f1s': label_f1s,
            'label_stats': label_stats,
            'proteins_with_positive_pred': int(proteins_with_positive_pred),
            'total_proteins': int(total_proteins),
            'positive_protein_ratio': proteins_with_positive_pred / total_proteins
        })
        
        return metrics

    def generate_classification_report(self, y_true, y_pred, target_list):
        """生成分類報告"""
        return classification_report(y_true, y_pred, target_names=target_list, digits=4)

    def evaluate_multiple_runs(self, model, test_loader, df_test, target_list, num_runs=5):
        """多次運行評估"""
        all_results = []
        all_predictions = []
        all_targets = []
        
        for run in range(num_runs):
            print(f"運行 {run + 1}/{num_runs}")
            
            # 獲取預測結果
            titles, predictions, prediction_probs, target_values, df_confidence, incorrect_predictions_df = \
                self.get_predictions(model, df_test, test_loader, f"final_prediction_{run + 1}", self.config.data_dir)
            
            # 存儲預測結果
            all_predictions.append(predictions)
            all_targets.append(target_values)
            
            # 計算指標
            metrics = self.calculate_metrics(target_values, predictions, prediction_probs, target_list)
            
            # 生成分類報告
            report = self.generate_classification_report(target_values, predictions, target_list)
            
            all_results.append({
                'run': run + 1,
                'metrics': metrics,
                'report': report
            })
            
            print(f"運行 {run + 1} - F1 Micro: {metrics['f1_micro']:.4f}")
            print(f"運行 {run + 1} - F1 Macro: {metrics['f1_macro']:.4f}")
        
        return all_results, all_predictions, all_targets