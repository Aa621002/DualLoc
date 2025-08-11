#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import torch
import os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score

class Utils:
    """工具函數類別"""
    
    @staticmethod
    def filter_high_confidence_labels_with_stability_check(df_train_label, df_confidence, 
                                                          predictions_for_label, label, epoch, data_dir, 
                                                          high_confidence_threshold=0.9, stability_threshold=3):
        """過濾高置信度標籤並進行穩定性檢查"""
        high_confidence_samples = []
        removed_positive_samples = []
        removed_negative_samples = []
        
        # 確保 predictions_for_label 中的所有元素都是 NumPy 陣列
        predictions_for_label = [pred.numpy() if isinstance(pred, torch.Tensor) else pred 
                               for pred in predictions_for_label]
        
        # 將 predictions_for_label 轉換為 NumPy 陣列
        prediction_probs = np.array(predictions_for_label)
        
        # 檢查 prediction_probs 的形狀
        print(f"prediction_probs shape before reshape: {prediction_probs.shape}")
        
        # 如果 prediction_probs 是一維數組，將其轉換為二維數組
        if len(prediction_probs.shape) == 1:
            prediction_probs = prediction_probs.reshape(-1, 1)
        
        print(f"prediction_probs shape after reshape: {prediction_probs.shape}")
        
        # 找出高置信度的樣本
        for i in range(len(prediction_probs)):
            high_confidence_indices = (prediction_probs[i] >= high_confidence_threshold).nonzero()[0]
            if len(high_confidence_indices) > 0:
                if i < len(df_train_label):
                    # 檢查該樣本的置信度是否穩定
                    if Utils.is_confidence_stable(df_confidence.iloc[i], epoch, data_dir, stability_threshold):
                        high_confidence_samples.append(df_train_label.iloc[i])
                    else:
                        # 只保留被刪掉的正樣本和負樣本
                        if df_train_label.iloc[i][label] == 1:
                            removed_positive_samples.append(df_train_label.iloc[i])
                        else:
                            removed_negative_samples.append(df_train_label.iloc[i])
        
        # 將高置信度的樣本保存為一個 DataFrame
        high_confidence_df = pd.DataFrame(high_confidence_samples)
        removed_positive_df = pd.DataFrame(removed_positive_samples)
        removed_negative_df = pd.DataFrame(removed_negative_samples)
        
        return high_confidence_df, removed_positive_df, removed_negative_df

    @staticmethod
    def is_confidence_stable(confidence_sample, epoch, data_dir, stability_threshold):
        """檢查置信度是否穩定"""
        confidence_changes = 0
        
        # 檢查當前 epoch 往前 3 次的置信度
        for e in range(max(1, epoch - 3), epoch):
            confidence_file_path = os.path.join(data_dir, f"df_train_adjusted_confidence_epoch_{e}.csv")
            if os.path.exists(confidence_file_path):
                df_confidence = pd.read_csv(confidence_file_path)
                # 找到該樣本的置信度
                sample_confidence = df_confidence.loc[df_confidence.index == confidence_sample.name, 'confidence'].values
                if len(sample_confidence) > 0:
                    current_confidence = confidence_sample['confidence']
                    if abs(current_confidence - sample_confidence[0]) > 0.1:
                        confidence_changes += 1
        
        return confidence_changes <= stability_threshold

    @staticmethod 
    def calculate_confidence_intervals(prediction_probs, target_list, target_values, epoch, data_dir):
        """計算置信度區間"""
        confidence_ranges = []
        results = []
        
        for i in range(prediction_probs.shape[1]):
            label_probs = prediction_probs[:, i]
            label_targets = target_values[:, i]
            
            # 計算正樣本和負樣本的置信度
            positive_probs = label_probs[label_targets == 1]
            negative_probs = label_probs[label_targets == 0]
            
            # 計算置信度區間
            confidence_range = (label_probs.min(), label_probs.max())
            confidence_ranges.append(confidence_range)
            
            # 計算每個區間的數量
            bins = np.arange(0, 1.1, 0.1)
            
            # 計算正樣本的區間數量
            positive_hist, _ = np.histogram(positive_probs, bins=bins)
            
            # 計算負樣本的區間數量
            negative_hist, _ = np.histogram(negative_probs, bins=bins)
            
            print(f"Label {target_list[i]} confidence range: {confidence_range}, count: {len(label_probs)}")
            print(f"Positive samples histogram for Label {target_list[i]}: {positive_hist}")
            print(f"Negative samples histogram for Label {target_list[i]}: {negative_hist}")
            
            # 將結果存儲到列表中
            results.append({
                'label': target_list[i],
                'confidence_range_min': confidence_range[0],
                'confidence_range_max': confidence_range[1],
                'positive_hist': positive_hist.tolist(),
                'negative_hist': negative_hist.tolist()
            })
        
        # 將結果轉換為 DataFrame
        results_df = pd.DataFrame(results)
        
        # 保存結果到 CSV 文件
        results_path = os.path.join(data_dir, f"confidence_intervals_epoch_{epoch}.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Confidence intervals saved to {results_path}")
        
        return confidence_ranges

    @staticmethod
    def calculate_f1_per_label(y_true, y_pred):
        """計算每個標籤的F1分數"""
        f1_scores = []
        for i in range(y_true.shape[1]):
            f1 = f1_score(y_true[:, i], y_pred[:, i])
            f1_scores.append(f1)
        return f1_scores

    @staticmethod
    def compare_epoch_predictions(epoch1, epoch2, data_dir):
        """比較不同epoch的預測結果"""
        # 讀取兩個 epoch 的錯誤預測
        epoch1_path = os.path.join(data_dir, f"incorrect_predictions_epoch_{epoch1}.csv")
        epoch2_path = os.path.join(data_dir, f"incorrect_predictions_epoch_{epoch2}.csv")
        
        if not (os.path.exists(epoch1_path) and os.path.exists(epoch2_path)):
            print(f"找不到epoch {epoch1} 或 {epoch2} 的預測檔案")
            return pd.DataFrame(), pd.DataFrame()
        
        df_epoch1 = pd.read_csv(epoch1_path)
        df_epoch2 = pd.read_csv(epoch2_path)
        
        # 比較兩個 epoch 的錯誤預測
        same_predictions = []
        different_predictions = []
        
        for index, row in df_epoch1.iterrows():
            title = row['title']
            true_label = row['true_label']
            predicted_label_epoch1 = row['predicted_label']
            confidence_epoch1 = row['confidence']
            
            # 在第二個 epoch 中查找相同的樣本
            matching_row = df_epoch2[df_epoch2['title'] == title]
            if not matching_row.empty:
                predicted_label_epoch2 = matching_row['predicted_label'].values[0]
                confidence_epoch2 = matching_row['confidence'].values[0]
                
                # 檢查預測值是否相同
                if predicted_label_epoch1 == predicted_label_epoch2:
                    same_predictions.append({
                        'title': title,
                        'true_label': true_label,
                        'predicted_label_epoch1': predicted_label_epoch1,
                        'predicted_label_epoch2': predicted_label_epoch2,
                        'confidence_epoch1': confidence_epoch1,
                        'confidence_epoch2': confidence_epoch2
                    })
                else:
                    different_predictions.append({
                        'title': title,
                        'true_label': true_label,
                        'predicted_label_epoch1': predicted_label_epoch1,
                        'predicted_label_epoch2': predicted_label_epoch2,
                        'confidence_epoch1': confidence_epoch1,
                        'confidence_epoch2': confidence_epoch2
                    })

        # 將結果轉換為 DataFrame
        same_predictions_df = pd.DataFrame(same_predictions)
        different_predictions_df = pd.DataFrame(different_predictions)
        
        # 保存到 CSV 文件
        same_output_path = os.path.join(data_dir, f"same_predictions_epoch_{epoch1}_vs_{epoch2}.csv")
        different_output_path = os.path.join(data_dir, f"different_predictions_epoch_{epoch1}_vs_{epoch2}.csv")
        same_predictions_df.to_csv(same_output_path, index=False)
        different_predictions_df.to_csv(different_output_path, index=False)
        
        print(f"相同預測結果已保存: epoch {epoch1} vs {epoch2}")
        print(f"不同預測結果已保存: epoch {epoch1} vs {epoch2}")
        
        return same_predictions_df, different_predictions_df

    @staticmethod
    def save_results_to_csv(data, filepath, description="結果"):
        """保存結果到CSV"""
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        else:
            pd.DataFrame(data).to_csv(filepath, index=False)
        print(f"{description}已保存到 {filepath}")

    @staticmethod
    def detect_anomalies(features, contamination=0.05, random_state=77):
        """使用Isolation Forest檢測異常樣本"""
        iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
        anomaly_predictions = iso_forest.fit_predict(features)
        return anomaly_predictions

    @staticmethod
    def balance_samples(df, label_col, random_state=77):
        """平衡正負樣本"""
        positive_samples = df[df[label_col] == 1]
        negative_samples = df[df[label_col] == 0]
        
        num_positive = len(positive_samples)
        num_negative = len(negative_samples)
        
        if num_positive <= num_negative:
            negative_samples = negative_samples.sample(n=num_positive, random_state=random_state)
        else:
            positive_samples = positive_samples.sample(n=num_negative, random_state=random_state)
        
        return pd.concat([positive_samples, negative_samples])

class FileManager:
    """檔案管理類別"""
    
    @staticmethod
    def save_adjusted_df_train(df_train, epoch, data_dir):
        """保存調整後的訓練資料"""
        adjusted_df_train_path = os.path.join(data_dir, f"adjusted_df_train_epoch_{epoch}.csv")
        df_train.to_csv(adjusted_df_train_path, index=False)
        print(f"調整後的 df_train 已保存，epoch {epoch}")

    @staticmethod
    def save_removed_samples(removed_positive_samples, removed_negative_samples, epoch, data_dir, prefix=""):
        """保存被移除的樣本"""
        removed_positive_path = os.path.join(data_dir, f"{prefix}removed_positive_samples_epoch_{epoch}.csv")
        removed_negative_path = os.path.join(data_dir, f"{prefix}removed_negative_samples_epoch_{epoch}.csv")
        
        removed_positive_samples.to_csv(removed_positive_path, index=False)
        removed_negative_samples.to_csv(removed_negative_path, index=False)
        
        print(f"被移除的正負樣本已保存，epoch {epoch}")

    @staticmethod
    def ensure_directory_exists(directory):
        """確保目錄存在"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"目錄已創建: {directory}")