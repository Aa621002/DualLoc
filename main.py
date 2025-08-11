#!/usr/bin/env python
# coding: utf-8

"""
多標籤文本分類主程式
使用T5模型進行蛋白質定位預測
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

# 導入自定義模組
from config import Config
from dataset import DataManager
from model import ModelManager
from trainer import Trainer
from evaluator import Evaluator
from visualizer import Visualizer
from utils import Utils, FileManager

class MultiLabelClassifier:
    """多標籤分類器主類別"""
    
    def __init__(self):
        self.config = Config()
        self.data_manager = DataManager(self.config)
        self.model_manager = ModelManager(self.config)
        self.evaluator = Evaluator(self.config)
        self.visualizer = Visualizer(self.config)
        
        # 確保輸出目錄存在
        FileManager.ensure_directory_exists(self.config.data_dir)
        
    def load_and_prepare_data(self):
        """載入和準備資料"""
        print("載入資料中...")
        
        # 載入資料
        csv_path = os.path.join(self.config.data_dir, "signal_sorting.csv")
        df_data = self.data_manager.load_data(csv_path, sample_size=1868)
        
        # 分割資料
        df_train, df_test = self.data_manager.split_data(df_data)
        
        # 獲取目標標籤
        target_list = self.data_manager.get_target_list(df_data)
        
        print(f"訓練集: {df_train.shape}, 測試集: {df_test.shape}")
        
        # 顯示標籤統計
        train_counts = self.data_manager.get_label_counts(df_train, self.config.label_columns)
        test_counts = self.data_manager.get_label_counts(df_test, self.config.label_columns)
        
        print("訓練集標籤統計:")
        print(train_counts)
        print("\n測試集標籤統計:")
        print(test_counts)
        
        return df_train, df_test, target_list
    
    def create_data_loaders(self, df_train, df_test, target_list):
        """創建資料載入器"""
        print("創建資料載入器...")
        
        # 創建資料集
        train_dataset = self.data_manager.create_dataset(df_train, target_list)
        test_dataset = self.data_manager.create_dataset(df_test, target_list)
        
        # 創建資料載入器
        train_loader = self.data_manager.create_dataloader(
            train_dataset, self.config.TRAIN_BATCH_SIZE, shuffle=True
        )
        test_loader = self.data_manager.create_dataloader(
            test_dataset, self.config.TEST_BATCH_SIZE, shuffle=False
        )
        
        return train_loader, test_loader
    
    def setup_model_and_optimizer(self):
        """設置模型和優化器"""
        print("設置模型和優化器...")
        
        # 創建模型
        model = self.model_manager.create_model()
        
        # 創建優化器
        optimizer = self.model_manager.create_optimizer(model)
        
        return model, optimizer
    
    def train_model(self, model, optimizer, train_loader, test_loader, target_list):
        """訓練模型"""
        print("開始訓練...")
        
        trainer = Trainer(self.config, self.model_manager)
        trained_model = trainer.train(model, optimizer, train_loader, test_loader, target_list)
        
        return trained_model, trainer.history
    
    def evaluate_model(self, model, test_loader, df_test, target_list):
        """評估模型"""
        print("評估模型...")
        
        # 獲取預測結果
        titles, predictions, prediction_probs, target_values, df_confidence, incorrect_predictions_df = \
            self.evaluator.get_predictions(model, df_test, test_loader, "final", self.config.data_dir)
        
        # 計算指標
        metrics = self.evaluator.calculate_metrics(target_values, predictions, prediction_probs, target_list)
        
        # 生成分類報告
        report = self.evaluator.generate_classification_report(target_values, predictions, target_list)
        
        print("評估結果:")
        print(f"F1 Micro: {metrics['f1_micro']:.4f}")
        print(f"F1 Macro: {metrics['f1_macro']:.4f}")
        print(f"Jaccard Micro: {metrics['jaccard_micro']:.4f}")
        print(f"Jaccard Macro: {metrics['jaccard_macro']:.4f}")
        print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
        print(f"Subset Accuracy: {metrics['subset_accuracy']:.4f}")
        
        print("\n分類報告:")
        print(report)
        
        return metrics, predictions, prediction_probs, target_values
    
    def visualize_results(self, model, train_loader, test_loader, history, epoch=1):
        """視覺化結果"""
        print("生成視覺化圖表...")
        
        # 繪製訓練歷史
        history_path = os.path.join(self.config.data_dir, "training_history.png")
        self.visualizer.plot_training_history(history, history_path)
        
        # 提取特徵
        print("提取訓練集特徵...")
        train_features, train_labels = self.visualizer.extract_features(model, train_loader)
        
        print("提取測試集特徵...")
        test_features, test_labels = self.visualizer.extract_features(model, test_loader)
        
        # 繪製降維圖
        self.visualizer.plot_pca(train_features, train_labels, 
                                f"PCA_Training_Set_Features_Epoch_{epoch}", epoch, self.config.data_dir)
        self.visualizer.plot_pca(test_features, test_labels, 
                                f"PCA_Test_Set_Features_Epoch_{epoch}", epoch, self.config.data_dir)
        
        self.visualizer.plot_umap(train_features, train_labels, 
                                 f"UMAP_Training_Set_Features_Epoch_{epoch}", epoch, self.config.data_dir)
        self.visualizer.plot_umap(test_features, test_labels, 
                                 f"UMAP_Test_Set_Features_Epoch_{epoch}", epoch, self.config.data_dir)
        
        self.visualizer.plot_tsne(train_features, train_labels, 
                                 f"tSNE_Training_Set_Features_Epoch_{epoch}", epoch, self.config.data_dir)
        self.visualizer.plot_tsne(test_features, test_labels, 
                                 f"tSNE_Test_Set_Features_Epoch_{epoch}", epoch, self.config.data_dir)
    
    def run_multiple_evaluations(self, model, test_loader, df_test, target_list, num_runs=5):
        """執行多次評估"""
        print(f"執行 {num_runs} 次評估...")
        
        all_results, all_predictions, all_targets = self.evaluator.evaluate_multiple_runs(
            model, test_loader, df_test, target_list, num_runs
        )
        
        # 計算平均結果
        avg_metrics = self._calculate_average_metrics(all_results)
        
        # 集成學習預測
        ensemble_metrics = self._ensemble_predictions(all_predictions, all_targets, target_list)
        
        print("平均評估結果:")
        print(f"平均 F1 Micro: {avg_metrics['f1_micro']:.4f} ± {avg_metrics['f1_micro_std']:.4f}")
        print(f"平均 F1 Macro: {avg_metrics['f1_macro']:.4f} ± {avg_metrics['f1_macro_std']:.4f}")
        
        print("\n集成學習結果:")
        print(f"集成 F1 Micro: {ensemble_metrics['f1_micro']:.4f}")
        print(f"集成 F1 Macro: {ensemble_metrics['f1_macro']:.4f}")
        
        # 保存結果
        self._save_evaluation_results(all_results, avg_metrics, ensemble_metrics)
        
        return all_results, avg_metrics, ensemble_metrics
    
    def _calculate_average_metrics(self, all_results):
        """計算平均指標"""
        metrics_list = [result['metrics'] for result in all_results]
        
        avg_metrics = {}
        for key in ['f1_micro', 'f1_macro', 'jaccard_micro', 'jaccard_macro', 'hamming_loss', 'subset_accuracy']:
            values = [metrics[key] for metrics in metrics_list]
            avg_metrics[key] = np.mean(values)
            avg_metrics[f'{key}_std'] = np.std(values)
        
        return avg_metrics
    
    def _ensemble_predictions(self, all_predictions, all_targets, target_list):
        """集成學習預測"""
        # 轉換為numpy數組
        all_predictions = np.array([pred.cpu().numpy() if torch.is_tensor(pred) else pred 
                                   for pred in all_predictions])
        all_targets = np.array([target.cpu().numpy() if torch.is_tensor(target) else target 
                               for target in all_targets])
        
        # 對所有模型的預測結果進行平均
        ensemble_predictions = np.mean(all_predictions, axis=0)
        ensemble_predictions = (ensemble_predictions > self.config.THRESHOLD).astype(int)
        
        # 計算集成模型的性能
        ensemble_metrics = self.evaluator.calculate_metrics(
            all_targets[0], ensemble_predictions, ensemble_predictions, target_list
        )
        
        return ensemble_metrics
    
    def _save_evaluation_results(self, all_results, avg_metrics, ensemble_metrics):
        """保存評估結果"""
        # 保存詳細結果
        results_data = []
        for i, result in enumerate(all_results):
            results_data.append({
                'run': i + 1,
                **result['metrics']
            })
        
        results_df = pd.DataFrame(results_data)
        results_path = os.path.join(self.config.data_dir, "evaluation_results.csv")
        results_df.to_csv(results_path, index=False)
        
        # 保存平均結果
        avg_results_df = pd.DataFrame([avg_metrics])
        avg_path = os.path.join(self.config.data_dir, "average_evaluation_results.csv")
        avg_results_df.to_csv(avg_path, index=False)
        
        # 保存集成結果
        ensemble_results_df = pd.DataFrame([ensemble_metrics])
        ensemble_path = os.path.join(self.config.data_dir, "ensemble_evaluation_results.csv")
        ensemble_results_df.to_csv(ensemble_path, index=False)
        
        print(f"評估結果已保存到: {results_path}")
        print(f"平均結果已保存到: {avg_path}")
        print(f"集成結果已保存到: {ensemble_path}")
    
    def predict_raw_text(self, model, raw_text, target_list):
        """對原始文本進行預測"""
        print("對原始文本進行預測...")
        
        # 預處理文本
        raw_text = " ".join(raw_text)
        
        # 編碼
        encoded_text = self.data_manager.tokenizer.encode_plus(
            raw_text,
            max_length=self.config.MAX_LEN,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        # 預測
        model.eval()
        with torch.no_grad():
            input_ids = encoded_text['input_ids'].to(self.config.device)
            attention_mask = encoded_text['attention_mask'].to(self.config.device)
            
            output = model(input_ids, attention_mask)
            output = torch.sigmoid(output).detach().cpu()
            output = output.flatten().round().numpy()
        
        # 顯示結果
        print(f"文本: {raw_text[:100]}...")
        print("預測標籤:")
        for idx, p in enumerate(output):
            if p == 1:
                print(f"  - {target_list[idx]}")
        
        return output
    
    def run_complete_pipeline(self):
        """執行完整的流程"""
        print("=" * 50)
        print("開始執行多標籤文本分類流程")
        print("=" * 50)
        
        try:
            # 1. 載入和準備資料
            df_train, df_test, target_list = self.load_and_prepare_data()
            
            # 2. 創建資料載入器
            train_loader, test_loader = self.create_data_loaders(df_train, df_test, target_list)
            
            # 3. 設置模型和優化器
            model, optimizer = self.setup_model_and_optimizer()
            
            # 4. 訓練模型
            trained_model, history = self.train_model(model, optimizer, train_loader, test_loader, target_list)
            
            # 5. 評估模型
            metrics, predictions, prediction_probs, target_values = self.evaluate_model(
                trained_model, test_loader, df_test, target_list
            )
            
            # 6. 視覺化結果
            self.visualize_results(trained_model, train_loader, test_loader, history)
            
            # 7. 繪製ROC曲線和混淆矩陣
            roc_path = os.path.join(self.config.data_dir, "roc_curves.png")
            self.visualizer.plot_roc_curves(target_values, prediction_probs, target_list, roc_path)
            
            cm_path = os.path.join(self.config.data_dir, "confusion_matrices.png")
            self.visualizer.plot_confusion_matrices(target_values, predictions, target_list, cm_path)
            
            # 8. 多次評估
            all_results, avg_metrics, ensemble_metrics = self.run_multiple_evaluations(
                trained_model, test_loader, df_test, target_list, self.config.num_runs
            )
            
            # 9. 測試原始文本預測
            sample_text = "MYWSNQITRRLGERVQGFMSGISPQQMGEPEGSWSGKNPGTMGASRLYTLVLVLQPQRVLLGMKKRGFGAGRWNGFGGKVQEGETIEDGARRELQEESGLTVDALHKVGQIVFEFVGEPELMDVHVFCTDSIQGTPVESDEMRPCWFQLDQIPFKDMWPDDSYWFPLLLQKKKFHGYFKFQGQDTILDYTLREVDTV"
            self.predict_raw_text(trained_model, sample_text, target_list)
            
            print("=" * 50)
            print("流程執行完成！")
            print("=" * 50)
            
            return {
                'model': trained_model,
                'metrics': metrics,
                'avg_metrics': avg_metrics,
                'ensemble_metrics': ensemble_metrics,
                'target_list': target_list
            }
            
        except Exception as e:
            print(f"執行過程中發生錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """主函數"""
    # 創建分類器實例
    classifier = MultiLabelClassifier()
    
    # 執行完整流程
    results = classifier.run_complete_pipeline()
    
    if results:
        print("所有結果已保存到:", classifier.config.data_dir)
    else:
        print("執行失败，請檢查錯誤信息")

if __name__ == "__main__":
    main()