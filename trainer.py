#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os

class Trainer:
    """訓練管理類別"""
    
    def __init__(self, config, model_manager):
        self.config = config
        self.model_manager = model_manager
        self.history = defaultdict(list)
        self.best_accuracy = 0
        
    def train_epoch(self, model, training_loader, optimizer):
        """訓練一個epoch"""
        losses = []
        correct_predictions = 0
        num_samples = 0
        
        model.train()
        loop = tqdm(enumerate(training_loader), total=len(training_loader), 
                   leave=True, colour='steelblue')
        
        for batch_idx, data in loop:
            ids = data['input_ids'].to(self.config.device, dtype=torch.long)
            mask = data['attention_mask'].to(self.config.device, dtype=torch.long)
            targets = data['targets'].to(self.config.device, dtype=torch.float)

            # 前向傳播
            outputs = model(ids, mask)
            loss = self.model_manager.loss_fn(outputs, targets)
            losses.append(loss.item())
            
            # 計算訓練準確率
            outputs_prob = torch.sigmoid(outputs).cpu().detach().numpy().round()
            targets_cpu = targets.cpu().detach().numpy()
            correct_predictions += np.sum(outputs_prob == targets_cpu)
            num_samples += targets_cpu.size

            # 反向傳播
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        return model, float(correct_predictions) / num_samples, np.mean(losses)

    def eval_model(self, model, validation_loader):
        """評估模型"""
        losses = []
        correct_predictions = 0
        num_samples = 0
        
        model.eval()
        
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(validation_loader, desc="Evaluating"), 0):
                ids = data['input_ids'].to(self.config.device, dtype=torch.long)
                mask = data['attention_mask'].to(self.config.device, dtype=torch.long)
                targets = data['targets'].to(self.config.device, dtype=torch.float)
                
                outputs = model(ids, mask)
                loss = self.model_manager.loss_fn(outputs, targets)
                losses.append(loss.item())

                # 驗證準確率
                outputs_prob = torch.sigmoid(outputs).cpu().detach().numpy().round()
                targets_cpu = targets.cpu().detach().numpy()
                correct_predictions += np.sum(outputs_prob == targets_cpu)
                num_samples += targets_cpu.size

        return float(correct_predictions) / num_samples, np.mean(losses)

    def train(self, model, optimizer, train_loader, test_loader, target_list):
        """完整訓練流程"""
        start_epoch = 1
        
        # 檢查是否有檢查點
        if os.path.exists(self.config.checkpoint_path):
            start_epoch, self.best_accuracy, self.history = \
                self.model_manager.load_checkpoint(model, optimizer, self.config.checkpoint_path)
            start_epoch += 1
            print(f"檢查點已載入，從第 {start_epoch} epoch 開始訓練")

        for epoch in range(start_epoch, self.config.EPOCHS + 1):
            print(f'Epoch {epoch}/{self.config.EPOCHS}')
            torch.cuda.empty_cache()
            
            # 訓練模型
            model, train_acc, train_loss = self.train_epoch(model, train_loader, optimizer)
            
            # 驗證模型
            test_acc, test_loss = self.eval_model(model, test_loader)
            
            print(f'train_loss={train_loss:.4f}, test_loss={test_loss:.4f} '
                  f'train_acc={train_acc:.4f}, test_acc={test_acc:.4f}')
            
            # 記錄歷史
            self.history['train_acc'].append(train_acc)
            self.history['train_loss'].append(train_loss)
            self.history['test_acc'].append(test_acc)
            self.history['test_loss'].append(test_loss)
            
            # 保存最佳模型
            if test_acc > self.best_accuracy:
                self.model_manager.save_model(model, self.config.model_path)
                self.best_accuracy = test_acc
                print(f"最佳模型已保存，test_acc={test_acc:.4f}")
            
            # 保存檢查點
            self.model_manager.save_checkpoint(
                model, optimizer, epoch, self.best_accuracy, 
                self.history, self.config.checkpoint_path
            )
            
            # 記錄訓練日誌
            self._log_epoch_info(epoch, train_loss, test_loss, train_acc, test_acc)
        
        print("訓練完成")
        return model

    def _log_epoch_info(self, epoch, train_loss, test_loss, train_acc, test_acc):
        """記錄epoch信息"""
        with open(self.config.log_path, "a") as log_file:
            log_file.write(f"Epoch {epoch}/{self.config.EPOCHS}\n")
            log_file.write(f'train_loss={train_loss:.4f}, test_loss={test_loss:.4f} '
                          f'train_acc={train_acc:.4f}, test_acc={test_acc:.4f}\n')
            log_file.write(f"Best accuracy: {self.best_accuracy:.4f}\n")
            log_file.write("-" * 50 + "\n")