#!/usr/bin/env python
# coding: utf-8

import os
import torch

class Config:
    """配置管理類別"""
    
    def __init__(self):
        # 模型超參數
        self.MAX_LEN = 512
        self.TRAIN_BATCH_SIZE = 1
        self.VALID_BATCH_SIZE = 1
        self.TEST_BATCH_SIZE = 1
        self.EPOCHS = 10
        self.LEARNING_RATE = 2e-04
        self.THRESHOLD = 0.5
        
        # 設備配置
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        
        # 資料路徑
        self.data_dir = '/home/kchang/T5-2'
        
        # 標籤列表
        self.label_columns = ['CH', ' GPI', ' MT', ' NES', ' NLS', ' PTS', ' SP', ' TH', ' TM']
        
        # 模型配置
        self.model_name = 'Rostlab/prot_t5_xl_uniref50'
        self.dropout_rate = 0.3
        self.hidden_size = 1024
        self.num_labels = 9
        
        # 訓練配置
        self.high_confidence_threshold = 0.9
        self.stability_threshold = 3
        self.num_runs = 5
        
        # 檔案路徑
        self.checkpoint_path = os.path.join(self.data_dir, "checkpoint_base_ProtT5_Dynamic_Label_Predict.pth")
        self.model_path = os.path.join(self.data_dir, "MLTC_model_state_base_ProtT5_Dynamic_Label_Predict.bin")
        self.log_path = os.path.join(self.data_dir, "training_log.txt")
