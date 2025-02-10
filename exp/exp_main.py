from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Corrformer, SIGMAformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, simple_metric
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.device = torch.device(f"cuda:{args.gpu}" if args.use_gpu else "cpu")
        self.model = self._build_model().to(self.device)
        
    def _build_model(self):
        model_dict = {
            'Corrformer': Corrformer,
            'SIGMAformer' : SIGMAformer
        }
        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder-decoder forward
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss = []
            time_now = time.time()

            for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # forward
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                # backward
                model_optim.zero_grad()
                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {vali_loss:.4f}, Test Loss: {test_loss:.4f}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # Save the best model
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        self.model.eval()
        preds, trues = [], []

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # forward
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                preds.append(outputs.cpu().numpy())
                trues.append(batch_y.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        # Save predictions and ground truth
        #folder_path = os.path.join('./results', setting)
        folder_path = './results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)
        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'true.npy'), trues)
        
        mse, mae = metric(preds, trues)

        print(f"Test Results - MSE: {mse:.4f}, MAE: {mae:.4f}")

        with open("result.txt", "a") as f:
            f.write(f"{setting}\n")
            f.write(f"MSE: {mse:.4f}, MAE: {mae:.4f}\n\n")

        print(f"Predictions and ground truth saved at {folder_path}")
        return preds, trues

    def _generate_visualizations(self, preds, trues, save_path):
        num_samples = preds.shape[0]
        for i in range(min(10, num_samples)):  
            gt = np.concatenate((trues[i, :, -1], preds[i, :, -1]), axis=0)
            pd = np.concatenate((trues[i, :, -1], preds[i, :, -1]), axis=0)
            visual(gt, pd, os.path.join(save_path, f"{i}.pdf"))
        print(f"Visualizations saved to {save_path}")
