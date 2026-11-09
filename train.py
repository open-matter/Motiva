import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from model.encoder_roberta import EncoderRoberta
from model.regressor_model import RegressorModel
from tokenizer.PolymerSumTokenization import PolymerSumTokenizer
from transformers import AdamW
from tqdm import tqdm
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# ============ 解析函数 ============
def parse_numerical(numerical_str):
    properties = []
    properties.append(float(numerical_str[:3]) / 100)  # 小数
    properties.append(float(numerical_str[3:7]) / 10)  # 小数
    properties.append(int(numerical_str[7:10]) / 10)
    properties.append(int(numerical_str[10:13]))
    properties.append(int(numerical_str[13:18]))
    properties.append(int(numerical_str[18:21]))
    return properties

# ============ 设备 ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ 数据准备 ============
df = pd.read_csv("data_1968AUG/processed/data_split_AUG.csv")
df["parsed_numerical"] = df["numerical"].apply(parse_numerical)


# 归一化标签
scaler = MinMaxScaler()
scaled_labels = scaler.fit_transform(df["parsed_numerical"].tolist())
df["scaled_numerical"] = scaled_labels.tolist()


# 读取验证集数据
df_valid = pd.read_csv("data_1968AUG/processed/data_valid_split.csv")
df_valid["parsed_numerical"] = df_valid["numerical"].apply(parse_numerical)

scaled_valid_labels = scaler.transform(df_valid["parsed_numerical"].tolist())
df_valid["scaled_numerical"] = scaled_valid_labels.tolist()


# ============ 数据集 ============
class RegDataset(Dataset):
    def __init__(self, tokenizer, df, use_scaled=True):
        self.df = df
        self.tokenizer = tokenizer
        self.use_scaled = use_scaled

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        structure = self.df.iloc[idx]["structure"]
        enc = self.tokenizer(structure, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        labels = self.df.iloc[idx]["scaled_numerical" if self.use_scaled else "parsed_numerical"]
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.float)
        }

# ============ 模型 ============
encoder = EncoderRoberta(pretrained_path="ckpt/finetune_SA_2.pt").to(device)
model = RegressorModel(encoder=encoder, output_dim=6).to(device)

# ============ Tokenizer ============
tokenizer = PolymerSumTokenizer.from_pretrained("model_rl", model_max_len=512)

# ============ 训练准备 ============
train_dataset = RegDataset(tokenizer, df, use_scaled=True)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

valid_dataset = RegDataset(tokenizer, df_valid, use_scaled=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=False)

optimizer = AdamW(model.parameters(), lr=1e-5)  # 学习率调低

def masked_mse_loss(preds, labels):
    mask = labels != 0
    diff = preds - labels
    squared_error = diff ** 2
    masked_squared_error = squared_error * mask
    loss = masked_squared_error.sum() / mask.sum().clamp(min=1)
    return loss

# ============ 训练 ============
# 设置保存路径
save_path = f"data_1968AUG/processed/regressor_29_save_3/train_final.pt"

model.train()
for epoch in range(25):
    epoch_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(train_dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        preds = model(input_ids, attention_mask)

        loss = masked_mse_loss(preds, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()

        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())

    # 汇总所有预测和标签
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # 计算 R² 和 MSE
    r2_scores, mse_scores = [], []
    for i in range(all_labels.shape[1]):
        y_true = all_labels[:, i]
        y_pred = all_preds[:, i]
        mask = y_true != 0
        if mask.sum() > 0:
            r2 = r2_score(y_true[mask], y_pred[mask])
            mse = mean_squared_error(y_true[mask], y_pred[mask])
        else:
            r2, mse = float('nan'), float('nan')
        r2_scores.append(r2)
        mse_scores.append(mse)

    print(f"Epoch {epoch+1} Training Loss: {epoch_loss / len(train_dataloader)}")
    print(f"Training R² per task: {[round(r, 4) for r in r2_scores]}")
    print(f"Training MSE per task: {[round(m, 4) for m in mse_scores]}")
    print(f"  Avg R²: {np.nanmean(r2_scores):.4f}")
    print(f"  Avg MSE: {np.nanmean(mse_scores):.4f}")

    # ==================== 保存训练结果 ====================
    # 将训练结果保存到 CSV 文件
    # 反归一化预测值
    unscaled_preds = scaler.inverse_transform(all_preds)

    # 指定每一列保留的小数位数
    decimal_places = [2, 1, 1, 0, 0, 0]

    # 对每个预测向量逐个元素进行指定精度的 round
    predictions_decimals = [
        [round(p, decimal_places[i]) for i, p in enumerate(pred)] for pred in unscaled_preds
    ]

    # 构造 DataFrame 并保存
    predictions_df = pd.DataFrame(predictions_decimals, columns=["property1", "property2", "property3", "property4", "property5", "property6"])

    df_train = pd.concat([df, predictions_df], axis=1)
    # 保存到 CSV 文件
    df_train.to_csv(f"data_1968AUG/processed/regressor_29_save_3/train_predictions_save_epoch_{epoch+1}.csv", index=False)
    print(f"✅ 训练结果已保存到 data_1968AUG/processed/regressor_29_save_3/train_predictions_save_epoch_{epoch+1}.csv")


    # 在每个 epoch 后进行验证
    model.eval()  # 切换为评估模式
    valid_loss = 0.0
    valid_preds = []
    valid_labels = []

    with torch.no_grad():  # 在验证集时不计算梯度
        for batch in tqdm(valid_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            preds = model(input_ids, attention_mask)

            loss = masked_mse_loss(preds, labels)
            valid_loss += loss.item()

            valid_preds.append(preds.detach().cpu())
            valid_labels.append(labels.detach().cpu())

    # 汇总验证集的预测和标签
    valid_preds = torch.cat(valid_preds).numpy()
    valid_labels = torch.cat(valid_labels).numpy()

    # 计算 R² 和 MSE (验证集)
    valid_r2_scores, valid_mse_scores = [], []
    for i in range(valid_labels.shape[1]):
        y_true = valid_labels[:, i]
        y_pred = valid_preds[:, i]
        mask = y_true != 0
        if mask.sum() > 0:
            r2 = r2_score(y_true[mask], y_pred[mask])
            mse = mean_squared_error(y_true[mask], y_pred[mask])
        else:
            r2, mse = float('nan'), float('nan')
        valid_r2_scores.append(r2)
        valid_mse_scores.append(mse)

    print(f"Epoch {epoch+1} Validation Loss: {valid_loss / len(valid_dataloader)}")
    print(f"Validation R² per task: {[round(r, 4) for r in valid_r2_scores]}")
    print(f"Validation MSE per task: {[round(m, 4) for m in valid_mse_scores]}")
    print(f"  Avg R²: {np.nanmean(valid_r2_scores):.4f}")
    print(f"  Avg MSE: {np.nanmean(valid_mse_scores):.4f}")
    
    model.train()  # 恢复为训练模式

    # 在 epoch 循环的最后一轮保存
    if epoch == 24:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),  # 可选
            'epoch': epoch + 1,
        }, save_path)
        print(f"Model saved to {save_path}")

# ============ 推理 ============ 
model.eval()
df_test = pd.read_csv("data_1968AUG/processed/data_test.csv")
df_test["parsed_numerical"] = df_test["numerical"].apply(parse_numerical)


# ✅ 使用训练时的归一化方法
scaled_test_labels = scaler.transform(df_test["parsed_numerical"].tolist())
df_test["scaled_numerical"] = scaled_test_labels.tolist()


test_dataset = RegDataset(tokenizer, df_test, use_scaled=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

predictions = []
true_labels = []
test_loss = 0.0

with torch.no_grad():
    for batch in tqdm(test_dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        preds = model(input_ids, attention_mask)

        # 计算 masked loss
        mask = labels != 0
        diff = preds - labels
        mse = (diff ** 2 * mask).sum() / mask.sum().clamp(min=1)
        test_loss += mse.item()

        predictions.append(preds.cpu()) # 预测值是归一化的
        true_labels.append(labels.cpu())

# 拼接
predictions = torch.cat(predictions).numpy()
true_labels = torch.cat(true_labels).numpy()

# ============ 指标计算 ============ 
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

r2_scores, mse_scores = [], []
for i in range(true_labels.shape[1]):
    y_true = true_labels[:, i]
    y_pred = predictions[:, i]
    mask = y_true != 0
    if mask.sum() > 0:
        r2 = r2_score(y_true[mask], y_pred[mask])
        mse = mean_squared_error(y_true[mask], y_pred[mask])
    else:
        r2, mse = float('nan'), float('nan')
    r2_scores.append(r2)
    mse_scores.append(mse)

print(f"\nTest Loss (masked MSE): {test_loss / len(test_dataloader):.4f}")
print(f"R² per task: {[round(r, 4) for r in r2_scores]}")
print(f"MSE per task: {[round(m, 4) for m in mse_scores]}")
print(f"Avg R²: {np.nanmean(r2_scores):.4f}")
print(f"Avg MSE: {np.nanmean(mse_scores):.4f}")

# ============ 反归一化预测并保存 ============ 

# 反归一化预测值
unscaled_preds = scaler.inverse_transform(predictions)


# 指定每一列保留的小数位数
decimal_places = [2, 1, 1, 0, 0, 0]

# 对每个预测向量逐个元素进行指定精度的 round
predictions_decimals = [
    [round(p, decimal_places[i]) for i, p in enumerate(pred)] for pred in unscaled_preds
]

# 构造 DataFrame 并保存
predictions_df = pd.DataFrame(predictions_decimals, columns=[
    "property1", "property2", "property3", "property4", "property5", "property6"
])

df_test = pd.concat([df_test, predictions_df], axis=1)
df_test.to_csv("data_1968AUG/processed/regressor_29_save_3/test_predictions_29_SUP_save.csv", index=False)

print("✅ 预测结果已保存到 data_1968AUG/processed/regressor_29_save_3/test_predictions_29_SUP_save.csv")
