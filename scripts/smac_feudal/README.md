# FeUdal + SMAC 訓練

這個訓練器直接使用既有的：

- `module/Algorithm/FeUdal/algorithm.py`
- `module/Algorithm/FeUdal/model/model.py`

環境採用官方 SMAC 地圖。`n_rollout_threads` 大於 1 時，所有演算法都會平行啟動多個 SMAC 環境：

- `haa2c`、`feudal_haa2c` 使用一套共用權重做向量化 rollout。
- `feudal`、`feudal_test`、`qmix` 使用獨立 replica，避免舊版 RNN／replay buffer 的狀態互相污染；各 replica 的模型會各自存到 `checkpoints/replica_<id>_*.pt`。

每回合會記錄：

- episode return
- 每回合步數
- 我方擊殺數（敵方死亡數）
- 我方剩餘單位數
- 單回合勝負與最近 100 回合勝率

輸出位於 `runs/<時間>_<演算法>/`：

- `metrics.csv`
- `training_metrics.png`
- `checkpoints/`
- `tensorboard/`（TensorBoard event logs）

## 1. 安裝 StarCraft II

Windows 必須先透過 Battle.net 安裝 StarCraft II；免費 Starter Edition 即可。預設路徑會自動偵測。若裝在別處，後面的設定命令傳入 `-SC2Path`。

## 2. 建立環境並安裝 SMAC 地圖

在 PowerShell 執行：

```powershell
C:\Users\邱鵬\Desktop\SC2\scripts\smac_feudal
.\setup_smac.ps1
```

自訂遊戲路徑：

```powershell
.\setup_smac.ps1 -SC2Path "D:\StarCraft II"
```

## 3. 設定訓練步數

先用記事本開啟 `training_config.yaml`，修改：

```yaml
map: 8m
total_steps: 1600000
chart_smoothing: 0.99
```

然後直接執行：

```powershell
.\run_train.ps1
```

`8m` 就是 8 Marines 對 8 Marines。也可換成 `3m`、`2s3z` 等其他 SMAC 地圖。
訓練累積達到 1,600,000 環境步後，會在當前回合結束、儲存最終模型，
並自動開啟圖表視窗顯示最新紀錄。

## 4. 離線選擇紀錄並看圖

```powershell
.\view_charts.ps1
```

會開啟離線圖表視窗。表格列出每次執行時間、地圖、回合數及紀錄名稱；
勾選一筆後按「顯示已勾選圖表」，右側會直接呈現五張統計圖。
所有圖表的橫軸都是累積環境步數（Environment steps）。

也可直接指定紀錄：

```powershell
C:\Users\P\Desktop\SC2\.venv-smac\Scripts\python.exe .\plot_run.py --run 8m_validation --show
```

## 5. 使用 TensorBoard 看即時數據

在另一個 PowerShell 視窗、專案根目錄執行：

```powershell
.\.venv-smac\Scripts\tensorboard.exe --logdir .\runs
```

開啟終端顯示的網址（通常是 `http://localhost:6006`）。Scalars 的前四個項目依序為
Episode return、最近 100 局勝率、擊殺敵人數、我軍剩餘單位數；其後還有回合步數與訓練 loss。
