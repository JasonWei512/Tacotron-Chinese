# Tacotron-Chinese

## 预训练模型

[标贝数据集100K步模型](https://github.com/JasonWei512/Tacotron-Chinese/releases/download/Biaobei-100K/logs-tacotron.zip)

## 安装依赖

1. 安装 Python 3 和 Tensorflow（>=1.3）.

2. 安装依赖包：
   ```
   pip install -r requirements.txt
   ```

3. 升级 Scipy：
   ```
   pip install scipy -U
   ```

## 训练

1. **下载[标贝数据集](https://weixinxcxdb.oss-cn-beijing.aliyuncs.com/gwYinPinKu/BZNSYP.rar)，解压至 `~/tacotron`**
   
   目录结构如下：

   ```
   tacotron
     |- BZNSYP
         |- PhoneLabeling
         |- ProsodyLabeling
         |- Wave
   ```

2. **预处理数据**
   ```
   python preprocess.py --dataset biaobei
   ```

3. **训练模型（自动从最新 Checkpoint 继续）**
   ```
   python train.py
   ```

4. **从最新 Checkpoint 合成语音** 

   ```
   python eval.py --checkpoint ~/tacotron/logs-tacotron/
   ```