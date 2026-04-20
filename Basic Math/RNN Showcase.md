# Basic RNN Showcase：簡單數值模擬歷程

> 參考你提供的數學筆記寫法，改寫成 **Basic RNN（最基本循環神經網路）** 的簡單數值範例。  
> 程度設定為 **高中生可讀**，重點放在「一步一步算出來」，避免過多矩陣。

---

## 目錄

1. [RNN 在做什麼？](#1-rnn-在做什麼)
2. [基本符號](#2-基本符號)
3. [前向傳播公式](#3-前向傳播公式)
4. [數值範例設定](#4-數值範例設定)
5. [Step 1 前向計算](#5-step-1-前向計算)
6. [Step 2 前向計算](#6-step-2-前向計算)
7. [Loss 計算](#7-loss-計算)
8. [輸出層反向傳播](#8-輸出層反向傳播)
9. [Hidden State 反向傳播](#9-hidden-state-反向傳播)
10. [時間方向的誤差回傳（BPTT）](#10-時間方向的誤差回傳bptt)
11. [參數更新的意思](#11-參數更新的意思)
12. [一句話總結](#12-一句話總結)

---

## 1. RNN 在做什麼？

一般神經網路看到一筆輸入，就只算一次輸出。  
但 **RNN（Recurrent Neural Network）** 不一樣，它會記住「前一刻的狀態」，再拿來幫助現在這一步的判斷。

所以 RNN 很適合處理：

- 句子
- 時間序列
- 語音
- 一個接一個出現的資料

最核心的想法是：

$$
\text{現在的 hidden state} = \text{現在的輸入} + \text{上一刻的記憶}
$$

---

## 2. 基本符號

我們先只看最簡單的情況：

- 輸入共有 2 維
- hidden state 共有 2 維
- 輸出共有 2 類（例如預測 A 或 B）
- 時間步只有 2 步：$t=1,2$

### 2.1 我們會用到的符號

| 符號 | 意義 |
|------|------|
| $\mathbf{x}_t$ | 第 $t$ 步的輸入 |
| $\mathbf{h}_t$ | 第 $t$ 步的 hidden state |
| $\mathbf{a}_t$ | hidden 層在進入 tanh 前的值 |
| $\mathbf{o}_t$ | 輸出層分數（logits） |
| $\hat{\mathbf{y}}_t$ | softmax 後的預測機率 |
| $\mathbf{y}_t^*$ | 真實答案（one-hot） |
| $\mathcal{L}_t$ | 第 $t$ 步 loss |
| $\mathcal{L}$ | 全部時間步的平均 loss |

---

## 3. 前向傳播公式

Basic RNN 的前向傳播很簡單：

### 3.1 Hidden state 更新

$$
\mathbf{a}_t = W_{xh}\mathbf{x}_t + W_{hh}\mathbf{h}_{t-1} + \mathbf{b}_h
$$

$$
\mathbf{h}_t = \tanh(\mathbf{a}_t)
$$

意思是：

- 先看現在輸入 $\mathbf{x}_t$
- 再看前一刻記憶 $\mathbf{h}_{t-1}$
- 加總之後
- 經過 $\tanh$ 壓到 $-1$ 到 $1$ 之間

---

### 3.2 輸出層

$$
\mathbf{o}_t = W_{hy}\mathbf{h}_t + \mathbf{b}_y
$$

再經過 softmax：

$$
\hat{\mathbf{y}}_t = \text{softmax}(\mathbf{o}_t)
$$

---

### 3.3 Loss

若總共有兩步，就取平均：

$$
\mathcal{L} = \frac{1}{2}(\mathcal{L}_1 + \mathcal{L}_2)
$$

每一步用 cross entropy：

$$
\mathcal{L}_t = -\log \hat{\mathbf{y}}_t[y_t^*]
$$

---

## 4. 數值範例設定

我們直接做一個很小的例子。

### 4.1 兩步輸入

第一步輸入：

$$
\mathbf{x}_1 = 
\begin{bmatrix}
1\\
0
\end{bmatrix}
\qquad
(\text{可想成 token A})
$$

第二步輸入：

$$
\mathbf{x}_2 =
\begin{bmatrix}
0\\
1
\end{bmatrix}
\qquad
(\text{可想成 token B})
$$

初始 hidden state 設成：

$$
\mathbf{h}_0 =
\begin{bmatrix}
0\\
0
\end{bmatrix}
$$

---

### 4.2 參數設定

輸入到 hidden：

$$
W_{xh}=
\begin{bmatrix}
0.5 & -0.1\\
0.3 & 0.8
\end{bmatrix}
$$

hidden 到 hidden：

$$
W_{hh}=
\begin{bmatrix}
0.4 & 0.2\\
-0.3 & 0.1
\end{bmatrix}
$$

hidden bias：

$$
\mathbf{b}_h=
\begin{bmatrix}
0.1\\
-0.2
\end{bmatrix}
$$

hidden 到 output：

$$
W_{hy}=
\begin{bmatrix}
0.7 & -0.2\\
-0.4 & 0.6
\end{bmatrix}
$$

output bias：

$$
\mathbf{b}_y=
\begin{bmatrix}
0.05\\
-0.05
\end{bmatrix}
$$

---

### 4.3 真實答案

我們假設：

- 第一步正確答案是類別 1
- 第二步正確答案是類別 2

所以 one-hot 寫成：

$$
\mathbf{y}_1^*=
\begin{bmatrix}
1\\
0
\end{bmatrix},
\qquad
\mathbf{y}_2^*=
\begin{bmatrix}
0\\
1
\end{bmatrix}
$$

---

## 5. Step 1 前向計算

### 5.1 先算 hidden 前的加總

因為 $\mathbf{h}_0=\mathbf{0}$，所以第一步比較單純：

$$
\mathbf{a}_1 = W_{xh}\mathbf{x}_1 + W_{hh}\mathbf{h}_0 + \mathbf{b}_h
$$

先算：

$$
W_{xh}\mathbf{x}_1=
\begin{bmatrix}
0.5 & -0.1\\
0.3 & 0.8
\end{bmatrix}
\begin{bmatrix}
1\\
0
\end{bmatrix}
=
\begin{bmatrix}
0.5\\
0.3
\end{bmatrix}
$$

又因為 $\mathbf{h}_0=\mathbf{0}$：

$$
W_{hh}\mathbf{h}_0=
\begin{bmatrix}
0\\
0
\end{bmatrix}
$$

所以：

$$
\mathbf{a}_1=
\begin{bmatrix}
0.5\\
0.3
\end{bmatrix}
+
\begin{bmatrix}
0.1\\
-0.2
\end{bmatrix}
=
\begin{bmatrix}
0.6\\
0.1
\end{bmatrix}
$$

---

### 5.2 經過 tanh

$$
\mathbf{h}_1=\tanh(\mathbf{a}_1)
$$

近似值：

$$
\tanh(0.6)\approx 0.537
\qquad
\tanh(0.1)\approx 0.100
$$

所以：

$$
\mathbf{h}_1 \approx
\begin{bmatrix}
0.537\\
0.100
\end{bmatrix}
$$

---

### 5.3 算輸出 logits

$$
\mathbf{o}_1 = W_{hy}\mathbf{h}_1 + \mathbf{b}_y
$$

先算：

$$
W_{hy}\mathbf{h}_1=
\begin{bmatrix}
0.7 & -0.2\\
-0.4 & 0.6
\end{bmatrix}
\begin{bmatrix}
0.537\\
0.100
\end{bmatrix}
=
\begin{bmatrix}
0.7(0.537)-0.2(0.100)\\
-0.4(0.537)+0.6(0.100)
\end{bmatrix}
$$

$$
=
\begin{bmatrix}
0.3559\\
-0.1548
\end{bmatrix}
$$

再加上 bias：

$$
\mathbf{o}_1=
\begin{bmatrix}
0.3559\\
-0.1548
\end{bmatrix}
+
\begin{bmatrix}
0.05\\
-0.05
\end{bmatrix}
=
\begin{bmatrix}
0.4059\\
-0.2048
\end{bmatrix}
$$

---

### 5.4 經過 softmax

$$
\hat{\mathbf{y}}_1=\text{softmax}(\mathbf{o}_1)
$$

兩個數做 softmax 時，只要算：

$$
e^{0.4059}\approx 1.501,\qquad e^{-0.2048}\approx 0.815
$$

總和：

$$
1.501+0.815=2.316
$$

所以：

$$
\hat{\mathbf{y}}_1 \approx
\begin{bmatrix}
1.501/2.316\\
0.815/2.316
\end{bmatrix}
=
\begin{bmatrix}
0.648\\
0.352
\end{bmatrix}
$$

---

## 6. Step 2 前向計算

這一步關鍵在於：  
**第二步不只看 $\mathbf{x}_2$，還會吃到第一步留下的 $\mathbf{h}_1$。**

### 6.1 先算 hidden 前的加總

$$
\mathbf{a}_2 = W_{xh}\mathbf{x}_2 + W_{hh}\mathbf{h}_1 + \mathbf{b}_h
$$

先算輸入部分：

$$
W_{xh}\mathbf{x}_2=
\begin{bmatrix}
0.5 & -0.1\\
0.3 & 0.8
\end{bmatrix}
\begin{bmatrix}
0\\
1
\end{bmatrix}
=
\begin{bmatrix}
-0.1\\
0.8
\end{bmatrix}
$$

再算前一步 hidden 的影響：

$$
W_{hh}\mathbf{h}_1=
\begin{bmatrix}
0.4 & 0.2\\
-0.3 & 0.1
\end{bmatrix}
\begin{bmatrix}
0.537\\
0.100
\end{bmatrix}
$$

$$
=
\begin{bmatrix}
0.4(0.537)+0.2(0.100)\\
-0.3(0.537)+0.1(0.100)
\end{bmatrix}
=
\begin{bmatrix}
0.2348\\
-0.1511
\end{bmatrix}
$$

所以：

$$
\mathbf{a}_2=
\begin{bmatrix}
-0.1\\
0.8
\end{bmatrix}
+
\begin{bmatrix}
0.2348\\
-0.1511
\end{bmatrix}
+
\begin{bmatrix}
0.1\\
-0.2
\end{bmatrix}
=
\begin{bmatrix}
0.2348\\
0.4489
\end{bmatrix}
$$

---

### 6.2 經過 tanh

$$
\mathbf{h}_2=\tanh(\mathbf{a}_2)
$$

近似：

$$
\tanh(0.2348)\approx 0.231
\qquad
\tanh(0.4489)\approx 0.421
$$

所以：

$$
\mathbf{h}_2\approx
\begin{bmatrix}
0.231\\
0.421
\end{bmatrix}
$$

---

### 6.3 算輸出 logits

$$
\mathbf{o}_2=W_{hy}\mathbf{h}_2+\mathbf{b}_y
$$

$$
W_{hy}\mathbf{h}_2=
\begin{bmatrix}
0.7 & -0.2\\
-0.4 & 0.6
\end{bmatrix}
\begin{bmatrix}
0.231\\
0.421
\end{bmatrix}
=
\begin{bmatrix}
0.7(0.231)-0.2(0.421)\\
-0.4(0.231)+0.6(0.421)
\end{bmatrix}
$$

$$
=
\begin{bmatrix}
0.0775\\
0.1602
\end{bmatrix}
$$

加上 bias：

$$
\mathbf{o}_2=
\begin{bmatrix}
0.1275\\
0.1102
\end{bmatrix}
$$

---

### 6.4 經過 softmax

$$
\hat{\mathbf{y}}_2=\text{softmax}(\mathbf{o}_2)
$$

$$
e^{0.1275}\approx 1.136,\qquad e^{0.1102}\approx 1.116
$$

總和：

$$
1.136+1.116=2.252
$$

所以：

$$
\hat{\mathbf{y}}_2\approx
\begin{bmatrix}
0.504\\
0.496
\end{bmatrix}
$$

---

## 7. Loss 計算

### 7.1 第一步 loss

第一步正確答案是第 1 類，所以：

$$
\mathcal{L}_1=-\log(0.648)\approx 0.434
$$

### 7.2 第二步 loss

第二步正確答案是第 2 類，所以：

$$
\mathcal{L}_2=-\log(0.496)\approx 0.701
$$

### 7.3 平均 loss

$$
\mathcal{L}=\frac{1}{2}(0.434+0.701)=0.5675
$$

---

## 8. 輸出層反向傳播

在 softmax + cross entropy 下，有一個非常常用的結果：

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{o}_t}
=
\frac{1}{2}(\hat{\mathbf{y}}_t-\mathbf{y}_t^*)
$$

前面的 $\frac{1}{2}$ 是因為總 loss 取了兩步平均。

---

### 8.1 Step 2 的輸出誤差

$$
\hat{\mathbf{y}}_2\approx
\begin{bmatrix}
0.504\\
0.496
\end{bmatrix},
\qquad
\mathbf{y}_2^*=
\begin{bmatrix}
0\\
1
\end{bmatrix}
$$

所以：

$$
\delta_2^o=
\frac{1}{2}
\left(
\hat{\mathbf{y}}_2-\mathbf{y}_2^*
\right)
=
\frac{1}{2}
\begin{bmatrix}
0.504\\
-0.504
\end{bmatrix}
=
\begin{bmatrix}
0.252\\
-0.252
\end{bmatrix}
$$

---

### 8.2 Step 1 的輸出誤差

$$
\hat{\mathbf{y}}_1\approx
\begin{bmatrix}
0.648\\
0.352
\end{bmatrix},
\qquad
\mathbf{y}_1^*=
\begin{bmatrix}
1\\
0
\end{bmatrix}
$$

所以：

$$
\delta_1^o=
\frac{1}{2}
\left(
\hat{\mathbf{y}}_1-\mathbf{y}_1^*
\right)
=
\frac{1}{2}
\begin{bmatrix}
-0.352\\
0.352
\end{bmatrix}
=
\begin{bmatrix}
-0.176\\
0.176
\end{bmatrix}
$$

---

### 8.3 輸出層權重梯度的直覺

因為：

$$
\mathbf{o}_t=W_{hy}\mathbf{h}_t+\mathbf{b}_y
$$

所以每一步對 $W_{hy}$ 的梯度可以理解成：

$$
\frac{\partial \mathcal{L}}{\partial W_{hy}}^{(t)}
=
\delta_t^o(\mathbf{h}_t)^T
$$

---

### 8.4 Step 2 對 $W_{hy}$ 的貢獻

$$
\delta_2^o=
\begin{bmatrix}
0.252\\
-0.252
\end{bmatrix},
\qquad
\mathbf{h}_2^T=
\begin{bmatrix}
0.231 & 0.421
\end{bmatrix}
$$

所以：

$$
\frac{\partial \mathcal{L}}{\partial W_{hy}}^{(t=2)}
=
\begin{bmatrix}
0.252\\
-0.252
\end{bmatrix}
\begin{bmatrix}
0.231 & 0.421
\end{bmatrix}
$$

$$
\approx
\begin{bmatrix}
0.058 & 0.106\\
-0.058 & -0.106
\end{bmatrix}
$$

---

### 8.5 Step 1 對 $W_{hy}$ 的貢獻

$$
\delta_1^o=
\begin{bmatrix}
-0.176\\
0.176
\end{bmatrix},
\qquad
\mathbf{h}_1^T=
\begin{bmatrix}
0.537 & 0.100
\end{bmatrix}
$$

所以：

$$
\frac{\partial \mathcal{L}}{\partial W_{hy}}^{(t=1)}
=
\begin{bmatrix}
-0.176\\
0.176
\end{bmatrix}
\begin{bmatrix}
0.537 & 0.100
\end{bmatrix}
$$

$$
\approx
\begin{bmatrix}
-0.095 & -0.018\\
0.095 & 0.018
\end{bmatrix}
$$

---

### 8.6 合起來的輸出層梯度

$$
\frac{\partial \mathcal{L}}{\partial W_{hy}}
=
\text{Step 1}+\text{Step 2}
$$

$$
\approx
\begin{bmatrix}
-0.095 & -0.018\\
0.095 & 0.018
\end{bmatrix}
+
\begin{bmatrix}
0.058 & 0.106\\
-0.058 & -0.106
\end{bmatrix}
$$

$$
\approx
\begin{bmatrix}
-0.037 & 0.088\\
0.037 & -0.088
\end{bmatrix}
$$

---

## 9. Hidden State 反向傳播

輸出層的誤差，會先傳回 hidden state。

因為：

$$
\mathbf{o}_t = W_{hy}\mathbf{h}_t+\mathbf{b}_y
$$

所以：

$$
\delta_t^h = W_{hy}^T\delta_t^o
$$

---

### 9.1 Step 2 傳回 hidden

$$
W_{hy}^T=
\begin{bmatrix}
0.7 & -0.4\\
-0.2 & 0.6
\end{bmatrix}
$$

$$
\delta_2^h=
W_{hy}^T\delta_2^o
=
\begin{bmatrix}
0.7 & -0.4\\
-0.2 & 0.6
\end{bmatrix}
\begin{bmatrix}
0.252\\
-0.252
\end{bmatrix}
$$

第一個分量：

$$
0.7(0.252)+(-0.4)(-0.252)=0.1764+0.1008=0.2772
$$

第二個分量：

$$
-0.2(0.252)+0.6(-0.252)=-0.0504-0.1512=-0.2016
$$

所以：

$$
\delta_2^h=
\begin{bmatrix}
0.2772\\
-0.2016
\end{bmatrix}
$$

---

### 9.2 經過 tanh 的 backward

因為：

$$
\mathbf{h}_2=\tanh(\mathbf{a}_2)
$$

所以要乘上：

$$
1-\mathbf{h}_2^2
$$

而

$$
\mathbf{h}_2\approx
\begin{bmatrix}
0.231\\
0.421
\end{bmatrix}
\Rightarrow
1-\mathbf{h}_2^2
\approx
\begin{bmatrix}
0.947\\
0.823
\end{bmatrix}
$$

因此：

$$
\delta_2^a=\delta_2^h\odot(1-\mathbf{h}_2^2)
$$

$$
\approx
\begin{bmatrix}
0.2772\\
-0.2016
\end{bmatrix}
\odot
\begin{bmatrix}
0.947\\
0.823
\end{bmatrix}
=
\begin{bmatrix}
0.262\\
-0.166
\end{bmatrix}
$$

這個 $\delta_2^a$，就是第二步在 hidden 內部真正往回傳的誤差。

---

## 10. 時間方向的誤差回傳（BPTT）

RNN 和一般前饋神經網路最大的不同就在這裡：  
**第二步的誤差，會沿著時間回傳到第一步。**

這件事叫做：

$$
\text{BPTT} = \text{Backpropagation Through Time}
$$

---

### 10.1 Step 2 傳回 Step 1

因為：

$$
\mathbf{a}_2 = W_{xh}\mathbf{x}_2 + W_{hh}\mathbf{h}_1 + \mathbf{b}_h
$$

所以第二步的誤差會經過 $W_{hh}^T$ 傳回前一步：

$$
\delta_{1}^{h,\text{future}} = W_{hh}^T\delta_2^a
$$

而

$$
W_{hh}^T=
\begin{bmatrix}
0.4 & -0.3\\
0.2 & 0.1
\end{bmatrix}
$$

所以：

$$
\delta_{1}^{h,\text{future}}
=
\begin{bmatrix}
0.4 & -0.3\\
0.2 & 0.1
\end{bmatrix}
\begin{bmatrix}
0.262\\
-0.166
\end{bmatrix}
$$

第一個分量：

$$
0.4(0.262)+(-0.3)(-0.166)=0.1048+0.0498=0.1546
$$

第二個分量：

$$
0.2(0.262)+0.1(-0.166)=0.0524-0.0166=0.0358
$$

所以：

$$
\delta_{1}^{h,\text{future}}
\approx
\begin{bmatrix}
0.155\\
0.036
\end{bmatrix}
$$

---

### 10.2 Step 1 自己本身也有輸出誤差

Step 1 不只收到未來傳回來的誤差，它自己也有輸出層誤差：

$$
\delta_1^h = W_{hy}^T\delta_1^o + \delta_1^{h,\text{future}}
$$

先算：

$$
W_{hy}^T\delta_1^o
=
\begin{bmatrix}
0.7 & -0.4\\
-0.2 & 0.6
\end{bmatrix}
\begin{bmatrix}
-0.176\\
0.176
\end{bmatrix}
$$

第一個分量：

$$
0.7(-0.176)+(-0.4)(0.176)=-0.1232-0.0704=-0.1936
$$

第二個分量：

$$
-0.2(-0.176)+0.6(0.176)=0.0352+0.1056=0.1408
$$

所以：

$$
W_{hy}^T\delta_1^o=
\begin{bmatrix}
-0.1936\\
0.1408
\end{bmatrix}
$$

再加上未來傳回來的部分：

$$
\delta_1^h=
\begin{bmatrix}
-0.1936\\
0.1408
\end{bmatrix}
+
\begin{bmatrix}
0.155\\
0.036
\end{bmatrix}
=
\begin{bmatrix}
-0.0386\\
0.1768
\end{bmatrix}
$$

---

### 10.3 再穿過 tanh

因為：

$$
\mathbf{h}_1\approx
\begin{bmatrix}
0.537\\
0.100
\end{bmatrix}
$$

所以：

$$
1-\mathbf{h}_1^2
\approx
\begin{bmatrix}
1-0.537^2\\
1-0.100^2
\end{bmatrix}
=
\begin{bmatrix}
0.712\\
0.990
\end{bmatrix}
$$

因此：

$$
\delta_1^a=\delta_1^h\odot(1-\mathbf{h}_1^2)
$$

$$
\approx
\begin{bmatrix}
-0.0386\\
0.1768
\end{bmatrix}
\odot
\begin{bmatrix}
0.712\\
0.990
\end{bmatrix}
=
\begin{bmatrix}
-0.0275\\
0.175
\end{bmatrix}
$$

這表示：  
第一步 hidden 的誤差，既包含「自己預測錯的責任」，也包含「它影響了第二步，所以要分攤第二步責任」。

這就是 RNN 最重要的觀念。

---

## 11. 參數更新的意思

有了梯度之後，就可以做梯度下降：

$$
W \leftarrow W-\alpha \frac{\partial \mathcal{L}}{\partial W}
$$

其中 $\alpha$ 是學習率。

如果某個梯度是正的，代表這個權重太大了，就減少一點。  
如果某個梯度是負的，代表這個權重太小了，就增加一點。

所以訓練本質上就是：

1. 先前向算出預測
2. 看和答案差多少
3. 把誤差往回傳
4. 每個參數小修正
5. 重複很多次

---

## 12. 一句話總結

$$
\text{RNN 的現在，不只由現在輸入決定，也由過去記憶決定}
$$

而在訓練時：

$$
\text{後面的錯誤，會沿著時間傳回前面}
$$

這就是 Basic RNN 的核心。

---

## 最後整理：你應該記住的 4 條式子

### 1. hidden 更新

$$
\mathbf{a}_t = W_{xh}\mathbf{x}_t + W_{hh}\mathbf{h}_{t-1} + \mathbf{b}_h
$$

$$
\mathbf{h}_t = \tanh(\mathbf{a}_t)
$$

### 2. 輸出

$$
\mathbf{o}_t = W_{hy}\mathbf{h}_t + \mathbf{b}_y
$$

$$
\hat{\mathbf{y}}_t = \text{softmax}(\mathbf{o}_t)
$$

### 3. 輸出層誤差

$$
\delta_t^o=\frac{\partial \mathcal{L}}{\partial \mathbf{o}_t}
$$

在這份範例中：

$$
\delta_t^o=\frac{1}{2}(\hat{\mathbf{y}}_t-\mathbf{y}_t^*)
$$

### 4. 時間回傳

$$
\delta_{t-1}^{h,\text{future}}=W_{hh}^T\delta_t^a
$$

也就是：

$$
\text{未來的誤差，會傳回過去}
$$

---

*這份文件刻意用「少量向量、少量矩陣、明確數值代入」來呈現 Basic RNN 的 forward / backward 核心。若要再往下一步延伸，就可以接到 many-to-many RNN、Seq2Seq、LSTM。*
