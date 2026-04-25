# Bidirectional Encoder 的數學推導：為什麼 RNN 需要 Bidirectional Encoder

> 適合大學一年級程度，從零開始理解 **Bidirectional Encoder** 的動機、前向傳播、雙向 hidden state、損失函數，以及向前 RNN 與向後 RNN 的反向傳播與時間反向傳播（BPTT）。
>
> 本文件聚焦於 **Bidirectional RNN Encoder** 本身，不討論 Attention。若要與 Decoder 或 Attention 結合，可以將本文件中的雙向 encoder states 作為後續模組的輸入表示。

---

## 目錄

1. [基本符號定義](#1-基本符號定義)
2. [為什麼 RNN 需要 Bidirectional Encoder](#2-為什麼-rnn-需要-bidirectional-encoder)
3. [單向 RNN Encoder 的限制](#3-單向-rnn-encoder-的限制)
4. [Bidirectional Encoder 的核心想法](#4-bidirectional-encoder-的核心想法)
5. [Forward Encoder 的前向傳播](#5-forward-encoder-的前向傳播)
6. [Backward Encoder 的前向傳播](#6-backward-encoder-的前向傳播)
7. [雙向 hidden state 的組合方式](#7-雙向-hidden-state-的組合方式)
8. [輸出層與損失函數](#8-輸出層與損失函數)
9. [反向傳播總覽](#9-反向傳播總覽)
10. [輸出層的反向傳播](#10-輸出層的反向傳播)
11. [雙向 hidden state 拆分梯度](#11-雙向-hidden-state-拆分梯度)
12. [Forward Encoder 的時間反向傳播 BPTT](#12-forward-encoder-的時間反向傳播-bptt)
13. [Backward Encoder 的時間反向傳播 BPTT](#13-backward-encoder-的時間反向傳播-bptt)
14. [Forward 與 Backward 的參數梯度](#14-forward-與-backward-的參數梯度)
15. [Bidirectional Encoder 為什麼改善資訊表示](#15-bidirectional-encoder-為什麼改善資訊表示)
16. [Bidirectional Encoder 為什麼改善梯度傳遞](#16-bidirectional-encoder-為什麼改善梯度傳遞)
17. [完整流程整理](#17-完整流程整理)
18. [核心結論](#18-核心結論)

---

## 1. 基本符號定義

假設輸入序列長度為 $T$。

輸入序列為

$$
x=(x_1,x_2,\ldots,x_T)
$$

其中 $x_t$ 是第 $t$ 個輸入 token 的 embedding 向量。

Bidirectional Encoder 會同時建立兩個方向的 hidden states：

$$
\overrightarrow{h}_t
$$

表示 forward RNN 在第 $t$ 個位置的 hidden state，也就是從左到右讀取序列後得到的表示。

$$
\overleftarrow{h}_t
$$

表示 backward RNN 在第 $t$ 個位置的 hidden state，也就是從右到左讀取序列後得到的表示。

最後將兩者組合成第 $t$ 個位置的雙向表示：

$$
h_t=[\overrightarrow{h}_t;\overleftarrow{h}_t]
$$

其中 $[\,;\,]$ 表示向量串接 concatenation。

### 1.1 主要符號表

| 符號 | 意義 |
|---|---|
| $T$ | 輸入序列長度 |
| $x_t$ | 第 $t$ 個輸入 token 的 embedding |
| $\overrightarrow{h}_t$ | forward RNN 在第 $t$ 步的 hidden state |
| $\overleftarrow{h}_t$ | backward RNN 在第 $t$ 步的 hidden state |
| $h_t$ | 第 $t$ 個位置的雙向 hidden state |
| $\overrightarrow{a}_t$ | forward RNN 的 pre-activation |
| $\overleftarrow{a}_t$ | backward RNN 的 pre-activation |
| $\hat{y}_t$ | 第 $t$ 個位置的預測機率分佈 |
| $z_t$ | softmax 前的 logits |
| $\mathcal{L}$ | 整體 loss |
| $V$ | 標籤或字彙類別數 |
| $\odot$ | 逐元素相乘 |

---

## 2. 為什麼 RNN 需要 Bidirectional Encoder

傳統單向 RNN 在第 $t$ 個位置的 hidden state 定義為

$$
h_t=\phi(W_xx_t+W_hh_{t-1}+b_h)
$$

因此 $h_t$ 只能依賴目前位置與過去位置：

$$
h_t=f(x_1,x_2,\ldots,x_t)
$$

也就是說，當模型要判斷第 $t$ 個 token 的語意或標籤時，它只看得到左側上下文，無法使用右側上下文。

但是很多序列任務中，第 $t$ 個位置的正確解讀需要同時依賴左側與右側資訊。

例如句子：

$$
\text{I went to the bank to deposit money}
$$

在讀到

$$
\text{bank}
$$

時，若只看左側

$$
\text{I went to the bank}
$$

模型可能無法確定是河岸還是銀行。後面的

$$
\text{to deposit money}
$$

提供了關鍵資訊。

因此，單向 RNN 的限制不是它不能處理序列，而是它在每個位置只能建立單向上下文表示。

Bidirectional Encoder 的核心想法是：對每個位置 $t$，同時建立

$$
\overrightarrow{h}_t=f_{\rightarrow}(x_1,\ldots,x_t)
$$

與

$$
\overleftarrow{h}_t=f_{\leftarrow}(x_t,\ldots,x_T)
$$

再組合成

$$
h_t=[\overrightarrow{h}_t;\overleftarrow{h}_t]
$$

因此

$$
h_t=f(x_1,\ldots,x_t,x_{t+1},\ldots,x_T)
$$

也就是讓每個位置的表示同時包含左側與右側上下文。

---

## 3. 單向 RNN Encoder 的限制

### 3.1 單向資訊限制

單向 RNN 的遞迴為

$$
a_t=W_xx_t+W_hh_{t-1}+b_h
$$

$$
h_t=\tanh(a_t)
$$

展開後可知：

$$
h_t=\tanh(W_xx_t+W_h\tanh(W_xx_{t-1}+W_hh_{t-2}+b_h)+b_h)
$$

繼續展開：

$$
h_t=f(x_t,x_{t-1},\ldots,x_1)
$$

因此

$$
\frac{\partial h_t}{\partial x_k}=0
\quad \text{for } k>t
$$

這代表第 $t$ 個位置的 hidden state 不可能依賴未來 token $x_{t+1},\ldots,x_T$。

在需要完整句子上下文的任務中，例如詞性標註、命名實體辨識、語音辨識、機器翻譯 encoder 表示，這會造成資訊不足。

### 3.2 長距離梯度問題

若 loss 主要來自最後狀態 $h_T$，則較早位置 $h_t$ 的梯度需經過很長的 recurrent Jacobian 連乘：

$$
\frac{\partial \mathcal{L}}{\partial h_t}
=
\frac{\partial \mathcal{L}}{\partial h_T}
\frac{\partial h_T}{\partial h_t}
$$

其中

$$
\frac{\partial h_T}{\partial h_t}
=
\prod_{k=t+1}^{T}
\frac{\partial h_k}{\partial h_{k-1}}
$$

對於 tanh RNN，

$$
h_k=\tanh(a_k)
$$

$$
a_k=W_xx_k+W_hh_{k-1}+b_h
$$

所以

$$
\frac{\partial h_k}{\partial h_{k-1}}
=
\operatorname{diag}(1-h_k\odot h_k)W_h
$$

因此

$$
\frac{\partial h_T}{\partial h_t}
=
\prod_{k=t+1}^{T}
\operatorname{diag}(1-h_k\odot h_k)W_h
$$

如果連乘矩陣的范數長期小於 $1$，梯度容易消失；如果大於 $1$，梯度容易爆炸。

這是單向 RNN 學習長距離依賴時的主要困難之一。

---

## 4. Bidirectional Encoder 的核心想法

Bidirectional Encoder 使用兩個獨立的 RNN。

第一個 RNN 從左到右讀取輸入：

$$
x_1 \rightarrow x_2 \rightarrow \cdots \rightarrow x_T
$$

它產生 forward hidden states：

$$
\overrightarrow{h}_1,\overrightarrow{h}_2,\ldots,\overrightarrow{h}_T
$$

第二個 RNN 從右到左讀取輸入：

$$
x_T \rightarrow x_{T-1} \rightarrow \cdots \rightarrow x_1
$$

它產生 backward hidden states：

$$
\overleftarrow{h}_T,\overleftarrow{h}_{T-1},\ldots,\overleftarrow{h}_1
$$

對每個位置 $t$，將兩個方向的 hidden state 組合：

$$
h_t=[\overrightarrow{h}_t;\overleftarrow{h}_t]
$$

此時

$$
\overrightarrow{h}_t=f_{\rightarrow}(x_1,\ldots,x_t)
$$

而

$$
\overleftarrow{h}_t=f_{\leftarrow}(x_T,\ldots,x_t)
$$

因此

$$
h_t=[f_{\rightarrow}(x_1,\ldots,x_t);f_{\leftarrow}(x_T,\ldots,x_t)]
$$

也就是說，$h_t$ 同時包含位置 $t$ 左側與右側的資訊。

---

## 5. Forward Encoder 的前向傳播

Forward RNN 從 $t=1$ 到 $T$ 依序計算。

令初始狀態為

$$
\overrightarrow{h}_0=0
$$

對每個 $t=1,2,\ldots,T$：

$$
\overrightarrow{a}_t
=
\overrightarrow{W}_x x_t
+
\overrightarrow{W}_h \overrightarrow{h}_{t-1}
+
\overrightarrow{b}_h
$$

$$
\overrightarrow{h}_t
=
\tanh(\overrightarrow{a}_t)
$$

其中：

- $\overrightarrow{W}_x$ 將輸入 embedding 轉換到 hidden space
- $\overrightarrow{W}_h$ 描述 forward hidden state 的時間遞迴
- $\overrightarrow{b}_h$ 是 forward RNN 的 bias
- $\overrightarrow{h}_t$ 包含 $x_1,\ldots,x_t$ 的資訊

因此 forward hidden state 可寫成

$$
\overrightarrow{h}_t
=
f_{\rightarrow}(x_1,x_2,\ldots,x_t)
$$

---

## 6. Backward Encoder 的前向傳播

Backward RNN 從 $t=T$ 到 $1$ 反向計算。

令初始狀態為

$$
\overleftarrow{h}_{T+1}=0
$$

對每個 $t=T,T-1,\ldots,1$：

$$
\overleftarrow{a}_t
=
\overleftarrow{W}_x x_t
+
\overleftarrow{W}_h \overleftarrow{h}_{t+1}
+
\overleftarrow{b}_h
$$

$$
\overleftarrow{h}_t
=
\tanh(\overleftarrow{a}_t)
$$

其中：

- $\overleftarrow{W}_x$ 將輸入 embedding 轉換到 backward hidden space
- $\overleftarrow{W}_h$ 描述 backward hidden state 的時間遞迴
- $\overleftarrow{b}_h$ 是 backward RNN 的 bias
- $\overleftarrow{h}_t$ 包含 $x_T,x_{T-1},\ldots,x_t$ 的資訊

因此 backward hidden state 可寫成

$$
\overleftarrow{h}_t
=
f_{\leftarrow}(x_T,x_{T-1},\ldots,x_t)
$$

注意，雖然稱為 backward encoder，但它在數學上仍然是一個普通 RNN，只是處理序列的方向相反。

---

## 7. 雙向 hidden state 的組合方式

最常見的組合方式是 concatenation：

$$
h_t=[\overrightarrow{h}_t;\overleftarrow{h}_t]
$$

若

$$
\overrightarrow{h}_t\in\mathbb{R}^{d_h}
$$

且

$$
\overleftarrow{h}_t\in\mathbb{R}^{d_h}
$$

則

$$
h_t\in\mathbb{R}^{2d_h}
$$

也可以用加總：

$$
h_t=\overrightarrow{h}_t+\overleftarrow{h}_t
$$

或線性投影：

$$
h_t=W_b[\overrightarrow{h}_t;\overleftarrow{h}_t]+b_b
$$

但在基礎 Bidirectional RNN 中，最常見且最直接的是 concatenation：

$$
h_t=[\overrightarrow{h}_t;\overleftarrow{h}_t]
$$

原因是 concatenation 不會強迫兩個方向的資訊相加混合，而是保留兩個方向的完整表示，讓後續層自行學習如何使用。

---

## 8. 輸出層與損失函數

Bidirectional Encoder 可以用於多種任務。這裡以 sequence labeling 為例，例如每個輸入 token 都要預測一個標籤。

對每個位置 $t$，輸出層為

$$
z_t=W_oh_t+b_o
$$

$$
\hat{y}_t=\operatorname{softmax}(z_t)
$$

若標籤類別數為 $V$，則

$$
\hat{y}_t\in\mathbb{R}^{V}
$$

對第 $k$ 個類別：

$$
\hat{y}_{t,k}
=
\frac{\exp(z_{t,k})}
{\sum_{m=1}^{V}\exp(z_{t,m})}
$$

若正確標籤 $y_t$ 用 one-hot 向量表示，cross-entropy loss 為

$$
\mathcal{L}_t
=
-\sum_{k=1}^{V}y_{t,k}\log\hat{y}_{t,k}
$$

整個序列的 loss 為

$$
\mathcal{L}
=
\sum_{t=1}^{T}\mathcal{L}_t
$$

若正確類別 index 是 $g_t$，則

$$
\mathcal{L}_t=-\log\hat{y}_{t,g_t}
$$

---

## 9. 反向傳播總覽

整體計算圖可概括為

$$
\mathcal{L}
\rightarrow
\hat{y}_t
\rightarrow
z_t
\rightarrow
h_t
\rightarrow
(\overrightarrow{h}_t,\overleftarrow{h}_t)
$$

接著 forward RNN 的梯度沿著時間反方向傳播：

$$
\overrightarrow{h}_T
\rightarrow
\overrightarrow{h}_{T-1}
\rightarrow
\cdots
\rightarrow
\overrightarrow{h}_1
$$

而 backward RNN 的梯度沿著它自己的 recurrent connection 反向傳播。

因為 backward RNN 的前向計算方向是

$$
\overleftarrow{h}_{T}
\rightarrow
\overleftarrow{h}_{T-1}
\rightarrow
\cdots
\rightarrow
\overleftarrow{h}_{1}
$$

所以對它做 BPTT 時，梯度會沿著相反方向：

$$
\overleftarrow{h}_{1}
\rightarrow
\overleftarrow{h}_{2}
\rightarrow
\cdots
\rightarrow
\overleftarrow{h}_{T}
$$

這一點很重要：forward RNN 與 backward RNN 都做 BPTT，但它們的時間方向相反。

---

## 10. 輸出層的反向傳播

對每個位置 $t$：

$$
z_t=W_oh_t+b_o
$$

$$
\hat{y}_t=\operatorname{softmax}(z_t)
$$

$$
\mathcal{L}_t
=
-\sum_{k=1}^{V}y_{t,k}\log\hat{y}_{t,k}
$$

softmax 加 cross-entropy 有標準梯度：

$$
\frac{\partial \mathcal{L}_t}{\partial z_t}
=
\hat{y}_t-y_t
$$

記

$$
\delta_t^z=\hat{y}_t-y_t
$$

則輸出層參數梯度為

$$
\frac{\partial \mathcal{L}}{\partial W_o}
=
\sum_{t=1}^{T}\delta_t^z h_t^T
$$

$$
\frac{\partial \mathcal{L}}{\partial b_o}
=
\sum_{t=1}^{T}\delta_t^z
$$

傳回雙向 hidden state 的梯度為

$$
g_t^h
=
\frac{\partial \mathcal{L}}{\partial h_t}
=
W_o^T\delta_t^z
$$

若後續還有其他層，例如 CRF、MLP 或 attention，則 $g_t^h$ 需再加上那些路徑傳回來的梯度。本文件為了清楚推導，先只考慮 softmax 輸出層。

---

## 11. 雙向 hidden state 拆分梯度

因為

$$
h_t=[\overrightarrow{h}_t;\overleftarrow{h}_t]
$$

所以

$$
g_t^h
=
\frac{\partial \mathcal{L}}{\partial h_t}
$$

可以拆成兩部分：

$$
g_t^h=
[
\overrightarrow{g}_t^h;
\overleftarrow{g}_t^h
]
$$

其中

$$
\overrightarrow{g}_t^h
=
\frac{\partial \mathcal{L}}{\partial \overrightarrow{h}_t}
$$

$$
\overleftarrow{g}_t^h
=
\frac{\partial \mathcal{L}}{\partial \overleftarrow{h}_t}
$$

如果 $h_t$ 是 concatenation，這個拆分只是把向量前半段給 forward hidden state，後半段給 backward hidden state。

如果使用線性投影

$$
h_t=W_b[\overrightarrow{h}_t;\overleftarrow{h}_t]+b_b
$$

則必須先計算

$$
\frac{\partial \mathcal{L}}
{\partial [\overrightarrow{h}_t;\overleftarrow{h}_t]}
=
W_b^Tg_t^h
$$

再拆分成 forward 與 backward 兩部分。

---

## 12. Forward Encoder 的時間反向傳播 BPTT

Forward RNN 的前向傳播為

$$
\overrightarrow{a}_t
=
\overrightarrow{W}_x x_t
+
\overrightarrow{W}_h\overrightarrow{h}_{t-1}
+
\overrightarrow{b}_h
$$

$$
\overrightarrow{h}_t
=
\tanh(\overrightarrow{a}_t)
$$

因為 $\overrightarrow{h}_t$ 會直接影響輸出層，也會影響下一個 forward hidden state $\overrightarrow{h}_{t+1}$，所以它的總梯度是兩部分相加：

$$
\overrightarrow{\bar{g}}_t^h
=
\overrightarrow{g}_t^h
+
\overrightarrow{W}_h^T\overrightarrow{\delta}_{t+1}
$$

其中

$$
\overrightarrow{g}_t^h
=
\frac{\partial \mathcal{L}}{\partial \overrightarrow{h}_t}
\Bigg|_{\text{output}}
$$

而

$$
\overrightarrow{W}_h^T\overrightarrow{\delta}_{t+1}
$$

是從下一個時間步 $\overrightarrow{h}_{t+1}$ 傳回來的 recurrent gradient。

若 $t=T$，則沒有下一個 forward state，因此

$$
\overrightarrow{\delta}_{T+1}=0
$$

接著，因為

$$
\overrightarrow{h}_t=\tanh(\overrightarrow{a}_t)
$$

所以

$$
\frac{\partial \overrightarrow{h}_t}
{\partial \overrightarrow{a}_t}
=
1-\overrightarrow{h}_t\odot\overrightarrow{h}_t
$$

因此 forward RNN 在第 $t$ 步的 pre-activation 梯度為

$$
\overrightarrow{\delta}_t
=
\frac{\partial \mathcal{L}}
{\partial \overrightarrow{a}_t}
=
\overrightarrow{\bar{g}}_t^h
\odot
(1-\overrightarrow{h}_t\odot\overrightarrow{h}_t)
$$

Forward BPTT 的計算順序是從 $t=T$ 到 $1$：

$$
T,T-1,\ldots,1
$$

也就是沿著 forward RNN 前向計算方向的相反方向回傳梯度。

---

## 13. Backward Encoder 的時間反向傳播 BPTT

Backward RNN 的前向傳播為

$$
\overleftarrow{a}_t
=
\overleftarrow{W}_x x_t
+
\overleftarrow{W}_h\overleftarrow{h}_{t+1}
+
\overleftarrow{b}_h
$$

$$
\overleftarrow{h}_t
=
\tanh(\overleftarrow{a}_t)
$$

注意這裡的 recurrent connection 是從 $\overleftarrow{h}_{t+1}$ 到 $\overleftarrow{h}_t$。

因此 $\overleftarrow{h}_t$ 會影響前一個索引位置的 backward state：

$$
\overleftarrow{h}_{t-1}
$$

因為

$$
\overleftarrow{a}_{t-1}
=
\overleftarrow{W}_x x_{t-1}
+
\overleftarrow{W}_h\overleftarrow{h}_{t}
+
\overleftarrow{b}_h
$$

所以在 backward RNN 中，$\overleftarrow{h}_t$ 的 recurrent 梯度來自 $\overleftarrow{\delta}_{t-1}$。

因此總梯度為

$$
\overleftarrow{\bar{g}}_t^h
=
\overleftarrow{g}_t^h
+
\overleftarrow{W}_h^T\overleftarrow{\delta}_{t-1}
$$

其中

$$
\overleftarrow{g}_t^h
=
\frac{\partial \mathcal{L}}
{\partial \overleftarrow{h}_t}
\Bigg|_{\text{output}}
$$

若 $t=1$，則沒有 $\overleftarrow{\delta}_0$，因此

$$
\overleftarrow{\delta}_{0}=0
$$

因為

$$
\overleftarrow{h}_t=\tanh(\overleftarrow{a}_t)
$$

所以

$$
\overleftarrow{\delta}_t
=
\frac{\partial \mathcal{L}}
{\partial \overleftarrow{a}_t}
=
\overleftarrow{\bar{g}}_t^h
\odot
(1-\overleftarrow{h}_t\odot\overleftarrow{h}_t)
$$

Backward BPTT 的計算順序是從 $t=1$ 到 $T$：

$$
1,2,\ldots,T
$$

這是因為 backward RNN 的前向計算方向是 $T$ 到 $1$，所以反向傳播方向就是 $1$ 到 $T$。

---

## 14. Forward 與 Backward 的參數梯度

### 14.1 Forward Encoder 參數梯度

Forward pre-activation 為

$$
\overrightarrow{a}_t
=
\overrightarrow{W}_x x_t
+
\overrightarrow{W}_h\overrightarrow{h}_{t-1}
+
\overrightarrow{b}_h
$$

由鏈式法則可得：

$$
\frac{\partial \mathcal{L}}
{\partial \overrightarrow{W}_x}
=
\sum_{t=1}^{T}
\overrightarrow{\delta}_t x_t^T
$$

$$
\frac{\partial \mathcal{L}}
{\partial \overrightarrow{W}_h}
=
\sum_{t=1}^{T}
\overrightarrow{\delta}_t \overrightarrow{h}_{t-1}^T
$$

$$
\frac{\partial \mathcal{L}}
{\partial \overrightarrow{b}_h}
=
\sum_{t=1}^{T}
\overrightarrow{\delta}_t
$$

傳回輸入 embedding 的 forward 梯度為

$$
\frac{\partial \mathcal{L}}
{\partial x_t}
\Bigg|_{\rightarrow}
=
\overrightarrow{W}_x^T\overrightarrow{\delta}_t
$$

### 14.2 Backward Encoder 參數梯度

Backward pre-activation 為

$$
\overleftarrow{a}_t
=
\overleftarrow{W}_x x_t
+
\overleftarrow{W}_h\overleftarrow{h}_{t+1}
+
\overleftarrow{b}_h
$$

因此：

$$
\frac{\partial \mathcal{L}}
{\partial \overleftarrow{W}_x}
=
\sum_{t=1}^{T}
\overleftarrow{\delta}_t x_t^T
$$

$$
\frac{\partial \mathcal{L}}
{\partial \overleftarrow{W}_h}
=
\sum_{t=1}^{T}
\overleftarrow{\delta}_t \overleftarrow{h}_{t+1}^T
$$

$$
\frac{\partial \mathcal{L}}
{\partial \overleftarrow{b}_h}
=
\sum_{t=1}^{T}
\overleftarrow{\delta}_t
$$

傳回輸入 embedding 的 backward 梯度為

$$
\frac{\partial \mathcal{L}}
{\partial x_t}
\Bigg|_{\leftarrow}
=
\overleftarrow{W}_x^T\overleftarrow{\delta}_t
$$

因為同一個輸入 $x_t$ 同時被 forward RNN 與 backward RNN 使用，所以輸入 embedding 的總梯度為

$$
\frac{\partial \mathcal{L}}
{\partial x_t}
=
\overrightarrow{W}_x^T\overrightarrow{\delta}_t
+
\overleftarrow{W}_x^T\overleftarrow{\delta}_t
$$

若 $x_t$ 是 embedding lookup 的結果，這個梯度會再傳回 embedding matrix 對應的 token row。

---

## 15. Bidirectional Encoder 為什麼改善資訊表示

單向 RNN 在第 $t$ 個位置的表示為

$$
h_t=f(x_1,\ldots,x_t)
$$

因此

$$
\frac{\partial h_t}{\partial x_k}=0
\quad \text{for } k>t
$$

也就是第 $t$ 個表示不可能直接使用右側資訊。

Bidirectional Encoder 的表示為

$$
h_t=
[
\overrightarrow{h}_t;
\overleftarrow{h}_t
]
$$

其中

$$
\overrightarrow{h}_t=f_{\rightarrow}(x_1,\ldots,x_t)
$$

$$
\overleftarrow{h}_t=f_{\leftarrow}(x_T,\ldots,x_t)
$$

因此

$$
h_t=f(x_1,\ldots,x_t,\ldots,x_T)
$$

此時第 $t$ 個位置的表示可以同時依賴任意位置的輸入 token：

$$
\frac{\partial h_t}{\partial x_k}
\neq 0
\quad \text{for many } k<t
$$

且

$$
\frac{\partial h_t}{\partial x_k}
\neq 0
\quad \text{for many } k>t
$$

因此 Bidirectional Encoder 對每個 token 建立的是整句上下文表示，而不是單側上下文表示。

---

## 16. Bidirectional Encoder 為什麼改善梯度傳遞

### 16.1 單向 RNN 的梯度路徑

在單向 RNN 中，如果某個早期 token $x_i$ 對後面位置 $h_t$ 有影響，其中 $i<t$，則梯度需要經過

$$
h_t
\rightarrow
h_{t-1}
\rightarrow
\cdots
\rightarrow
h_i
$$

對應的 Jacobian 連乘為

$$
\frac{\partial h_t}{\partial h_i}
=
\prod_{k=i+1}^{t}
\operatorname{diag}(1-h_k\odot h_k)W_h
$$

當 $t-i$ 很大時，梯度容易消失或爆炸。

### 16.2 Bidirectional Encoder 的雙向梯度路徑

Bidirectional Encoder 對位置 $t$ 的 loss 會同時傳回

$$
\overrightarrow{h}_t
$$

與

$$
\overleftarrow{h}_t
$$

因此輸入 $x_i$ 對位置 $t$ 的影響可以透過兩個方向建立。

若 $i<t$，forward RNN 提供路徑：

$$
x_i
\rightarrow
\overrightarrow{h}_i
\rightarrow
\overrightarrow{h}_{i+1}
\rightarrow
\cdots
\rightarrow
\overrightarrow{h}_t
$$

若 $i>t$，backward RNN 提供路徑：

$$
x_i
\rightarrow
\overleftarrow{h}_i
\rightarrow
\overleftarrow{h}_{i-1}
\rightarrow
\cdots
\rightarrow
\overleftarrow{h}_t
$$

因此對每個位置 $t$ 而言，左側資訊由 forward RNN 傳入，右側資訊由 backward RNN 傳入。

### 16.3 不是完全消除梯度消失，而是改善表示路徑

需要注意的是，Bidirectional Encoder 不會從數學上完全消除 RNN 的梯度消失問題。因為 forward RNN 與 backward RNN 本身仍然有 recurrent Jacobian 連乘。

但是它改善了兩件事。

第一，每個位置的表示不再只依賴單一方向。

單向 RNN：

$$
h_t=f(x_1,\ldots,x_t)
$$

Bidirectional Encoder：

$$
h_t=f(x_1,\ldots,x_t,\ldots,x_T)
$$

第二，對於需要右側上下文的任務，模型不需要把未來資訊硬塞進過去的 hidden state，而是由 backward RNN 直接建模右側資訊。

例如右側 token $x_k$，其中 $k>t$，對第 $t$ 個位置的 backward 表示影響為

$$
\frac{\partial \overleftarrow{h}_t}
{\partial \overleftarrow{h}_k}
=
\prod_{j=t}^{k-1}
\frac{\partial \overleftarrow{h}_j}
{\partial \overleftarrow{h}_{j+1}}
$$

其中

$$
\frac{\partial \overleftarrow{h}_j}
{\partial \overleftarrow{h}_{j+1}}
=
\operatorname{diag}(1-\overleftarrow{h}_j\odot\overleftarrow{h}_j)
\overleftarrow{W}_h
$$

這條路徑讓未來資訊能以 backward recurrent chain 的方式進入目前位置，而不是在單向 RNN 中完全不可見。

---

## 17. 完整流程整理

### 17.1 前向傳播

1. 將輸入 tokens 轉成 embeddings：

$$
x_1,x_2,\ldots,x_T
$$

2. Forward RNN 從左到右計算：

$$
\overrightarrow{h}_t
=
\tanh(
\overrightarrow{W}_x x_t
+
\overrightarrow{W}_h\overrightarrow{h}_{t-1}
+
\overrightarrow{b}_h
)
$$

3. Backward RNN 從右到左計算：

$$
\overleftarrow{h}_t
=
\tanh(
\overleftarrow{W}_x x_t
+
\overleftarrow{W}_h\overleftarrow{h}_{t+1}
+
\overleftarrow{b}_h
)
$$

4. 組合雙向表示：

$$
h_t=[\overrightarrow{h}_t;\overleftarrow{h}_t]
$$

5. 輸出 logits：

$$
z_t=W_oh_t+b_o
$$

6. 輸出機率分佈：

$$
\hat{y}_t=\operatorname{softmax}(z_t)
$$

7. 計算 cross-entropy loss：

$$
\mathcal{L}
=
\sum_{t=1}^{T}
-\sum_{k=1}^{V}y_{t,k}\log\hat{y}_{t,k}
$$

### 17.2 反向傳播

1. 從 softmax 加 cross-entropy 得到

$$
\delta_t^z=\hat{y}_t-y_t
$$

2. 輸出層參數梯度為

$$
\frac{\partial \mathcal{L}}{\partial W_o}
=
\sum_{t=1}^{T}\delta_t^z h_t^T
$$

$$
\frac{\partial \mathcal{L}}{\partial b_o}
=
\sum_{t=1}^{T}\delta_t^z
$$

3. 傳回雙向 hidden state：

$$
g_t^h=W_o^T\delta_t^z
$$

4. 將 $g_t^h$ 拆成

$$
\overrightarrow{g}_t^h
$$

與

$$
\overleftarrow{g}_t^h
$$

5. Forward RNN 從 $T$ 到 $1$ 做 BPTT：

$$
\overrightarrow{\bar{g}}_t^h
=
\overrightarrow{g}_t^h
+
\overrightarrow{W}_h^T\overrightarrow{\delta}_{t+1}
$$

$$
\overrightarrow{\delta}_t
=
\overrightarrow{\bar{g}}_t^h
\odot
(1-\overrightarrow{h}_t\odot\overrightarrow{h}_t)
$$

6. Backward RNN 從 $1$ 到 $T$ 做 BPTT：

$$
\overleftarrow{\bar{g}}_t^h
=
\overleftarrow{g}_t^h
+
\overleftarrow{W}_h^T\overleftarrow{\delta}_{t-1}
$$

$$
\overleftarrow{\delta}_t
=
\overleftarrow{\bar{g}}_t^h
\odot
(1-\overleftarrow{h}_t\odot\overleftarrow{h}_t)
$$

7. 分別累加 forward 與 backward encoder 的參數梯度。

8. 輸入 embedding 的梯度同時來自兩個方向：

$$
\frac{\partial \mathcal{L}}
{\partial x_t}
=
\overrightarrow{W}_x^T\overrightarrow{\delta}_t
+
\overleftarrow{W}_x^T\overleftarrow{\delta}_t
$$

---

## 18. 核心結論

RNN 需要 Bidirectional Encoder，主要不是因為單向 RNN 無法處理序列，而是因為單向 RNN 在每個位置只能使用一側上下文。

單向 RNN 的位置表示為

$$
h_t=f(x_1,\ldots,x_t)
$$

所以它無法在第 $t$ 個位置使用未來資訊 $x_{t+1},\ldots,x_T$。

Bidirectional Encoder 建立兩個方向的 recurrent representation：

$$
\overrightarrow{h}_t=f_{\rightarrow}(x_1,\ldots,x_t)
$$

$$
\overleftarrow{h}_t=f_{\leftarrow}(x_T,\ldots,x_t)
$$

並組合成

$$
h_t=[\overrightarrow{h}_t;\overleftarrow{h}_t]
$$

因此每個位置的表示都能同時包含左側與右側上下文。

從反向傳播角度看，forward RNN 與 backward RNN 分別沿著相反方向做 BPTT：

$$
\overrightarrow{\delta}_t
=
\left(
\overrightarrow{g}_t^h
+
\overrightarrow{W}_h^T\overrightarrow{\delta}_{t+1}
\right)
\odot
(1-\overrightarrow{h}_t\odot\overrightarrow{h}_t)
$$

$$
\overleftarrow{\delta}_t
=
\left(
\overleftarrow{g}_t^h
+
\overleftarrow{W}_h^T\overleftarrow{\delta}_{t-1}
\right)
\odot
(1-\overleftarrow{h}_t\odot\overleftarrow{h}_t)
$$

這使模型能同時學習過去到現在、未來到現在的依賴關係。

因此 Bidirectional Encoder 的本質是：用兩個方向的 RNN，讓每個 token 的 hidden representation 變成完整上下文表示，而不是單向上下文表示。
