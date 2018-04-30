<H1>Programming Assignment 1: Recognition of Handwritten Digits by MultilayerPerceptrons</H1>

來自mit_neol:
https://hackmd.io/lnu1A1GXRaWeat_2kFHsFA?view

這個專案創建了擁有簡易功能的Multilayer Perceptrons，當變數nn被指派為類別NeuralNetMLP時，此時nn即可以簡易的方式建構出MLP網路架構。
![](https://i.imgur.com/RV80S54.png)
nn有兩個參數可調整，epochs次數、learning rate數alpha。
隨即我們呼叫nn裡的add_layer方法，以建構神經網路。
 ![](https://i.imgur.com/oq4uD0E.png)
上圖中是以一層50X25的隱藏層配合sigmoid激活函數，前一層為輸入層，後一層為輸出層(輸出層通常使用softmax居多)。
這邊我們可使用add_layer這個方法，建構不同架構的神經網路。
![](https://i.imgur.com/YBO3JFB.png)
![](https://i.imgur.com/PGEQBqg.png)
設計此種方法能很便捷的變換不同層數或神經元數，以測驗最適合我們的網路架構。
建完網路架構後，我們呼叫fit方法，進入訓練階段。
![](https://i.imgur.com/FVoZ0zW.png)
X_train,y_train是我們的訓練資料，因input資料為8組手寫數字(x,y)座標數值組成，我們先做正規畫以避免sigmoid在過大或過小的區間效果不彰。
![](https://i.imgur.com/kGY4tLM.png)
接著我們進入NeuralNetMLP類別詳細說明fit方法是如何訓練的。

fit方法裡對所有樣本作了epochs變數值的次數訓練，每一樣本將會一一進入神經網路裡，所以將會是epoch*train_data_len次數的訓練。
一個樣本的第一步是進入feedforward裡，經過每一次的![](https://i.imgur.com/MngALK4.png)和緊接著的激活函數![](https://i.imgur.com/ANBZq82.png)，目的是算出每一層神經層的a，比如說 : 輸入層+3個隱藏層+輸出層有5層，那將會有5個a被計算出來。
![](https://i.imgur.com/tQbIWL9.png)

Feedforward裡每層的W會乘上前一層的a，而第一層的W則是乘上input的特徵值，到了最後一層時，a輸出會是10個數字，也就是和每一個數字互相對應著，最大的就是神經網路計算出最有可能的數字。
![](https://i.imgur.com/zWJb6PH.png)
計算完每一層的a之後，進入backpropagation的步驟，如同其名，是由最後一層往回算，其目的是算出每一層的delta_W, delta_b，再以減掉原本的W, b，作為更新每一層的權重質，每一次的樣本輸入都會做一次的feedforward和backpropagation，全部樣本跑完就是一個epoch。
![](https://i.imgur.com/ndPNwXN.png)
我們這邊先再詳細講解backpropagation裡的步驟，再回到fit。

Backpropagation裡，輸出層的error是最先算的，也就是算完最後一層，才往前一層算該層的error，回過來再說輸出層的error算法也是與其他層不同的![](https://i.imgur.com/73lkFAp.png)，這邊我們先算![](https://i.imgur.com/XTDEcBo.png)也就是程式碼中的F，也就是![](https://i.imgur.com/D3hagSQ.png)
F是一個對角矩陣，這邊我們可以利用到numpy中的diag函數，它塞1維矩陣進去會變成對角矩陣，也就是我們要的。
而裡面f’的意思是f’=f(1-f)，而f=a(n)，也就是我們之前feedforward計算的a，因此簡化成f’ = a(1-a)。
而它必須對a裡面的每一個質做計算，也就是會有10個f’。
![](https://i.imgur.com/ZNQJKVb.png)

![](https://i.imgur.com/EzjSNo8.png)
接著，經過one hot後的t減掉輸出層的a再乘上F，就成了我們要的輸出層的error。

隨即的其他層的error算法是將該層的下一層的error還有W相乘(記得轉置)後，再乘上該層算出的F即可得出，F如同前述。
![](https://i.imgur.com/omu01tU.png)
一次的樣本輸入經過feedforward算出a後，再經backpropagation算出error，接下來的步驟回到nn裡的fit。
我們將算出來的所有error，經過公式推出每一層delta_W和delta_b，並更新神經網路裡的W和b。
![](https://i.imgur.com/HbhBIJv.png)
![](https://i.imgur.com/PxeXNrJ.png)
將所有的樣本跑過epochs次數後，得出的W和b即是最適合該訓練資料的權重值和偏差值。

來自mit_neol:https://github.com/noelop/Multilayer-Perceptron/blob/master/main.py

Programming Assignment 1: Recognition of Handwritten Digits by MultilayerPerceptrons

這個專案創建了擁有簡易功能的Multilayer Perceptrons，當變數nn被指派為類別NeuralNetMLP時，此時nn即可以簡易的方式建構出MLP網路架構。
![](https://i.imgur.com/RV80S54.png)
nn有兩個參數可調整，epochs次數、learning rate數alpha。
隨即我們呼叫nn裡的add_layer方法，以建構神經網路。
 ![](https://i.imgur.com/oq4uD0E.png)
上圖中是以一層50X25的隱藏層配合sigmoid激活函數，前一層為輸入層，後一層為輸出層(輸出層通常使用softmax居多)。
這邊我們可使用add_layer這個方法，建構不同架構的神經網路。
![](https://i.imgur.com/YBO3JFB.png)
![](https://i.imgur.com/PGEQBqg.png)
設計此種方法能很便捷的變換不同層數或神經元數，以測驗最適合我們的網路架構。
建完網路架構後，我們呼叫fit方法，進入訓練階段。
![](https://i.imgur.com/FVoZ0zW.png)
X_train,y_train是我們的訓練資料，因input資料為8組手寫數字(x,y)座標數值組成，我們先做正規畫以避免sigmoid在過大或過小的區間效果不彰。
![](https://i.imgur.com/kGY4tLM.png)
接著我們進入NeuralNetMLP類別詳細說明fit方法是如何訓練的。

fit方法裡對所有樣本作了epochs變數值的次數訓練，每一樣本將會一一進入神經網路裡，所以將會是epoch*train_data_len次數的訓練。
一個樣本的第一步是進入feedforward裡，經過每一次的![](https://i.imgur.com/MngALK4.png)和緊接著的激活函數![](https://i.imgur.com/ANBZq82.png)，目的是算出每一層神經層的a，比如說 : 輸入層+3個隱藏層+輸出層有5層，那將會有5個a被計算出來。
![](https://i.imgur.com/tQbIWL9.png)

Feedforward裡每層的W會乘上前一層的a，而第一層的W則是乘上input的特徵值，到了最後一層時，a輸出會是10個數字，也就是和每一個數字互相對應著，最大的就是神經網路計算出最有可能的數字。
![](https://i.imgur.com/zWJb6PH.png)
計算完每一層的a之後，進入backpropagation的步驟，如同其名，是由最後一層往回算，其目的是算出每一層的delta_W, delta_b，再以減掉原本的W, b，作為更新每一層的權重質，每一次的樣本輸入都會做一次的feedforward和backpropagation，全部樣本跑完就是一個epoch。
![](https://i.imgur.com/ndPNwXN.png)
我們這邊先再詳細講解backpropagation裡的步驟，再回到fit。

Backpropagation裡，輸出層的error是最先算的，也就是算完最後一層，才往前一層算該層的error，回過來再說輸出層的error算法也是與其他層不同的![](https://i.imgur.com/73lkFAp.png)，這邊我們先算![](https://i.imgur.com/XTDEcBo.png)也就是程式碼中的F，也就是![](https://i.imgur.com/D3hagSQ.png)
F是一個對角矩陣，這邊我們可以利用到numpy中的diag函數，它塞1維矩陣進去會變成對角矩陣，也就是我們要的。
而裡面f’的意思是f’=f(1-f)，而f=a(n)，也就是我們之前feedforward計算的a，因此簡化成f’ = a(1-a)。
而它必須對a裡面的每一個質做計算，也就是會有10個f’。
![](https://i.imgur.com/ZNQJKVb.png)

![](https://i.imgur.com/EzjSNo8.png)
接著，經過one hot後的t減掉輸出層的a再乘上F，就成了我們要的輸出層的error。

隨即的其他層的error算法是將該層的下一層的error還有W相乘(記得轉置)後，再乘上該層算出的F即可得出，F如同前述。
![](https://i.imgur.com/omu01tU.png)
一次的樣本輸入經過feedforward算出a後，再經backpropagation算出error，接下來的步驟回到nn裡的fit。
我們將算出來的所有error，經過公式推出每一層delta_W和delta_b，並更新神經網路裡的W和b。
![](https://i.imgur.com/HbhBIJv.png)
![](https://i.imgur.com/PxeXNrJ.png)
將所有的樣本跑過epochs次數後，得出的W和b即是最適合該訓練資料的權重值和偏差值。


