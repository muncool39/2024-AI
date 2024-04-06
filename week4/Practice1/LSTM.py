import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def encode_sentence(data):
    sentences = [s[0].split() for s in data]  
    labels = [s[1] for s in data]  
    max_len = max(len(s) for s in sentences)
    for i in range(len(sentences)):
        while len(sentences[i]) < max_len:
            sentences[i].append("<PAD>") 
    vocab = set(word for sentence in sentences for word in sentence)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    indexed_sentences = [[word_to_idx[word] for word in sentence] for sentence in sentences]
    return indexed_sentences, labels, vocab

data = [
    ("This is great", 1),
    ("I love it", 1),
    ("So good", 1),
    ("Amazing work", 1),
    ("Fantastic job", 1),
    ("Not good", 0),
    ("I am disappointed", 0),
    ("Bad experience", 0),
    ("Not happy with this", 0),
    ("Could be better", 0),
    ("Amazing work", 1),
    ("Fantastic job", 1),
    ("Not happy with this", 0),
    ("Could be better", 0),
    ("I am very good", 1),
    ("An excellent experience", 1),
    ("I am not happy", 0),
    ("It was a bad day", 0),
]


class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SimpleLSTM, self).__init__()
        # LSTM : 시계열 정보를 순차적으로 줘야 한다
        # embedding 을 통해 데이터 크기 / 몇으로 줄일건지 정보 할당
        # 임베디드된 값만 LSTM 에 전달
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # 분류 문제로 바꿔주기 위해 Linear 레이블 필요
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, _) = self.lstm(embedded)
        hidden = hidden.squeeze(0)
        out = self.fc(hidden)
        return out

# 학습/테스트 데이터 생성
indexed_sentences, labels, vocab = encode_sentence(data)
inputs = torch.tensor(indexed_sentences, dtype=torch.long)
targets = torch.tensor(labels, dtype=torch.float32)

inputs_train, inputs_test = inputs[:10], inputs[10:]
targets_train, targets_test = targets[:10], targets[10:]

# 모델 초기화
# 모델 정의
model = SimpleLSTM(len(vocab), embedding_dim=8, hidden_dim=4, output_dim=1)
loss_fn = nn.BCEWithLogitsLoss()
# 옵티마이저 정의
optimizer = torch.optim.Adam(model.parameters(), lr=0.5)

# 학습 루프
for epoch in range(3000):
    model.train()
    optimizer.zero_grad()
    outputs = model(inputs_train)
    loss = loss_fn(outputs.squeeze(), targets_train)
    loss.backward()
    optimizer.step()    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

model.eval()  
with torch.no_grad(): 
    predictions = model(inputs_test)
    predictions = torch.sigmoid(predictions).squeeze()
    predicted_labels = predictions.round() 
    
    accuracy = (predicted_labels == targets_test.squeeze()).float().mean() 
    print(f'Test Accuracy: {accuracy.item()}')
    
    print("Test Predictions:")
    for i, (sentence, _) in enumerate(data[10:]):        
        print(f"{sentence} - Predicted: {'Positive' if predicted_labels[i] == 1 else 'Negative'}")

"""실행결과
Epoch 0, Loss: 0.6989835500717163
Epoch 10, Loss: 0.1082368716597557
Epoch 20, Loss: 0.03269238397479057
Epoch 30, Loss: 0.0108425198122859
Epoch 40, Loss: 0.005045468453317881
Epoch 50, Loss: 0.0009436032851226628
Epoch 60, Loss: 0.0008151374058797956
Epoch 70, Loss: 0.0006854759412817657
Epoch 80, Loss: 0.0005775552708655596
Epoch 90, Loss: 0.0004929297720082104
Epoch 100, Loss: 0.00042674707947298884
Epoch 110, Loss: 0.00037426737253554165
Epoch 120, Loss: 0.0003319725510664284
Epoch 130, Loss: 0.0002972822985611856
Epoch 140, Loss: 0.00026832910953089595
Epoch 150, Loss: 0.00024389968893956393
Epoch 160, Loss: 0.00022305396851152182
Epoch 170, Loss: 0.0002050301991403103
Epoch 180, Loss: 0.00018932856619358063
Epoch 190, Loss: 0.00017555621161591262
Epoch 200, Loss: 0.00016336787666659802
Epoch 210, Loss: 0.0001525493134977296
Epoch 220, Loss: 0.00014287422527559102
Epoch 230, Loss: 0.0001341640017926693
Epoch 240, Loss: 0.00012631148274522275
Epoch 250, Loss: 0.00011920950055355206
Epoch 260, Loss: 0.00011272700794506818
Epoch 270, Loss: 0.0001068044439307414
Epoch 280, Loss: 0.00010139421647181734
Epoch 290, Loss: 9.640098141971976e-05
Epoch 300, Loss: 9.183670044876635e-05
Epoch 310, Loss: 8.75821933732368e-05
Epoch 320, Loss: 8.366132533410564e-05
Epoch 330, Loss: 8.001451351447031e-05
Epoch 340, Loss: 7.661792915314436e-05
Epoch 350, Loss: 7.344775076489896e-05
Epoch 360, Loss: 7.04920748830773e-05
Epoch 370, Loss: 6.771513290004805e-05
Epoch 380, Loss: 6.512885738629848e-05
Epoch 390, Loss: 6.269750883802772e-05
Epoch 400, Loss: 6.038532592356205e-05
Epoch 410, Loss: 5.823998435516842e-05
Epoch 420, Loss: 5.6213815696537495e-05
Epoch 430, Loss: 5.428298027254641e-05
Epoch 440, Loss: 5.248324669082649e-05
Epoch 450, Loss: 5.07788427057676e-05
Epoch 460, Loss: 4.912211443297565e-05
Epoch 470, Loss: 4.760840238304809e-05
Epoch 480, Loss: 4.6118519094306976e-05
Epoch 490, Loss: 4.4747819629264995e-05
Epoch 500, Loss: 4.341287421993911e-05
Epoch 510, Loss: 4.214944056002423e-05
Epoch 520, Loss: 4.0933671698439866e-05
Epoch 530, Loss: 3.978941822424531e-05
Epoch 540, Loss: 3.869283682433888e-05
Epoch 550, Loss: 3.7643927498720586e-05
Epoch 560, Loss: 3.661885784822516e-05
Epoch 570, Loss: 3.566529630916193e-05
Epoch 580, Loss: 3.4747492463793606e-05
Epoch 590, Loss: 3.385352829354815e-05
Epoch 600, Loss: 3.3007239835569635e-05
Epoch 610, Loss: 3.219670179532841e-05
Epoch 620, Loss: 3.139808904961683e-05
Epoch 630, Loss: 3.063522308366373e-05
Epoch 640, Loss: 2.9931961762486026e-05
Epoch 650, Loss: 2.9228696803329512e-05
Epoch 660, Loss: 2.854927151929587e-05
Epoch 670, Loss: 2.7917520128539763e-05
Epoch 680, Loss: 2.7297690394334495e-05
Epoch 690, Loss: 2.6665937184588984e-05
Epoch 700, Loss: 2.6105703000212088e-05
Epoch 710, Loss: 2.5545468815835193e-05
Epoch 720, Loss: 2.4985234631458297e-05
Epoch 730, Loss: 2.4472672521369532e-05
Epoch 740, Loss: 2.398395554337185e-05
Epoch 750, Loss: 2.347139707126189e-05
Epoch 760, Loss: 2.3006516130408272e-05
Epoch 770, Loss: 2.2553558665094897e-05
Epoch 780, Loss: 2.2088675905251876e-05
Epoch 790, Loss: 2.1671472495654598e-05
Epoch 800, Loss: 2.125426908605732e-05
Epoch 810, Loss: 2.0848987333010882e-05
Epoch 820, Loss: 2.0479465092648752e-05
Epoch 830, Loss: 2.0074179701623507e-05
Epoch 840, Loss: 1.9692735804710537e-05
Epoch 850, Loss: 1.9347051420481876e-05
Epoch 860, Loss: 1.8989445379702374e-05
Epoch 870, Loss: 1.8655682652024552e-05
Epoch 880, Loss: 1.832191446737852e-05
Epoch 890, Loss: 1.8000067939283326e-05
Epoch 900, Loss: 1.7690144886728376e-05
Epoch 910, Loss: 1.739213985274546e-05
Epoch 920, Loss: 1.7094132999773137e-05
Epoch 930, Loss: 1.6796126146800816e-05
Epoch 940, Loss: 1.6521958968951367e-05
Epoch 950, Loss: 1.6212030459428206e-05
Epoch 960, Loss: 1.5949786757119e-05
Epoch 970, Loss: 1.5687539416830987e-05
Epoch 980, Loss: 1.542529025755357e-05
Epoch 990, Loss: 1.518688350188313e-05
Epoch 1000, Loss: 1.4960397493268829e-05
Epoch 1010, Loss: 1.4721992556587793e-05
Epoch 1020, Loss: 1.4483584891422652e-05
Epoch 1030, Loss: 1.4245178135752212e-05
Epoch 1040, Loss: 1.4030610145709943e-05
Epoch 1050, Loss: 1.3827964721713215e-05
Epoch 1060, Loss: 1.3637238225783221e-05
Epoch 1070, Loss: 1.3422672054730356e-05
Epoch 1080, Loss: 1.3231945558800362e-05
Epoch 1090, Loss: 1.3029299225308932e-05
Epoch 1100, Loss: 1.2838572729378939e-05
Epoch 1110, Loss: 1.2671686818066519e-05
Epoch 1120, Loss: 1.2492880159697961e-05
Epoch 1130, Loss: 1.2302152754273266e-05
Epoch 1140, Loss: 1.2135265933466144e-05
Epoch 1150, Loss: 1.1956459275097586e-05
Epoch 1160, Loss: 1.18014922918519e-05
Epoch 1170, Loss: 1.1634607290034182e-05
Epoch 1180, Loss: 1.1491561963339336e-05
Epoch 1190, Loss: 1.1300833648419939e-05
Epoch 1200, Loss: 1.1145866665174253e-05
Epoch 1210, Loss: 1.1002821338479407e-05
Epoch 1220, Loss: 1.0859776011784561e-05
Epoch 1230, Loss: 1.0716730685089715e-05
Epoch 1240, Loss: 1.0585604286461603e-05
Epoch 1250, Loss: 1.046639954438433e-05
Epoch 1260, Loss: 1.0311432561138645e-05
Epoch 1270, Loss: 1.0192227819061372e-05
Epoch 1280, Loss: 1.0073022167489398e-05
Epoch 1290, Loss: 9.91805427474901e-06
Epoch 1300, Loss: 9.798849532671738e-06
Epoch 1310, Loss: 9.679642971605062e-06
Epoch 1320, Loss: 9.560439139022492e-06
Epoch 1330, Loss: 9.465074072068091e-06
Epoch 1340, Loss: 9.345869329990819e-06
Epoch 1350, Loss: 9.20282218430657e-06
Epoch 1360, Loss: 9.083615623239893e-06
Epoch 1370, Loss: 9.01209295989247e-06
Epoch 1380, Loss: 8.892886398825794e-06
Epoch 1390, Loss: 8.77368074725382e-06
Epoch 1400, Loss: 8.67831568029942e-06
Epoch 1410, Loss: 8.571030775783584e-06
Epoch 1420, Loss: 8.475666618323885e-06
Epoch 1430, Loss: 8.380300641874783e-06
Epoch 1440, Loss: 8.30877797852736e-06
Epoch 1450, Loss: 8.189572326955386e-06
Epoch 1460, Loss: 8.094207260000985e-06
Epoch 1470, Loss: 8.02268368715886e-06
Epoch 1480, Loss: 7.92731862020446e-06
Epoch 1490, Loss: 7.843874300306197e-06
Epoch 1500, Loss: 7.748509233351797e-06
Epoch 1510, Loss: 7.676985660509672e-06
Epoch 1520, Loss: 7.581620593555272e-06
Epoch 1530, Loss: 7.510096565965796e-06
Epoch 1540, Loss: 7.438572993123671e-06
Epoch 1550, Loss: 7.343207926169271e-06
Epoch 1560, Loss: 7.259762696776306e-06
Epoch 1570, Loss: 7.176317922130693e-06
Epoch 1580, Loss: 7.104794349288568e-06
Epoch 1590, Loss: 7.033270321699092e-06
Epoch 1600, Loss: 6.9617462941096164e-06
Epoch 1610, Loss: 6.878301974211354e-06
Epoch 1620, Loss: 6.806777491874527e-06
Epoch 1630, Loss: 6.7352534642850515e-06
Epoch 1640, Loss: 6.687571385555202e-06
Epoch 1650, Loss: 6.5922054091061e-06
Epoch 1660, Loss: 6.5445228756289e-06
Epoch 1670, Loss: 6.484920049842913e-06
Epoch 1680, Loss: 6.413395567506086e-06
Epoch 1690, Loss: 6.34187153991661e-06
Epoch 1700, Loss: 6.270347512327135e-06
Epoch 1710, Loss: 6.2226645241025835e-06
Epoch 1720, Loss: 6.151140496513108e-06
Epoch 1730, Loss: 6.103457963035908e-06
Epoch 1740, Loss: 6.0557749748113565e-06
Epoch 1750, Loss: 5.984250947221881e-06
Epoch 1760, Loss: 5.93656795899733e-06
Epoch 1770, Loss: 5.865043931407854e-06
Epoch 1780, Loss: 5.817361852678005e-06
Epoch 1790, Loss: 5.757757662649965e-06
Epoch 1800, Loss: 5.686232725565787e-06
Epoch 1810, Loss: 5.6623925956955645e-06
Epoch 1820, Loss: 5.614708697976312e-06
Epoch 1830, Loss: 5.543184670386836e-06
Epoch 1840, Loss: 5.495502591656987e-06
Epoch 1850, Loss: 5.47166064279736e-06
Epoch 1860, Loss: 5.4001366152078845e-06
Epoch 1870, Loss: 5.352453626983333e-06
Epoch 1880, Loss: 5.328612132871058e-06
Epoch 1890, Loss: 5.257087650534231e-06
Epoch 1900, Loss: 5.20940466230968e-06
Epoch 1910, Loss: 5.185563168197405e-06
Epoch 1920, Loss: 5.114038685860578e-06
Epoch 1930, Loss: 5.0901971917483024e-06
Epoch 1940, Loss: 5.030593456467614e-06
Epoch 1950, Loss: 5.006752417102689e-06
Epoch 1960, Loss: 4.935228389513213e-06
Epoch 1970, Loss: 4.911386440653587e-06
Epoch 1980, Loss: 4.863703452429036e-06
Epoch 1990, Loss: 4.8160209189518355e-06
Epoch 2000, Loss: 4.756416728923796e-06
Epoch 2010, Loss: 4.73257523481152e-06
Epoch 2020, Loss: 4.708733740699245e-06
Epoch 2030, Loss: 4.637209258362418e-06
Epoch 2040, Loss: 4.6133677642501425e-06
Epoch 2050, Loss: 4.589526270137867e-06
Epoch 2060, Loss: 4.541843281913316e-06
Epoch 2070, Loss: 4.494159838941414e-06
Epoch 2080, Loss: 4.470318344829138e-06
Epoch 2090, Loss: 4.422635811351938e-06
Epoch 2100, Loss: 4.398794317239663e-06
Epoch 2110, Loss: 4.351110874267761e-06
Epoch 2120, Loss: 4.327269380155485e-06
Epoch 2130, Loss: 4.291507138987072e-06
Epoch 2140, Loss: 4.243824150762521e-06
Epoch 2150, Loss: 4.219982656650245e-06
Epoch 2160, Loss: 4.172299213678343e-06
Epoch 2170, Loss: 4.148457719566068e-06
Epoch 2180, Loss: 4.124616225453792e-06
Epoch 2190, Loss: 4.100774731341517e-06
Epoch 2200, Loss: 4.053091288369615e-06
Epoch 2210, Loss: 4.0054083001450635e-06
Epoch 2220, Loss: 3.981567260780139e-06
Epoch 2230, Loss: 3.9577253119205125e-06
Epoch 2240, Loss: 3.933884272555588e-06
Epoch 2250, Loss: 3.886200829583686e-06
Epoch 2260, Loss: 3.86235933547141e-06
Epoch 2270, Loss: 3.838517841359135e-06
Epoch 2280, Loss: 3.8146761198731838e-06
Epoch 2290, Loss: 3.7908346257609082e-06
Epoch 2300, Loss: 3.7431514101626817e-06
Epoch 2310, Loss: 3.719309916050406e-06
Epoch 2320, Loss: 3.6954679671907797e-06
Epoch 2330, Loss: 3.671626927825855e-06
Epoch 2340, Loss: 3.6358640045364155e-06
Epoch 2350, Loss: 3.612022965171491e-06
Epoch 2360, Loss: 3.5881810163118644e-06
Epoch 2370, Loss: 3.5643392948259134e-06
Epoch 2380, Loss: 3.5285768262838246e-06
Epoch 2390, Loss: 3.504735332171549e-06
Epoch 2400, Loss: 3.4689728636294603e-06
Epoch 2410, Loss: 3.445131369517185e-06
Epoch 2420, Loss: 3.397447926545283e-06
Epoch 2430, Loss: 3.3736064324330073e-06
Epoch 2440, Loss: 3.3736064324330073e-06
Epoch 2450, Loss: 3.3497647109470563e-06
Epoch 2460, Loss: 3.3259232168347808e-06
Epoch 2470, Loss: 3.3020817227225052e-06
Epoch 2480, Loss: 3.2782402286102297e-06
Epoch 2490, Loss: 3.2543982797506033e-06
Epoch 2500, Loss: 3.2305567856383277e-06
Epoch 2510, Loss: 3.2067150641523767e-06
Epoch 2520, Loss: 3.182873570040101e-06
Epoch 2530, Loss: 3.1590320759278256e-06
Epoch 2540, Loss: 3.1590320759278256e-06
Epoch 2550, Loss: 3.1113486329559237e-06
Epoch 2560, Loss: 3.087507593590999e-06
Epoch 2570, Loss: 3.0636656447313726e-06
Epoch 2580, Loss: 3.0636656447313726e-06
Epoch 2590, Loss: 3.0398239232454216e-06
Epoch 2600, Loss: 3.015982429133146e-06
Epoch 2610, Loss: 3.015982429133146e-06
Epoch 2620, Loss: 2.992140707647195e-06
Epoch 2630, Loss: 2.944457264675293e-06
Epoch 2640, Loss: 2.9325365176191553e-06
Epoch 2650, Loss: 2.9086947961332044e-06
Epoch 2660, Loss: 2.884853302020929e-06
Epoch 2670, Loss: 2.884853302020929e-06
Epoch 2680, Loss: 2.861011580534978e-06
Epoch 2690, Loss: 2.8371700864227023e-06
Epoch 2700, Loss: 2.8371700864227023e-06
Epoch 2710, Loss: 2.7894866434508003e-06
Epoch 2720, Loss: 2.7656449219648493e-06
Epoch 2730, Loss: 2.7656449219648493e-06
Epoch 2740, Loss: 2.7418032004788984e-06
Epoch 2750, Loss: 2.706040959310485e-06
Epoch 2760, Loss: 2.706040959310485e-06
Epoch 2770, Loss: 2.682199237824534e-06
Epoch 2780, Loss: 2.682199237824534e-06
Epoch 2790, Loss: 2.658357516338583e-06
Epoch 2800, Loss: 2.6106738459930057e-06
Epoch 2810, Loss: 2.6106738459930057e-06
Epoch 2820, Loss: 2.58683235188073e-06
Epoch 2830, Loss: 2.58683235188073e-06
Epoch 2840, Loss: 2.5629908577684546e-06
Epoch 2850, Loss: 2.5629908577684546e-06
Epoch 2860, Loss: 2.539148908908828e-06
Epoch 2870, Loss: 2.539148908908828e-06
Epoch 2880, Loss: 2.5153074147965526e-06
Epoch 2890, Loss: 2.4914656933106016e-06
Epoch 2900, Loss: 2.467623744450975e-06
Epoch 2910, Loss: 2.4437822503386997e-06
Epoch 2920, Loss: 2.431861503282562e-06
Epoch 2930, Loss: 2.408019781796611e-06
Epoch 2940, Loss: 2.408019781796611e-06
Epoch 2950, Loss: 2.38417806031066e-06
Epoch 2960, Loss: 2.38417806031066e-06
Epoch 2970, Loss: 2.3603365661983844e-06
Epoch 2980, Loss: 2.3603365661983844e-06
Epoch 2990, Loss: 2.3364948447124334e-06
Test Accuracy: 0.625
Test Predictions:
Amazing work - Predicted: Positive
Fantastic job - Predicted: Positive
Not happy with this - Predicted: Negative
Could be better - Predicted: Negative
I am very good - Predicted: Negative
An excellent experience - Predicted: Negative
I am not happy - Predicted: Positive
It was a bad day - Predicted: Negative
"""