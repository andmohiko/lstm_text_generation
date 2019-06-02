##LSTMモデルを得るプログラム
# -*- coding: utf-8 -*-
from keras.models import Sequential,load_model
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop #学習のためのアルゴリズム
from keras.utils.data_utils import get_file #ファイルを扱うためのutilities
import numpy as np #行列を扱う
import random
import sys #ファイルのパス取得など


##辞書作成，sentence40文字,next_char1文字でtoVec準備
path = "./data_neko.txt"
bindata = open(path, "rb").read()
text = bindata.decode("utf-8")
print("Size of text: ",len(text))
#文字の分解，ソート
chars = sorted(list(set(text)))
print("Total chars :",len(chars))

#辞書作成！
#key が文字　値が番号　enumerate 番号付きのデータを作成
char_indices = dict((c,i) for i,c in enumerate(chars))
indices_char = dict((i,c) for i,c in enumerate(chars))

#40文字の次の文字に何が来るか(格納)学習させてベクトル化までの用意 3文字づつずらして40文字と1文字というセットを作る
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text)-maxlen, step):
    sentences.append(text[i:i+maxlen])
    next_chars.append(text[i+maxlen])

##テキストのベクトル化
#len(sentences)配列のサイズ　maxlen各要素のデータ数　len(chars)辞書
X = np.zeros((len(sentences),maxlen,len(chars)),dtype=np.bool)
y = np.zeros((len(sentences),len(chars)),dtype=np.bool)
#すべての40文字からなる文章(11022個存在)について1つずつ、11022回
for i, sentence in enumerate(sentences):
    #40文字からなる各文章の中、各文字について1つずつ、40回
    for t ,char in enumerate(sentence):
        #ある文章iの中の ある文字tについて　文字から番号を引いたところに1
        X[i,t,char_indices[char]] = 1 #入力を格納
    #次のデータを格納
    y[i,char_indices[next_chars[i]]] = 1

##モデルを定義する
model = Sequential() #連続的なデータを扱う
#ニューラルネットのレイヤを追加 LSTMを使用　サイズ128
model.add(LSTM(128, input_shape=(maxlen,len(chars))))
#Dense 全結合、すべてのセルを使ったニューラルネットをつくる サイズはlen(chars)
model.add(Dense(len(chars)))
#文字の推定、Activation活性化関数　softmax出力した値を0~1に収まるよう変換
model.add(Activation("softmax"))
#RMSpropというアルゴリズム　学習率0.01
optimizer = RMSprop(lr = 0.01)
#トレーニング宣言　loss損失関数、どれくらい離れているか
model.compile(loss="categorical_crossentropy",optimizer=optimizer)

#トレーニング後値を取り出す関数
def sample(preds, temperature=1.0):
    #prediction予想値　変数の型float64
    preds = np.asarray(preds).astype("float64") #初期化
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1) #probability　確率
    #argmax配列の中で最大値の位置を返すを返す
    return np.argmax(probs)

#文字ベースのベクトル化
#それをもとにある文章の後にどんな文字来るか
#新しいSeedを与えて次に来る文字を予測

for iteration in range(1,40):
    print()
    print("-"*50)
    print("繰り返し回数: ",iteration)
    #学習をする　1回に投入するデータの長さ　全体のデータを何回回すか
    model.fit(X, y, batch_size=128, epochs=1)
    #最初に開始するテキストをランダムに、整数発行 0から最後までの間で
    start_index = random.randint(0, len(text)-maxlen-1)

#サンプルデータを出力するためのパラメータ
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print("-----diversity", diversity)
    #生成する文章を入れる
        generated =""
    #最初の文章　ランダムなとこから一回の文書の長さ（４０）
        sentence = text[start_index: start_index + maxlen ]
        generated += sentence
        print("-----Seedを生成しました: " + sentence + '"')
        sys.stdout.write(generated)

        #次の文字を予測して足していく
        for i in range(400):
            x = np.zeros((1,maxlen,len(chars)))
            for t,char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1

            preds = model.predict(x, verbose =9)[0] #次の文字を予測
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
model.save('neko_model.h5')
file = open('gentext_neko.txt','w+',encoding='utf-8').write(generated)
