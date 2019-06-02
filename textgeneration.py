## 学習済みのモデルから文章を自動生成するプログラム
# -*- coding: utf-8 -*-
from keras.models import Sequential,load_model
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop #学習のためのアルゴリズム
from keras.utils.data_utils import get_file #ファイルを扱うためのutilities
import numpy as np #行列を扱う
import random
import sys #ファイルのパス取得など

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

def sample(preds, temperature=1.0):
    #prediction予想値　変数の型float64
    preds = np.asarray(preds).astype("float64") #初期化
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1) #probability　確率
    #argmax配列の中で最大値の位置を返すを返す
    return np.argmax(probs)

model=load_model('neko_model.h5')

start_index = random.randint(0, len(text)-maxlen-1)
for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print("-----diversity", diversity)
    #生成する文章を入れる
        generated =""
    #最初の文章　ランダムなとこから一回の文書の長さ（４０）
        sentence = text[start_index: start_index + maxlen ]
        generated += sentence
        print("-----Seedを生成しました: " +'\n"'+ sentence + '"')
        print()
        sys.stdout.write(generated)

        #次の文字を予測して足していく
        for i in range(400):
            x = np.zeros((1,maxlen,len(chars)))
            for t,char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1

            preds = model.predict(x, verbose =0)[0] #次の文字を予測
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            #if next_char=='。':
            #    a='\n'
            #elif next_char=='\n':
            #    a=''
            #    next_char=''
            #else:
            #    a=''


            generated += next_char#+a
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()#生成と同時に出力できるようにするための処理
        print()
file = open('gentext_neko.txt','w+',encoding='utf-8').write(generated)     
