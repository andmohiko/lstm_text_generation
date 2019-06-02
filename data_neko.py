# -*- coding: utf-8 -*-
import sys
import re

path = './wagahaiwa_nekodearu.txt'
bindata = open(path, "rb")
lines = bindata.readlines()
for line in lines:
    text = line.decode('Shift_JIS')
    text = re.split(r'\r',text)[0]
    text = re.split(r'底本',text)[0]
    text = text.replace('｜','')
    text = re.sub(r'《.+?》','',text)
    text = re.sub(r'［＃.+?］','',text)
    print(text)
    file = open('data_neko.txt','a',encoding='utf-8').write(text)

    # match = re.search(r'《(.+)》',line)
    # if match:
    #     line.replace('《(.+)》', null)
    #     serif = match.group(1)+'\n'
    #     sys.stdout.write(serif)
    #     file = open('data_neko.txt','a',encoding='utf-8').write(serif)
