verzió = "b6"

def get_dict():
    data = open("kiszedendok.txt", "r")
    for i in data:
        dictionary = i
    data.close()
    d = dictionary[1:]
    d = d[:-1]
    d = d.split(';')
    for i in range(len(d)):
        d[i] = d[i].split(":")
    d = dict(d)
    for i in range(1, 1122):
        d[str(i)] = list(d[str(i)])
        while ',' in d[str(i)]:
            d[str(i)].remove(',')
    return d

import shutil

def main():
    d = get_dict()
    for j in range(1, 1122):
        lista = d[str(j)]
        for i in lista:
            path = '../input/rendezett_festmenyek_' + verzió + '/train/nem-Bosch/' + str(j) + '_tr_' + i + ".jpeg"
            to = '../input/unused followers/' + verzió + '/' + str(j) + '_tr_' + i + ".jpeg"
            try:
                shutil.move(path, to)
            except FileNotFoundError as e:
                pass

import os
def delete():
    for j in ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6']:
        fileok = os.listdir('../input/rendezett_festmenyek_' + str(j) + '/train/nem-Bosch')
        for i in fileok:
            if i[-8:] == '(1).jpeg':
                print('van')
                os.remove('../input/rendezett_festmenyek_' + str(j) + '/train/nem-Bosch/' + i)

if __name__ == '__main__':
    delete()