import random
kiszedendok = {}
for i in range(1, 1122):
    lista = random.sample([1, 2, 3, 4, 5, 6, 7], 3)
    kiszedendok[i] = lista
    string = str(kiszedendok)
    string = string.replace('],', ';')
    string = string.replace('[', '')
    string = string.replace(" ", "")
if __name__ == '__main__':
    data = open("kiszedendok.txt", "w")
    data.write(string)
    data.close()
    