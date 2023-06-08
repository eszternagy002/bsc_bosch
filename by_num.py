import os
root = '../kepek'
verzió = 'b7'
def getting_the_titles():
    boschprojekt = os.listdir(root + '/boschprojekt')
    JBAC = os.listdir(root + '/JBAC')
    trans_ones_b = os.listdir(root + '/transformed ones/bosch_' + verzió)
    trans_ones_f = os.listdir(root + '/transformed ones/followers_' + verzió)
    detail_pics = os.listdir(root + '/boschprojekt/detail images')
    uncertain_JBAC = os.listdir(root + '/JBAC/artist uncertain, circle, school of Bosch')
    uncertain_bosch = os.listdir(root + '/boschprojekt/artist uncertain')
    workshop_of_bosch = os.listdir(root + '/boschprojekt/workshop of bosch')

    boschprojekt_szurt = []
    JBAC_szurt = []
    trans_ones_b_szurt = []
    trans_ones_f_szurt = []
    detail_pics_szurt = []
    uncertain_JBAC_szurt = []
    uncertain_bosch_szurt = []
    workshop_of_bosch_szurt = []
    
    for i in boschprojekt:
        if i[-4:] == "jpeg":
            boschprojekt_szurt.append(i)

    for i in JBAC:
        if i[-4:] == "jpeg":
            JBAC_szurt.append(i)

    for i in trans_ones_b:
        if i[-4:] == "jpeg":
            trans_ones_b_szurt.append(i)
            
    for i in trans_ones_f:
        if i[-4:] == "jpeg":
            trans_ones_f_szurt.append(i)
        
    for i in detail_pics:
        if i[-4:] == "jpeg":
            detail_pics_szurt.append(i)
            
    for i in uncertain_JBAC:
        if i[-4:] == "jpeg":
            uncertain_JBAC_szurt.append(i)
            
    for i in uncertain_bosch:
        if i[-4:] == "jpeg":
            uncertain_bosch_szurt.append(i)
            
    for i in workshop_of_bosch:
        if i[-4:] == "jpeg":
            workshop_of_bosch_szurt.append(i)
    
    return [boschprojekt_szurt, JBAC_szurt, trans_ones_b_szurt, trans_ones_f_szurt, uncertain_bosch_szurt, workshop_of_bosch_szurt, uncertain_JBAC_szurt, detail_pics_szurt]

from sklearn.model_selection import train_test_split
def get_the_paintings(list1, list2, test_size=0.2, val=False, val_size=0.1, egybe=True):
    """list1 és list2 azok a listák, amik a festmények neveit tartalmazzák
    test_size megadja, mekkora legyen a teszthalmaz aránya
    a val egy bool típusú változó, ha értéke True, csinál validációs halmazt, különben nem
    val_size mekkora legyen a validációs halmaz aránya
    egybe bool típusú, megadja, hogy a két festményből álló alkotásokat csak egybe vigye-e
    csak az első listára csinálja meg, oda kell a bosch_pics listát tenni!"""
    ordered_list1 = []
    if egybe:
        for i in list1:
            if i[-7:-5]=='_2':
                ordered_list1.append([i[:-7] + "_1.jpeg",i])
            else:
                ordered_list1.append(i)
                #így még az egyesek benne vannak!
        for i in ordered_list1:
            for j in ordered_list1:
                if i[0] == j:
                    ordered_list1.remove(j)
        list1=ordered_list1
    train_set_B, test_set_B = train_test_split(list1, test_size=test_size, random_state=8)
    train_set_f, test_set_f = train_test_split(list2, test_size=test_size, random_state=8)
    if val:
        train_set_B, val_set_B = train_test_split(train_set_B, test_size=val_size/(1-test_size), random_state=8)
        train_set_f, val_set_f = train_test_split(train_set_f, test_size=val_size/(1-test_size), random_state=8)
    if val:
        return [train_set_B, train_set_f, test_set_B, test_set_f, val_set_B, val_set_f]
    else:
        return [train_set_B, train_set_f, test_set_B, test_set_f]
    #megkaptuk a nem dúsított képeket felbontva, most a train és val halmazhoz meg kell keresni a hozzájuk tartozó dúsítottakat


def get_trans_and_details(train_set_B, train_set_f, transformed_pics_B, transformed_pics_f, detail_pics):
    #úgy veszem, hogy az előző függvényben az egybe True volt!
    train_B_trans_and_det = []
    train_f_trans = []
    
    #a transzformált Bosch-okból kiszedjük a tanítóhalmazba kellőket
    for trans in transformed_pics_B:
            eleje_trans = trans[0:2]
            if trans[2] != '_':
                eleje_trans = eleje_trans + trans[2]
            for i in train_set_B:
                if type(i) == list:
                    eleje = i[0][:-7]
                else:
                    eleje = i[:-7]
                if eleje==eleje_trans:
                    train_B_trans_and_det.append(trans)
        
    #a detail pic-ekből is
    for det in detail_pics:
            eleje_det = det[0:2]
            if trans[2] != '_':
                eleje_det = eleje_det + det[2]
            for i in train_set_B:
                if type(i) == list:
                    eleje = i[0][:-7]
                else:
                    eleje = i[:-7]
                if eleje==eleje_det:
                    train_B_trans_and_det.append(det)
            
    #a followerek transzformált képeit is megkeressük
    #nem valami szép, sok az ismétlés, de elsőre jó lesz
    for trans in transformed_pics_f:
        h = 0
        eleje_trans = ''
        while trans[h] != '_':
            eleje_trans = eleje_trans + trans[h]
            h = h + 1
        for i in train_set_f:
            if type(i) == list:
                eleje = i[0][:-5]
            else:
                eleje = i[:-5]
            if eleje==eleje_trans:
                train_f_trans.append(trans)
    return [train_B_trans_and_det, train_f_trans]

def szetszedo(lista):
    """visszaalakítja a listát is tartalmazó listát sima listává"""
    szetszedett = []
    for i in lista:
        if type(i) == list:
            szetszedett.append(i[0])
            szetszedett.append(i[1])
        else:
            szetszedett.append(i)
    return szetszedett


def listak():
    painting_lists = getting_the_titles()
    paint = get_the_paintings(painting_lists[0], painting_lists[1], val=True) #így már lesz plusz két elem a lista végén, a val halmazok
    trans = get_trans_and_details(paint[0], paint[1], painting_lists[2], painting_lists[3], painting_lists[7])
    cimkek =  [szetszedo(paint[0]), paint[1], paint[2], paint[3], trans[0], trans[1], painting_lists[5], painting_lists[4], painting_lists[6], paint[4], paint[5]]
    #cimek ebben a sorrendben:
    #(0)train set Bosch része, (1)train set follower része, (2)test set Bosch része, (3)test set follower része, (4)részletek train Bosch-hoz, (5)részletek train followerhez
    #(6)workshop_képek, (7)uncertain Bosch-képek, (8) uncertain follower-képek, (9) Bosch validációs, (10) follower validációs
    #már egyik listában sincsen lista!
    #adjuk hozzá az elérési útvonalakat!
    for i in range(len(cimkek[0])):
        cimkek[0][i] = root + '/boschprojekt/' + cimkek[0][i] #train set Bosch
    for i in range(len(cimkek[1])):
        cimkek[1][i] = root + '/JBAC/' + cimkek[1][i] #train set follower
        
    for i in range(len(cimkek[2])):
        cimkek[2][i] = root + '/boschprojekt/' + cimkek[2][i] #test set Bosch
    for i in range(len(cimkek[3])):
        cimkek[3][i] = root + '/JBAC/' + cimkek[3][i] #test set follower
        
    for i in range(len(cimkek[4])): #augmented and details Bosch
        if cimkek[4][i][0:4] == "det":
            cimkek[4][i] = root + '/boschprojekt/detail images/' + cimkek[4][i]
        else:
            cimkek[4][i] = root + '/transformed ones/bosch_' + verzió + '/' + cimkek[4][i]
    for i in range(len(cimkek[5])):
        cimkek[5][i] = root + '/transformed ones/followers_' + verzió + '/' + cimkek[5][i] #train follower augmented
        
    for i in range(len(cimkek[9])):
        cimkek[9][i] = root + '/boschprojekt/' + cimkek[9][i] #validációs Bosch
    for i in range(len(cimkek[10])):
        cimkek[10][i] = root + '/JBAC/' + cimkek[10][i] #validációs follower
        
    for i in range(len(cimkek[6])):
        cimkek[6][i] = root + '/boschprojekt/workshop of bosch/' + cimkek[6][i] #workshop képek
    for i in range(len(cimkek[7])):
        cimkek[7][i] = root + '/boschprojekt/artist uncertain/' + cimkek[7][i] #uncertain Bosch
    for i in range(len(cimkek[8])):
        cimkek[8][i] = root + '/JBAC/artist uncertain, circle, school of Bosch/' + cimkek[8][i] #uncertain follower
    
    return cimkek

import shutil
import stat

def remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def train_val_test_and_copy():
    lists = listak()

    cel = os.path.join('input' + os.sep, 'rendezett_festmenyek_' + verzió)
    shutil.rmtree(cel, onerror=remove_readonly)

    os.makedirs(cel + '/train/Bosch')
    os.makedirs(cel + '/train/nem-Bosch')
    os.makedirs(cel + '/val/Bosch')
    os.makedirs(cel + '/val/nem-Bosch')
    os.makedirs(cel + '/test/Bosch')
    os.makedirs(cel + '/test/nem-Bosch')
    os.makedirs(cel + '/test/other')

    #mappaszerkezet
    #train: Bosch, nem-Bosch
    #val: Bosch, nem-Bosch
    #test: Bosch, nem-Bosch, other
    train = []
    val = []
    test = []
    train.append(lists[0]) #train mappához adja a Bosch train részét
    train.append(lists[4])
    train.append(lists[1]) #a follower részét
    train.append(lists[5])
    #train szerkezet: Bosch train, Bosch részlet train, follower train, follower részlet train
    val.append(lists[9])
    val.append(lists[10])
    #val szerkezet: Bosch validációs, follower validációs
    test.append(lists[2])
    test.append(lists[3])
    test.append(lists[6])
    test.append(lists[7])
    test.append(lists[8])
    #test szerkezet: Bosch test, follower test, uncertain képek 3 listában

    #copying the files
    #train, Bosch
    for i in train[0]:
        shutil.copy2(i, cel + '/train/Bosch')
    for i in train[1]:
        shutil.copy2(i, cel + '/train/Bosch')
    #train, follower
    for i in train[2]:
        shutil.copy2(i, cel + '/train/nem-Bosch')
    for i in train[3]:
        shutil.copy2(i, cel + '/train/nem-Bosch')

    #val, Bosch
    for i in val[0]:
        shutil.copy2(i, cel + '/val/Bosch')
    #val, follower
    for i in val[1]:
        shutil.copy2(i, cel + '/val/nem-Bosch')

    #test, Bosch
    for i in test[0]:
        shutil.copy2(i, cel + '/test/Bosch')
    #test follower
    for i in test[1]:
        shutil.copy2(i, cel + '/test/nem-Bosch')
    #test other
    for i in test[2]:
        shutil.copy2(i, cel + '/test/other')
    for i in test[3]:
        shutil.copy2(i, cel + '/test/other')
    for i in test[4]:
        shutil.copy2(i, cel + '/test/other')

if __name__ == '__main__':
    train_val_test_and_copy()


    