import random


def deal_data():
    with open('dataset/all.txt', 'r', encoding="utf-8") as f, \
            open('dataset/positive_example.txt', 'w', encoding="utf-8") as new:
        read_1 = f.readlines()
        list_1 = []
        for a in [a.strip('\n') for a in read_1]:
            one_li = a
            if a != None:
                list_1.append(one_li)
                for b in list_1:
                    for c in list_1:
                        new.write("1" + "\t" + b + "|||" + c)
                        new.write("\n")

        for d in list_1:
            for e in list_1:
                for f in list_1:
                    new.write("1" + "\t" + d + "," + f + "|||" + e)
                    new.write("\n")

        for g in list_1:
            for h in list_1:
                for i in list_1:
                    new.write("1" + "\t" + g + "|||" + h + "," + i)
                    new.write("\n")


def deal_final_positive():
    with open('dataset/positive_example.txt', 'r', encoding="utf-8") as f, \
            open('dataset/final_positive_example.txt', 'w', encoding="utf-8") as new:
        read_1 = f.readlines()
        resultList = random.sample(range(0, len(read_1) - 1), 6800)
        for i in resultList:
            new.write(read_1[i])


if __name__ == '__main__':
    deal_data()
    deal_final_positive()
