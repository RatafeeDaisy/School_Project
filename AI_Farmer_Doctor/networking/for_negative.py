def deal_for_fuli():
    with open('dataset/negative_example.txt', 'r', encoding="utf-8") as f, open('dataset/final_negative_example.txt',
                                                                                'w', encoding="utf-8") as new, open(
            'dataset/final_positive_example.txt', 'r', encoding="utf-8") as kl:
        read_1 = f.readlines()
        read_2 = kl.readlines()
        list_fuli = []
        list_zhenli = []
        for a in [a.strip('\n').replace("\t", "", 1).replace("0", "", 1).replace("\t", "|||", 1) for a in read_1]:
            list_fuli.append(a.split("|||")[1])
        for q in [q.strip('\n').replace("\t", "", 1).replace("1", "", 1).replace("\t", "|||", 1) for q in read_2]:
            list_zhenli.append(q.split("|||")[0])
        for p in range(len(list_zhenli)):
            str_1 = "0" + "\t" + list_zhenli[p] + "\t" + list_fuli[p]
            new.write(str_1 + "\n")


if __name__ == '__main__':
    deal_for_fuli()
