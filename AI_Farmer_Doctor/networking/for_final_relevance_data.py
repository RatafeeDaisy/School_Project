def deal_final_positive():
    with open('dataset/final_positive_example.txt', 'r', encoding="utf-8") as f1, open(
            'dataset/final_negative_example.txt', 'r', encoding="utf-8") as f2, \
            open('dataset/train_data.csv', 'w', encoding="utf-8") as new:
        read_1 = f1.readlines()
        read_2 = f2.readlines()
        for a in [a.replace("|||", "\t", 1) for a in read_1]:
            new.write(a)
        for b in read_2:
            new.write(b)


if __name__ == '__main__':
    deal_final_positive()
