import pandas as pd
#读取文件
data = pd.read_csv("../Dataset/second_cars_info.csv", encoding='GB18030')
#数据清洗
delete_nocard = data.loc[data.Boarding_time !="未上牌",:]
nocard_car_total = data[data.Boarding_time == "未上牌"].count()[0]
print("未上牌比例：",nocard_car_total/11281)

print(data.shape)