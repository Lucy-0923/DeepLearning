import csv




def processdate(date):
    date_num = (int(date[:4]) - 2014)*12 + (int(date[4:6])-5)
    return date_num

path= "kc_house_data.csv"
with open(path, 'r') as f:
    csv_data = list(csv.reader(f))
for index in range(1, len(csv_data)):
    csv_data[index][1] = processdate(csv_data[index][1])
    csv_data[index][2] = eval(csv_data[index][2])

path2 = "kc_house_data_pro.csv"
with open(path2, 'w', newline='') as f:
    csv_writer = csv.writer(f)  # 初始化一个写文件器 writer
    csv_writer.writerows(csv_data)
    # for i in range(len(testset)):  # 把测试结果的每一行放入输出的excel表中。
    #     csv_writer.writerow([str(i), str(val_rel[i])])