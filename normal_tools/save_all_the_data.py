from normal_tools import save_data
def save_all_data_two(i,j,path,data):
    str3 = "total number" + str(i) + "predict" + str(j) + "time"
    c = path + "/" + str3 + ".csv"
    save_data.save_data(c, data)

def save_all_data_one(i,path,data):
    str2 = "data number" + str(i) + "score"
    b = path + "/" + str2 + ".csv"
    save_data.save_data(b, data)
