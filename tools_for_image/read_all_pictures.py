import os


def read_pictures(path):
    files = os.listdir(path)
    all_dict = []
    for i in files:
        a = path + '/' + i
        all_dict.append(a)
    all_files = []
    for i in all_dict:
        b = os.listdir(i)
        for j in b:
            c = i + '/' + j
            all_files.append(c)
    return all_files


if __name__ == '__main__':
    path = '../data/picture'
    files = read_pictures(path)
    a = files[1].split('/', )
    b = a[4].split('.')
    print(a)
    print(b)
    num = (int(a[3]) - 1) * 100 + int(b[0])
    print(num, type(num))
