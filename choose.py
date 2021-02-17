import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--list_a', type=str, default='/extracephonline/medai_data2/zhengdzhang/eyes/qikan/path/to/dataset-new/source/eyesgan/level_0.5-0.7/noise_labels_0.5/noise_selected_0.1.txt')
parser.add_argument('--list_b', type=str, default='/extracephonline/medai_data2/zhengdzhang/eyes/qikan/pOSAL-master-noise/data/refuge-new/level_0.5-0.7/noise_labels_0.5/noise_sample_0.1.txt')
parser.add_argument('--list_noise_save', type=str, default='/extracephonline/medai_data2/zhengdzhang/eyes/qikan/correction/refuge/level_0.5-0.7/select_0.1/jiao.txt')
args = parser.parse_args()
print(args)
with open(args.list_a) as g:
    x = g.read()
    b = x.split('\n')
    print(len(b)-1)
with open(args.list_b) as f:
    data = f.read()
    a = data.split('\n')
    print(len(a)-1)
num = 0
listx= []
for i in range(0, (len(a)-1)):
    for j in range(0, (len(b)-1)):
        if a[i]==b[j]:
            num = num + 1
            listx.append(a[i])
print(num)
print(num/(len(a)-1))

with open(args.list_noise_save, "w") as h:
    for i in range(0, num):
        h.write(str(listx[i])+'\n')

