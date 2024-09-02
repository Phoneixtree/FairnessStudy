from MCboost import MCboost
import time
import csv

def main():
    start=time.perf_counter()
    m=MCboost()
    end=time.perf_counter()
    f=[round(i) for i in m.f]
    #o=[round(i) for i in m.origin]
    f=m.f
    o=m.origin
    t=m.t

    with open('comparison.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        for v1, v2,v3 in zip(f,t,o):
            writer.writerow([v1, v2, v3])

    print('running time:{:.3f} in seconds'.format(end-start))

if __name__=="__main__":
    main()