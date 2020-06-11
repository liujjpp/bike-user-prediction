import csv
import datetime
import math


def preprocess(src, dest):
    f = csv.reader(open(src, 'r'))
    out = open(dest, 'a', newline='')
    w = csv.writer(out, dialect='excel')

    is_title = True

    for record in f:
        if is_title:
            is_title = False
            continue

        if len(record[8]) < 1 or len(record[9]) < 1 or len(record[10]) < 1 or len(record[11]) < 1:
            continue

        row = []

        start_time = datetime.datetime.strptime(record[2], "%Y-%m-%d %H:%M:%S")
        start_hour = record[2].split()[1].split(':')[0]
        end_time = datetime.datetime.strptime(record[3], "%Y-%m-%d %H:%M:%S")
        end_hour = record[3].split()[1].split(':')[0]
        seconds = (end_time - start_time).total_seconds()
        if seconds <= 0:
            continue
        row.append(math.log(seconds))
        row.append(float(start_hour))
        row.append(float(end_hour))

        row.append(float(record[8]))
        row.append(float(record[9]))
        row.append(float(record[10]))
        row.append(float(record[11]))

        if record[-1] == 'member':
            row.append(1)
        else:
            row.append(0)

        w.writerow(row)


if __name__ == "__main__":
    preprocess('./202004-capitalbikeshare-tripdata.csv', './data.csv')
