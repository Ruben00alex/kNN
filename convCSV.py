import csv


def convFile(inFile, outFile):
    with open(inFile, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        with open(outFile, 'w') as f2:
            writer = csv.writer(f2, delimiter=',')
            for row in reader:
                writer.writerow([row[1], row[2], row[3], row[4], row[5]])


convFile('participante_yh13_dataset.csv', 'p13.csv')
