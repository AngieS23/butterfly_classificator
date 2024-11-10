import csv

def write_epoch_metrics(metrics):
    with open('../results/epoch_metrics.csv', 'w', newline='') as csvfile:
        fieldnames = ['mode', 'epoch', 'metric', 'result']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

def write_final_results(metrics):
    with open('../results/final_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['mode', 'metric', 'result']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

def write_confusion_matrix(name, matrix):
    with open(f'../results/{name.lower()}_matrix.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(matrix)