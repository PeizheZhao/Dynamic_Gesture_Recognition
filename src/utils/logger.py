import csv
import os
from datetime import datetime

class Logger(object):
    def __init__(self, header):
        os.makedirs("logs", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.path = f"logs/log_{timestamp}.csv"
        
        self.log_file = open(self.path, 'w', newline='')
        self.logger = csv.writer(self.log_file, delimiter='\t')
        self.logger.writerow(header)
        self.header = header
        self.write_interval = 10  
        self.buffer = [] 

    def log(self, values):
        self.buffer.append(values)

        if int(values['Epoch']) % self.write_interval == 0:
            self.write_to_file()

    def write_to_file(self):
        rows = []
        epoch_set = set()

        try:
            with open(self.path, 'r', newline='') as read_file:
                reader = csv.DictReader(read_file, delimiter='\t')
                for row in reader:
                    rows.append(row)
                    epoch_set.add(int(row['Epoch']))
        except FileNotFoundError:
            pass

        for values in self.buffer:
            if int(values['Epoch']) in epoch_set:
                for row in rows:
                    if int(row['Epoch']) == int(values['Epoch']):
                        row.update(values)
                        break
            else:
                rows.append(values)

        self.buffer = []

        with open(self.path, 'w', newline='') as write_file:
            writer = csv.DictWriter(write_file, fieldnames=self.header, delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)
