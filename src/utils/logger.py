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

    def __del__(self):
        self.log_file.close()

    def log(self, values):
        rows = []
        epoch_exists = False
        with open(self.path, 'r', newline='') as read_file:
            reader = csv.DictReader(read_file, delimiter='\t')
            for row in reader:
                if int(row['Epoch']) == int(values['Epoch']):
                    row.update(values)
                    epoch_exists = True
                rows.append(row)
        
        if not epoch_exists:
            rows.append(values)

        with open(self.path, 'w', newline='') as write_file:
            writer = csv.DictWriter(write_file, fieldnames=self.header, delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)
