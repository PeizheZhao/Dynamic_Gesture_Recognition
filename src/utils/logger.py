import csv
import os
from datetime import datetime
from collections import defaultdict

class Logger(object):
    def __init__(self, header):
        os.makedirs("logs", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.path = f"logs/log_{timestamp}.csv"
        
        self.log_file = open(self.path, 'w', newline='')
        self.logger = csv.writer(self.log_file, delimiter='\t')
        self.logger.writerow(header)
        self.header = header
        self.write_interval = 5  
        self.buffer = [] 

    def log(self, values):
        self.buffer.append(values)

        if int(values['Epoch']) % self.write_interval == 0:
            self.write_to_file()
    
    def merge_logs_by_epoch(self, buffer_dict):
        epoch_dict = defaultdict(dict)
        for d in buffer_dict:
            epoch = int(d['Epoch'])
            epoch_dict[epoch].update({k: v for k, v in d.items() if v != ''})
        return list(epoch_dict.values())

    def write_to_file(self):
        rows = self.merge_logs_by_epoch(self.buffer)

        with open(self.path, 'w', newline='') as write_file:
            writer = csv.DictWriter(write_file, fieldnames=self.header, delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)