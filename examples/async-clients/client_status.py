import time
import csv
from datetime import datetime
from config import settings
from fedn import APIClient

api_client = APIClient(
    host=settings["DISCOVER_HOST"],
    port=settings["DISCOVER_PORT"],
    secure=settings["SECURE"],
    verify=settings["VERIFY"],
    token=settings["ADMIN_TOKEN"],
)

# Create/open CSV file with header if it doesn't exist
csv_filename = 'client_status_per_vm.csv'
with open(csv_filename, 'a', newline='') as f:
    writer = csv.writer(f)
    if f.tell() == 0:  # Check if file is empty
        writer.writerow(['timestamp', 
                        'renberget_online', 'renberget_offline', 'renberget_available',
                        'svaipa_online', 'svaipa_offline', 'svaipa_available',
                        'mavas_online', 'mavas_offline', 'mavas_available'])

try:
    while True:
        fl_clients = api_client.get_clients()
        
        # Initialize counters for each host/status combination
        counts = {
            'renberget': {'online': 0, 'offline': 0, 'available': 0},
            'svaipa': {'online': 0, 'offline': 0, 'available': 0},
            'mavas': {'online': 0, 'offline': 0, 'available': 0}
        }
        
        # Count clients for each combination
        for fl_client in fl_clients['result']:
            for host in ['renberget', 'svaipa', 'mavas']:
                if fl_client['name'].startswith(host):
                    status = fl_client['status']
                    counts[host][status] += 1
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Write all counts to CSV in a pandas/matplotlib friendly format
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                counts['renberget']['online'], 
                counts['renberget']['offline'],
                counts['renberget']['available'],
                counts['svaipa']['online'],
                counts['svaipa']['offline'], 
                counts['svaipa']['available'],
                counts['mavas']['online'],
                counts['mavas']['offline'],
                counts['mavas']['available']
            ])
        time.sleep(5)
except KeyboardInterrupt:
    print("\nStopped monitoring clients")
    print(f"Data saved to {csv_filename}. Use pandas.read_csv('{csv_filename}') to analyze the data.")