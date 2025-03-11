import time
import csv
import click
from datetime import datetime
from config import settings
from fedn import APIClient

# List of VM names and client statuses
VM_NAMES = ["renberget", "svaipa", "mavas"] # Replace with machines that are running clients (start of name variable in run_clients.py)
CLIENT_STATUSES = ["online", "offline", "available"]

@click.command()
@click.option('--csv-filename', '-f', default=None, 
              help='CSV filename to store client status data. Defaults to a timestamped filename.')
@click.option('--interval', '-i', default=5, 
              help='Time interval in seconds between status checks. Default is 5 seconds.')
def monitor_client_status(csv_filename, interval):
    """Monitor and record the status of FEDn clients across different VMs."""
    
    # Set default filename with timestamp if not provided
    if not csv_filename:
        csv_filename = f"client_status_per_vm_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    
    api_client = APIClient(
        host=settings["DISCOVER_HOST"],
        port=settings["DISCOVER_PORT"],
        secure=settings["SECURE"],
        verify=settings["VERIFY"],
        token=settings["ADMIN_TOKEN"],
    )
    
    # Create header row dynamically based on VM names and statuses
    header = ['timestamp']
    for vm in VM_NAMES:
        for status in CLIENT_STATUSES:
            header.append(f'{vm}_{status}')
    
    # Create/open CSV file with header if it doesn't exist
    with open(csv_filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if f.tell() == 0:  # Check if file is empty
            writer.writerow(header)
    
    while True:
        fl_clients = api_client.get_clients()
        
        # Initialize counters for each host/status combination
        counts = {vm: {status: 0 for status in CLIENT_STATUSES} for vm in VM_NAMES}
        
        # Count clients for each combination
        for fl_client in fl_clients['result']:
            for vm in VM_NAMES:
                if fl_client['name'].startswith(vm):
                    status = fl_client['status']
                    if status in CLIENT_STATUSES:
                        counts[vm][status] += 1
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Prepare row data dynamically
        row_data = [timestamp]
        for vm in VM_NAMES:
            for status in CLIENT_STATUSES:
                row_data.append(counts[vm][status])
        
        # Write all counts to CSV
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
            
        time.sleep(interval)

if __name__ == "__main__":
    monitor_client_status()