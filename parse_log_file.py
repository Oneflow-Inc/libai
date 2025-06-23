import re
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime


def parse_log_file(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Extract the first and last timestamps
    first_time = None
    last_time = None
    for line in lines:
        match = re.match(r'\[(\d{2}/\d{2} \d{2}:\d{2}:\d{2})', line)
        if match:
            if first_time is None:
                first_time = match.group(1)
            last_time = match.group(1)

    # Calculate the duration
    if first_time and last_time:
        first_time_obj = datetime.strptime(first_time, '%m/%d %H:%M:%S')
        last_time_obj = datetime.strptime(last_time, '%m/%d %H:%M:%S')
        duration = (last_time_obj - first_time_obj).total_seconds()
    else:
        duration = None

    # Extract total_loss and total_throughput
    losses = []
    total_throughputs = []
    for line in lines:
        match = re.search(r'total_loss: (\d+\.\d+)', line)
        if match:
            losses.append(float(match.group(1)))
        match = re.search(r'total_throughput: (\d+\.\d+)', line)
        if match:
            total_throughputs.append(float(match.group(1)))

    # Extract latency
    latency = None
    for line in lines:
        match = re.search(r'Overall training speed: .+ \((\d+\.\d+) s / it\)', line)
        if match:
            latency = float(match.group(1))
            break

    # Prepare the result
    result = {
        'filename': log_file.name,
        'duration': duration,
        'latency': latency,
        'throughput': sum(total_throughputs) / len(total_throughputs) if total_throughputs else None,
        'losses': losses,
        #'total_throughputs': total_throughputs,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description='Parse log files and save results as JSON.')
    parser.add_argument('--log_dir', type=str, default='./log',
                        help='Directory containing log files (default: ./log)')
    parser.add_argument('--pattern', type=str, default='*.log',
                        help='Glob pattern for log files (default: *.log)')
    parser.add_argument('--output', type=str, default='training_results.csv',
                        help='Output csv file (default: training_results.csv)')
    # parser.add_argument('--output', type=str, default='training_results.json',
    #                     help='Output JSON file (default: training_results.json)')

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_files = sorted(log_dir.glob(args.pattern), key=lambda f: f.stat().st_mtime)

    results = []
    for log_file in log_files:
        result = parse_log_file(log_file)
        results.append(result)

    # with open(args.output, 'w') as f:
    #     f.write(',\n'.join(json.dumps(item, separators=(',', ': ')) for item in results))
    with open(args.output, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'duration', 'latency', 'throughput', 'losses']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            result['losses'] = json.dumps(result['losses'])
            writer.writerow(result)


if __name__ == '__main__':
    main()

