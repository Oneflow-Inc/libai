import re
import os
import matplotlib.pyplot as plt

log_directory = 'log'
output_image_path = log_directory +'/total_loss_plot.png'
pattern = r'total_loss:\s([0-9.]+)'

all_loss = {}
for filename in os.listdir(log_directory):
    if filename.endswith('loss.log'):
        file_path = os.path.join(log_directory, filename)

        total_loss_values = []

        with open(file_path, 'r') as file:
            for line in file:
                match = re.search(pattern, line)
                if match:
                    total_loss = float(match.group(1))
                    total_loss_values.append(total_loss)

        if total_loss_values:
            plt.plot(total_loss_values, label=filename)
            all_loss[filename] = total_loss_values

plt.xlabel('Iterations')
plt.ylabel('Total Loss')
plt.title('Total Loss over Iterations for Each Log File')
plt.legend()
plt.savefig(output_image_path, format='png')
print(f"图表已保存到 {output_image_path}")

def calculate_mae_rmae(all_values):
    baseline = all_values['cuda_loss.log']
    N = len(baseline)

    results = {}

    for key, values in all_values.items():
        if key == 'cuda_loss.log':
            continue

        #assert len(values) == N, f"Length mismatch between baseline and {key}, {len(values)} != {N}"

        #errors = [abs(b - v) for b, v in zip(baseline, values)]
        # 计算两个列表中最短长度
        N = min(len(baseline), len(values))
        #N = 3000
        # 根据最短长度比较
        errors = [abs(b - v) for b, v in zip(baseline[:N], values[:N])]

        mae = sum(errors) / N

        mean_baseline = sum(baseline) / N
        mean_values = sum(values) / N
        rmae = mae / (0.5 * (mean_baseline + mean_values))

        results[key] = {'MAE': mae, 'RMAE': rmae}

    return results

res = calculate_mae_rmae(all_loss)
print(res)
