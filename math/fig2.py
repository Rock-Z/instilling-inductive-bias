import pandas as pd
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

#import scienceplots
#plt.style.use('science')

#style = plt.style.core.read_style_directory("style.mplstyle")

plt.style.use('style.mplstyle')
matplotlib.rcParams["savefig.bbox"] =  "tight"
matplotlib.rcParams["savefig.pad_inches"] =  0.05

make_path = lambda layers, frozen: f"checkpoints/{layers}layers_val/transfer_1e-7_frozen" if frozen else f"checkpoints/{layers}layers_val/transfer_1e-7_trainable"
read_logs = lambda path: pd.read_csv(os.path.join(path, "accuracy_logs.csv"), index_col=0)

result_dfs = []

trainable_subnet_results = []

for layers in [2, 4, 6, 8, 10, 12]:
    plots_row = []
    for frozen in [True]:
        path =make_path(layers, frozen)        
        result_dfs.append(read_logs(path))
        
    trainable_subnet_results.append(read_logs(make_path(layers, False)))

n_disambiguation = result_dfs[0][result_dfs[0]["inherit_from"] == "subnet"]["n_disambiguation"].values[:16]
subnet_logs = np.mean([logs[logs["inherit_from"] == "subnet"]["best_test_acc"].values[:16] for logs in result_dfs], axis = 0)
sampled_logs = np.mean([logs[logs["inherit_from"] == "subnet_sampled"]["best_test_acc"].values[:16] for logs in result_dfs], axis = 0)
complement_logs = np.mean([logs[logs["inherit_from"] == "subnet_sampled_complement"]["best_test_acc"].values[:16] for logs in result_dfs], axis = 0)
full_model_logs = np.mean([logs[logs["inherit_from"] == "original"]["best_test_acc"].values[:16] for logs in result_dfs], axis = 0)
random_init_logs = np.mean([logs[logs["inherit_from"] == "random"]["best_test_acc"].values[:16] for logs in result_dfs], axis = 0)

trainable_subnet_logs = np.mean([logs[logs["inherit_from"] == "subnet_sampled_complement"]["best_test_acc"].values[:16] for logs in trainable_subnet_results], axis = 0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2), sharex=True)

ax1.plot(n_disambiguation, subnet_logs, marker='.', label = "Subtask Induction")
ax1.plot(n_disambiguation, full_model_logs, marker='d', markersize=4, label = "Full Model Transfer")
ax1.plot(n_disambiguation, sampled_logs, marker='^', markersize=4, label = "Sampled Subnetwork")
ax1.plot(n_disambiguation, random_init_logs, marker='s', markersize=3.5, label = "Random Initialization")

ax1.set_yticks(np.arange(0.3, 1.1, 0.1))
ax1.set_xlabel('Number of disambiguation training samples')
ax1.set_ylabel('Test Accuracy')
ax1.set_title("Mean Over All Model Configurations")
ax1.set_xscale("log")
ax1.legend(loc='upper left', facecolor='white', framealpha=0.5, fontsize=8)
#ax1.xaxis.grid(True, which='both', color='grey', alpha=0.3)
ax1.grid(alpha=0.3, linewidth=0.5)

make_path_12l = lambda run: f"checkpoints/12layers_val/transfer_1e-7_run{run}"
read_logs = lambda path: pd.read_csv(os.path.join(path, "accuracy_logs.csv"), index_col=0)

result_dfs = []

trainable_subnet_results = []

for run in range(5):
    plots_row = []
    path =make_path_12l(run)        
    result_dfs.append(read_logs(path))

n_disambiguation = result_dfs[0][result_dfs[0]["inherit_from"] == "subnet"]["n_disambiguation"].values[:17]
subnet_logs = np.mean([logs[logs["inherit_from"] == "subnet"]["best_test_acc"].values[:17] for logs in result_dfs], axis = 0)
subnet_std = np.std([logs[logs["inherit_from"] == "subnet"]["best_test_acc"].values[:17] for logs in result_dfs], axis = 0)
sampled_logs = np.mean([logs[logs["inherit_from"] == "subnet_sampled"]["best_test_acc"].values[:17] for logs in result_dfs], axis = 0)
sampled_std = np.std([logs[logs["inherit_from"] == "subnet_sampled"]["best_test_acc"].values[:17] for logs in result_dfs], axis = 0)
full_model_logs = np.mean([logs[logs["inherit_from"] == "original"]["best_test_acc"].values[:17] for logs in result_dfs], axis = 0)
full_model_std = np.std([logs[logs["inherit_from"] == "original"]["best_test_acc"].values[:17] for logs in result_dfs], axis = 0)
random_init_logs = np.mean([logs[logs["inherit_from"] == "random"]["best_test_acc"].values[:17] for logs in result_dfs], axis = 0)
random_init_std = np.std([logs[logs["inherit_from"] == "random"]["best_test_acc"].values[:17] for logs in result_dfs], axis = 0)

trainable_subnet_logs = np.mean([logs[logs["inherit_from"] == "subnet_sampled_complement"]["best_test_acc"].values[:16] for logs in trainable_subnet_results], axis = 0)

ax2.errorbar(n_disambiguation, subnet_logs, yerr=subnet_std, marker='.', capsize=2, elinewidth=0.8, capthick=0.8, label = "Subtask Induction")
ax2.fill_between(n_disambiguation, subnet_logs - subnet_std, subnet_logs + subnet_std, alpha=0.2)
ax2.errorbar(n_disambiguation, full_model_logs, yerr=full_model_std, marker='d', markersize=4, capsize=2, elinewidth=0.8, capthick=0.8, label = "Full Model Transfer")
ax2.fill_between(n_disambiguation, full_model_logs - full_model_std, full_model_logs + full_model_std, alpha=0.2)
ax2.errorbar(n_disambiguation, sampled_logs, yerr=sampled_std, marker='^', markersize=4, capsize=2, elinewidth=0.8, capthick=0.8, label = "Sampled Subnetwork")
ax2.fill_between(n_disambiguation, sampled_logs - sampled_std, sampled_logs + sampled_std, alpha=0.2)
ax2.errorbar(n_disambiguation, random_init_logs, marker='s', yerr=random_init_std, markersize=3.5, capsize=2, elinewidth=0.8, capthick=0.8, label = "Random Initialization")
ax2.fill_between(n_disambiguation, random_init_logs - random_init_std, random_init_logs + random_init_std, alpha=0.2)

ax2.set_yticks(np.arange(0.3, 1.1, 0.1))
ax2.set_xlabel('Number of disambiguation training samples')
ax2.set_xscale("log")
ax2.set_title("GPT-2, 12 Layers")
#ax2.xaxis.grid(True, which='both', color='grey', alpha=0.3)
ax2.grid(alpha=0.3, linewidth=0.5)

#fig.suptitle("Mean Test Accuracy vs Amount of Training Data", y =1.05, fontsize=14.5)

plt.savefig("fig2.png", dpi = 1200)
plt.savefig("fig2.svg")
