import numpy
import os
import glob
import sys
import socket
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

base_path = os.getenv('DRLNAV_BASE_PATH') + "/src/turtlebot3_drl/model/" + str(socket.gethostname() + "/")

def main(args=sys.argv[1:]):
    EPISODES = int(args[0])
    models = args[1:]

    plt.figure(figsize=(16,10))

    j = 0
    for model in models:
        logfile = glob.glob(base_path + model + '/_train_*.txt')
        if len(logfile) != 1:
            print(f"ERROR: found less or more than 1 logfile for: {base_path}{model}")
        df = pd.read_csv(logfile[0])
        outcome_column = df[' success']
        outcome_column = outcome_column.tolist()
        success_history = []
        success_count = 0
        episode_range = min(EPISODES, len(df.index))
        xaxis = numpy.array(range(0, episode_range))
        for i in range (episode_range):
            if outcome_column[i] == 1:
                success_count += 1
            success_history.append(success_count)
        plt.plot(xaxis, success_history, label=models[j])
        j += 1

    plt.ylabel('Success count', fontsize=25, fontweight='bold')
    plt.xlabel('Episode', fontsize=25, fontweight='bold')

    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=25, ncols=2, prop={'size': 20})

    plt.grid(True, linestyle='--')
    # plt.savefig("graph_output.pdf", format="pdf", bbox_inches="tight")
    dt_string = datetime.now().strftime("%d-%m-%H:%M:%S")
    suffix = '-'.join(models).replace(' ', '_').replace('/', '-')
    plt.savefig(os.path.join("graphs/", 'success_graph_' + dt_string + '__' + suffix + ".png"), format="png", bbox_inches="tight")
    plt.show()

if __name__ == '__main__':
    main()