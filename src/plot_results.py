import matplotlib.pyplot as plt
import ast
import sys

NUM_CLIENTS = int(sys.argv[1])
ILR = float(sys.argv[2])
DECAY_RATE = float(sys.argv[3])
NUM_EPOCHS = int(sys.argv[4])
NUM_ROUNDS = int(sys.argv[5])

results_file_path = 'results_{0}_{1}_{2}_{3}_{4}.txt'.format(str(NUM_CLIENTS),
                                                                     str(ILR),
                                                                     str(DECAY_RATE),
                                                                     str(NUM_EPOCHS),
                                                                     str(NUM_ROUNDS))
results_file = open(results_file_path, 'r')
results = results_file.read().split('\n')
results_file.close()

for i in range(len(results) - 1):
    results[i] = ast.literal_eval(results[i])
    plt.plot(*zip(*results[i]['acc']), label=str(results[i]['strategy']))

plt.xlabel("Number of rounds")
plt.ylabel("Accuracy")
plt.title("Accuracy of central model over test data")
plt.legend(loc='best')
# plt.show()
plots_path = 'results_{0}_{1}_{2}_{3}_{4}.png'.format(str(NUM_CLIENTS),
                                                            str(ILR),
                                                            str(DECAY_RATE),
                                                            str(NUM_EPOCHS),
                                                            str(NUM_ROUNDS))
plt.savefig(plots_path, dpi=500)
