import cv2
import numpy as np
import IPython
import cPickle as pickle

if __name__=='__main__':
    all_results = []
    print("AVG PERCENT IMPROVEMENTS")
    dirs = []
    dirs.append(("rollouts/rollout", 21))

    dirs.append(("rollouts2/dart_adv_stats/stat", 10))
    dirs.append(("rollouts2/stat_adv_analytic/stat", 7))
    dirs.append(("rollouts2/stats/stat", 10))
    dirs.append(("rollouts2/stats_adverserial/stat", 9))
    dirs.append(("rollouts2/stats_analytic/stat", 7))
    dirs.append(("rollouts2/stats_dart/stat", 10))
    dirs.append(("rollouts2/supervisor_dart/stat", 10))

    for statdir, statnum in dirs:
        all_results = []
        for rnum in range(0, statnum):
            # print("Processing rollout " + str(rnum))
            folder = statdir + "_" + str(rnum) + "/"
            if statnum == 21:
                results = pickle.load(open(folder + "results.p", "rb"))
            else:
                results = pickle.load(open(folder + "m2_results.p", "rb"))
            all_results.append(results)

        #side1 and side2 improvements
        improves = [0, 0]
        for res in all_results:
            initial = 100.0 - res[0]

            #res1 messed up
            improves[0] += (res[2] - res[0])/initial * 100
            improves[1] += (res[3] - res[0])/initial * 100

        improves = [a/(statnum * 1.0) for a in improves]

        print("Type:" + statdir)
        print("Improvements:" + str(improves))
        print("Num trials:" + str(statnum) + "\n")
        # #side1 before, side1 after, side2 before, side2 after
        # avgPercents = [0, 0, 0, 0]
        # #side1 diff, side2 diff, diff between end of side1 and start of side2
        # avgDeltas = [0, 0, 0]
        # for res in all_results:
        #     for i in range(4):
        #         avgPercents[i] += res[i]
        #     for j in range(2):
        #         avgDeltas[j] += res[j * 2 + 1] - res[j * 2]
        #     avgDeltas[2] += abs(res[2] - res[1])
        #
        # avgPercents = [a/(n * 1.0) for a in avgPercents]
        # avgDeltas = [a/(n * 1.0) for a in avgDeltas]
        #
        # #compute stddev of avg PERCENTS
        # squaredErrors = [0, 0, 0, 0]
        # for res in all_results:
        #     for i in range(4):
        #         squaredErrors[i] += (avgPercents[i] - res[i])**2
        # squaredErrors = [(a/(n*1.0))**(0.5) for a in squaredErrors]
        #
        # print("AVG PERCENTS")
        # print(avgPercents)
        # print("STD DEVS")
        # print(squaredErrors)
        # print("AVG DELTAS")
        # print(avgDeltas)
