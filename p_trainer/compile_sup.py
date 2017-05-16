import sys
import numpy as np
import random
import IPython


#from alan.lfd_amazon.options import AmazonOptions as Options
#we can pass this into the Compile_Sup object.



class Compile_Sup:
    def __init__(self, Options):
        self.Options = Options

    def get_rollout(self,f_name):
        i = f_name.find('_')
        rollout_num = int(f_name[7:i])
        return rollout_num

    def compile_reg(self,img_path = None): #might need selfs here
        train_path = self.Options.train_file
        test_path = self.Options.test_file
        deltas_path = self.Options.deltas_file
        scale_constants = self.get_range()

        if(img_path == None):
            img_path = self.Options.binaries_dir

        print "Moving deltas from " + deltas_path + " to train: " + train_path + " and test: " + test_path
        train_file = open(train_path, 'w+')
        test_file = open(test_path, 'w+')
        deltas_file = open(deltas_path, 'r')
        i=0
        cur_rollout = 0
        p_rollout = -1

        for line in deltas_file:

            l = line.split()
            cur_rollout = self.get_rollout(l[0])

            if(cur_rollout != p_rollout):
                p_rollout = cur_rollout
                if random.random() > .2:
                    train = True
                else:
                    train = False

            path = img_path
            labels = line.split()

            print labels
        
            deltas = self.scale(labels[1:4],scale_constants)

            line = labels[0] + " "
            for bit in deltas:
                line += str(bit) + " "
            line = line[:-1] + '\n'

            if train:
                train_file.write(path + line)
            else:
                test_file.write(path + line)

            i=i+1

    if __name__ == '__main__':
        compile_reg()
