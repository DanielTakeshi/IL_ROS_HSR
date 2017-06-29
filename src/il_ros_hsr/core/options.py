"""
Options Class for Neural Network Go to State Policies 
Specifies Parameters for Learning and File Structure

Author: Michael Laskey


"""


import os

class Options():
    translations = [0.0, 0.0, 0.0, 0.0]
    scales = [40.0, 120.0, 90.0, 100.0]
    drift = 20.0


    def setup(self,root_dir,setup_dir,folder = None):
        """
        Sets the file path to store the image data and labels 

        Parameters
        ----------
        root_dir: string
            Specifies the root directory to save everything in 

        setup_dir: string
            Specifies the specific neural network that data is being collected for  
        
        folder: string
            Specifies in a higher folder should be made to save multiple network files
            (Defualt=None and corresponds to one net per primitive)

        """

        if(not folder == None):
            root_dir = root_dir+folder+'/'

        
        self.data_dir = root_dir + 'data/'
        self.datasets_dir = self.data_dir + 'datasets/'
        self.frames_dir = self.data_dir + 'record_frames/'
        self.videos_dir = self.data_dir + 'record_videos/'
        self.hdf_dir = root_dir + 'Net/hdf/'
        self.tf_dir = root_dir + 'Net/tensor/'
        self.images_dir = self.data_dir + 'images/'
    

        self.setup_dir = self.data_dir + setup_dir

        self.train_file = self.setup_dir + "train.txt"        #if code doesn't work properly, add a "optType."
        self.test_file = self.setup_dir + "test.txt"
        self.deltas_file = self.setup_dir + "deltas.txt"
        self.rollouts_file = self.setup_dir + "rollouts.txt"

        self.rollouts_dir = self.setup_dir + "rollouts/"

        self.evaluations_dir = self.setup_dir + "evaluation/"
        self.movies_dir = self.setup_dir + "movies/"
        self.sup_dir = self.setup_dir + "supervisor/"
        self.binaries_dir = self.setup_dir + "binaries/"
        self.stats_dir = self.setup_dir + "stats/"

        self.originals_dir = self.setup_dir + "originals/"

        self.policies_dir = self.setup_dir + "policies/"

        self.test = False
        self.deploy = False
        self.learn = False
        self.model_path = ""        # path to network architecture prototxt
        self.weights_path = ""      # path to weights (should match model)
        self.show = False           # whether to show a preview window from bincam
        self.record = False         # whether to record frames with bincam
        self.scales = Options.scales
        self.translations = Options.translations
        self.drift = Options.drift
        self.tf_net = None          # tensorflow model
        self.tf_net_path = ""       # path to tensorflow model's weights
