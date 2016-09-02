import numpy as np
import matplotlib.pyplot as plt
import re
from scipy import linalg
#from ipywidgets import *
#from IPython.display import *
from collections import Counter

class Network(object):

    def __init__(self, trainLen=0, testLen=0, initLen=100) :
        """Initialization of the network. Contains
        default values that can be modified."""
        
        #Data type: "characters", "words", "pixels", "images"
        self.data_type = "characters"
        self.file = self.file = open("text/Shakespeare.txt", "r").read()

        #Network lengths:
        self.initLen = initLen # remove the corresponding number of time steps to trainLen
        self.trainLen = trainLen # trainLen *includes* initLen (i.e. the learning happens on (trainLen - initLen))
        self.testLen = testLen
        self.auto_adapt_initLen = True #will auto-adapt the initLen to the reservoir size and leak rate (a)

        #Network size:
        self.resSize = 0

        #Network formula constants :
        self.a = 0.3
        self.spectral_radius = 0.25
        self.input_scaling = 1.
        
        #Learning parameters
        self.reg =  1e-8 # ridge regression parameters (for offline learning only for the moment)
        self.learning_rate = 10**-4 #10**-3 # for online learning only

        #Network mode
        self.mode = 'prediction'
        self.compute_type = "offline"

        #Random seed
        self.seed = None #42

    def set_seed(self):
        """Making the seed (for random values) variable if None."""

        if self.seed is None:
            import time
            self.seed = int((time.time()*10**6) % 4294967295)
        try:
            np.random.seed(self.seed)
            print("Seed used for random values:", self.seed, "\n")
        except:
            print("!!! WARNING !!!: Seed was not set correctly.")
        return self.seed

    def filter_characters(self, keep_upper=True, keep_punctuation=True, keep_numbers=True) :
        """Filters the characters of the input, returns self.input_text,
        a list corresponding to the filtered text, where every element is
        a unique character.
        - keep_upper (default True) : keeps uppercase characters ; if False, converts the text
        in lowcase.
        - keep_punctuation (default True) : keeps punctuation signs (i.e., every character that
        isn't a letter, a number or a space) ; if False, deletes every punctuation sign.
        - keep_numbers (default True) : keeps numbers in the text ; if False, deletes every
        number."""

        alphabet = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")
        numbers = list("0123456789")

        if keep_upper == False : self.file = self.file.lower()
        self.input_text = list(self.file)

        if keep_punctuation == False :
            self.input_text = [i for i in self.input_text if i in alphabet]     

        if keep_numbers == False :
            self.input_text = [i for i in self.input_text if i not in numbers]

    def characters(self) :
        """Creates the input/output units according to all the different characters
        in the text, and prints the corresponding list and its length."""
        self.input_units, self.output_units = dict(), dict()
        for i, item in enumerate(set(self.input_text)) : self.input_units[item] = i
        for i, item in enumerate(set(self.input_text)) : self.output_units[i] = item
        #self.input_units = dict(enumerate(set(self.input_text)))
        print("\nExisting characters in the text :", sorted(self.input_units),"\nNumber of different characters :", len(self.input_units), "\n")

    def words(self) :
        """Creates the input units according to all the different words
        in the text, and prints the number of different words."""
        self.input_text = re.findall(r"[\w']+|[.,!?;]", "".join(self.input_text))
        self.input_units, self.output_units = dict(), dict()
        for i, item in enumerate(set(self.input_text)) : self.input_units[item] = i
        for i, item in enumerate(set(self.input_text)) : self.output_units[i] = item
        print("\nNumber of different words :", len(self.input_units), "\n")

    def convert_input(self) :
        print("Converting input into ID numbers...", end=" ")
        self.data = np.array([self.input_units[i] for i in self.input_text])
        #self.data = np.array([self.input_units.index(i) for i in self.input_text])
        self.inSize = self.outSize = len(self.input_units)
        print("done.")

    def binary_data(self) :
        #TODO: limit size of data_b to the number of characters asked by the user (i.e self.trainLen = self.testLen)
        #TODO: adapt the modulo(self.trainLen) /self.trainLen in all the code
        print("Creating the input binary matrix...", end=" ")
        self.data_b = np.zeros((len(self.input_text), len(self.input_units)))
        for i, item in enumerate(self.data) :
            self.data_b[i][item] = 1
        #np.save("inputdata.np", self.data_b)
        print("done.\n")

    def initialization(self) :
        print("_"*20,"\n\n  LAUNCH NUMBER ", self.current_launch+1,"\n","_"*20,sep="")
        print("\nInitializing the network matrices...", end=" ")
        if self.trainLen-self.initLen <= 0:
            raise Exception("The training length (i.e. number of time steps) is lower than the initialization length, should be bigger.")
        self.set_seed()
        self.Win = (np.random.rand(self.resSize,1+self.inSize)-0.5) * self.input_scaling
        self.W = np.random.rand(self.resSize,self.resSize)-0.5 
        self.X = np.zeros((1+self.inSize+self.resSize,self.trainLen-self.initLen))
        self.Ytarget = self.data_b[self.initLen+1:self.trainLen+1].T
        self.x = np.zeros((self.resSize,1))
        if self.compute_type == "online":
            self.Wout = np.zeros((self.inSize,self.resSize+self.inSize+1))
        print("done.")

    def compute_spectral_radius(self):
        print('Computing spectral radius...',end=" ")
        rhoW = max(abs(linalg.eig(self.W)[0]))
        print('done.')
        self.W *= self.spectral_radius / rhoW

    def run_and_record_network(self) :
        print('Training the network...', end=" ")
        percent = 0.1
        for t in range(self.trainLen):
            percent = self.progression(percent, t, self.trainLen)
            self.u = self.data_b[t%len(self.data)] #%len(self.data) : we return at the beginning of text if we reach the end.
            self.x = (1-self.a)*self.x + self.a*np.tanh( np.dot(self.Win, np.concatenate((np.array([1]),self.u)).reshape(len(self.input_units)+1,1) ) + np.dot( self.W, self.x ) )
            if t >= self.initLen :
                #TODO: why do you use x[:,0] ?
                self.X[:,t-self.initLen] = np.concatenate((np.array([1]),self.u,self.x[:,0])).reshape(len(self.input_units)+self.resSize+1,1)[:,0]      
        print('done.')

    def train_output(self) :
        print('Training the output...', end=" ")
        self.X_T = self.X.T
        if self.reg is not None:
            self.Wout = np.dot(np.dot(self.Ytarget,self.X_T), linalg.inv(np.dot(self.X,self.X_T) + \
                self.reg*np.eye(1+self.inSize+self.resSize) ) )
        else:
            self.Wout = np.dot(self.Ytarget, linalg.pinv(self.X) )   
        print('done.')

    def test(self) :
        print('Testing the network... (', self.mode, ' mode)', sep="", end=" ")
        self.Y = np.zeros((self.outSize,self.testLen))
        self.u = self.data_b[self.trainLen%len(self.data)] 
        percent = 0.1
        for t in range(self.testLen):
            percent = self.progression(percent, t, self.trainLen)
            self.x = (1-self.a)*self.x + self.a*np.tanh( np.dot(self.Win, np.concatenate((np.array([1]),self.u)).reshape(len(self.input_units)+1,1)\
                                                       ) + np.dot(self.W,self.x ) )
            self.y = np.dot(self.Wout, np.concatenate((np.array([1]),self.u,self.x[:,0])).reshape(len(self.input_units)+self.resSize+1,1)[:,0] )
            self.Y[:,t] = self.y
            if self.mode == 'generative':
                # generative mode:
                self.u = self.y
            elif self.mode == 'prediction':
                ## predictive mode:
                self.u = np.zeros(len(self.input_units))
                self.u[self.data[(self.trainLen+t+1)%len(self.data)]] = 1
            else:
                raise(Exception, "ERROR: 'mode' was not set correctly.")
        print('done.\n')

    def compute_error(self) :
        print("Computing the error...", end=" ")
        errorLen = self.testLen #500
        #TODO: check if %len(self.data) still works here
        mse = sum( np.square( self.data[(self.trainLen+1)%len(self.data):(self.trainLen+errorLen+1)%len(self.data)] - self.Y[0,0:errorLen] ) ) / errorLen
        print('MSE = ' + str( mse ))

    def probabilities(self, i) :
        """ Provide a vector of probabilities for the character/word output.
        filter0: replacing x<0 by 0, x within [0,+infinity[
        filter01: replacing x<0 by 0, replacing x>1 by 1, x within [0,1]        
        add_min: scaling all the outputs such as the min value equals 0, x within [0,+infinity[
        max: (does nothing to the values, because we will take the maximum value anyway, so we do not need to compute probabilities)
        """
        if self.probamode == "filter0" :
#            proba_weights = abs((self.Y.T[i] > 0)*self.Y.T[i])
            proba_weights = (self.Y.T[i] > 0)*self.Y.T[i] # should work without abs
        elif self.probamode == "filter01" :
#            proba_weights = abs((self.Y.T[i] > 0)*self.Y.T[i])
#            proba_weights = proba_weights-((proba_weights > 1)*proba_weights) + (proba_weights > 1)*1
            proba_weights = np.all([(0<=self.Y.T[i]), (self.Y.T[i]<=1)], axis=0)*self.Y.T[i] + (self.Y.T[i]>1)*1.
        elif self.probamode == "add_min" :
            proba_weights = (self.Y.T[i]) - np.min(self.Y.T[i])
        elif self.probamode == "max" :
            proba_weights = self.Y.T[i]
        proba_weights = proba_weights/sum(proba_weights)
        return(proba_weights)

    def convert_output(self) :
        print("Converting the output...", end=" ")
        self.output_text = ""
        for i in range(len(self.Y.T)) :
            proba_weights = self.probabilities(i)
            if not self.probamode == "max" :
                self.output_text += np.random.choice(list(self.output_units), p=proba_weights)
            else :
                self.output_text += self.output_units[np.argmax(proba_weights)]
        print("done.")

    def record_output(self) :
        print("Saving the output as a text file.")
        record_file = open("out/output"+str(self.current_launch)+".txt", "w")
        record_file.write(self.output_text)
        record_file.close()

    def words_list(self, existing_words=True, language="EN") :

        print("_"*20)
        print("\n  TRIAL NUMBER", self.current_launch+1)
        print("_"*20)
        
        if existing_words == True :
            if language == "EN" : words_dict = open("words_list_EN.txt", "r").read()
            elif language == "FR" : words_dict = open("words_list_FR.txt", "r").read()
            words_dict = words_dict.split()

        alphabet = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ")
        self.allwords = "".join([i for i in self.output_text if i in alphabet])
        self.allwords = self.allwords.lower().split()
        words_occurences = Counter(self.allwords)
        
        if existing_words == False :
            print("\nMost common words (real or not) in the generated text", end="\n\n")
            longest_size = len(sorted(list(words_occurences.most_common(self.nb_words)), key=len)[-1])
            print("| Word", end=" "*(max(longest_size,4)-3))
            print("| Occurences ")
            print("-"*(longest_size+17))
            words_occurences = words_occurences.most_common(self.nb_words)
            
            for i in range(self.nb_words) :
                w = str(words_occurences[i][0])
                n = str(words_occurences[i][1])
                print("| " + w + " "*(max(longest_size,3)-len(w)+2) + "| " + n)
            print("-"*(longest_size+17))
                
        else :
            print("\nMost common valid words in the generated text", end="\n\n")
            words_in_dictionary = set(words_dict) & set(self.allwords)
            i = 0
            j = 0
            real_words_occurences = []
            words_occurences = words_occurences.most_common()
            
            while i < self.nb_words :
                if words_occurences[j][0] in words_in_dictionary :
                    real_words_occurences.append(words_occurences[j])
                    i += 1
                j += 1
                if j == len(words_occurences) :
                    break

            longest_size = len(sorted(words_in_dictionary, key=len)[-1])
            print("| Word", end=" "*(max(longest_size,4)-3))
            print("| Occurences ")
            print("-"*(longest_size+17))
            
            for k in range(i) :
                w = str(real_words_occurences[k][0])
                n = str(real_words_occurences[k][1])
                print("| " + w + " "*(max(longest_size,3)-len(w)+2) + "| " + n)
            print("-"*(longest_size+17))
            print("\nLongest valid word :", sorted(words_in_dictionary, key=len)[-1])

    def progression(self, percent, i, total) :
        if i == 0 :
            print("Progress :", end= " ")
            percent = 0.1
        elif (i/total) > percent :
#            print(round(percent*100), end="")
#            print("%", end=" ")
            print(round(percent*100),"%")
            percent += 0.1
#        if total-i == 1 :
        elif total-i == 1 :
            print("100%")

        return(percent)

    def setup_user(self):
        #TYPE AND DATA SETUP
        self.type = 0
        while self.type not in [1, 2] :
            print("Type of input/output?\n 1. Characters\n 2. Words\n > ", end="")
            self.type = int(input())

        self.file = 0
        while self.file not in [1, 2, 3, 4] :
            print("\nInput text?\n 1. Shakespeare's complete works(4 573 338 chars.)\n 2. Sherlock Holmes (3 868 223 chars.)\n 3. Harry Potter and the Sorcerer's Stone (439 743 chars)\n 4. Harry Potter and the Prisoner of Azkaban (611 584 chars.)\n > ", end="")
            self.file = int(input())
        texts = ["Shakespeare.txt", "SherlockHolmes.txt", "HarryPotter1.txt", "HarryPotter3.txt"]
        self.file = open("text/"+texts[self.file-1], "r").read()

        selectmode = 0
        while selectmode not in [1, 2] :
            selectmode = int(input("\nMode?\n 1. Prediction\n 2. Generative\n > "))
        if selectmode == 1 : self.mode = 'prediction'
        else : self.mode = 'generative'

        selecttype = 0
        while selecttype not in [1, 2] :
            selecttype = int(input("\nComputing type?\n 1. Offline\n 2. Online\n > "))
        if selecttype == 1 : self.compute_type = "offline"
        else : self.compute_type = "online"

        #CHARACTERS SETUP
        keep_upper, keep_punctuation, keep_numbers = "", "", ""

        dico_yn = {"Y" : True, "O" : True, "T" : True,
                   "N" : False, "F" : False}

        while keep_upper not in [True, False] :
            keep_upper = input("\nKeep upper case letters? Y/N ")
            try :
                keep_upper = dico_yn[keep_upper.upper()]
            except :
                pass

        while keep_punctuation not in [True, False] :
            keep_punctuation = input("Keep punctuation? Y/N ")
            try :
                keep_punctuation = dico_yn[keep_punctuation.upper()]
            except :
                pass

        while keep_numbers not in [True, False] :
            keep_numbers = input("Keep numbers? Y/N ")
            try :
                keep_numbers = dico_yn[keep_numbers.upper()]
            except :
                pass

        self.filter_characters(keep_upper, keep_punctuation, keep_numbers)

        #NETWORK SETUP
        while not 0 < self.resSize :
            print("\nReservoir Size?", end=" ")
            self.resSize = int(input())

        while not 0 < self.trainLen :
            print("Training length? (0-", str(len(self.input_text)), ")", sep="", end=" ")
            self.trainLen = int(input())
        
        while not 0 < self.testLen :
            print("Testing length? (0-", str(len(self.input_text)-self.trainLen), ")", sep="", end=" ")
            self.testLen = int(input())
        
        probamodes = ["filter0", "filter01", "add_min", "max"]
        self.probamode = 0
        while self.probamode not in [1, 2, 3, 4] :
            print("\nProbability mode of calculation?\n 1. Filter negative (ReLu)\n 2. Filter negative and > 1\n 3. Normalization\n 4. Maximum value\n > ", end="")
            self.probamode = int(input())
        self.probamode = probamodes[self.probamode-1]

        self.launches = 0
        while self.launches <= 0:
            self.launches = int(input("\nHow many network launches? "))

        self.nb_words = 0
        while self.nb_words <= 0:
            self.nb_words = int(input("\nHow long do you want the words occurences list to be? "))

    def setup(self) :
        """ Ask the user if (s)he wants to load the predifined parameters (here below),
            or if (s)he wants to go through a series of questions to parameterize it on the fly (in setup_user())."""

        self.predefined_params = 0
        while self.predefined_params not in [1, 2] :
            print("Use predefined parameters?\n 1. Yes\n 2. No\n > ", end="")
            self.predefined_params = int(input())
           
        if self.predefined_params == 1:
            self.type = 1
            self.file = open("text/Shakespeare.txt", "r").read()
            self.mode = 'prediction'
            self.compute_type = "online"
            self.filter_characters(False, True, False) # (keep_upper, keep_punctuation, keep_numbers)
            self.resSize = 10**3
            self.trainLen = 10**5 #200000
            self.testLen = 10**3
            self.probamode = "max"
            self.launches = 1
            self.nb_words = 50
        else:
            setup_user()
            
        # Adapting initLen to the number of neurons inside the reservoir and the time constant
        # xavier's proposition to set the initial warming-up time for the reservoir
        if self.auto_adapt_initLen:
            self.initLen = int(np.floor(self.resSize/float(self.a))) 
            

    def compute_network(self) :
        self.setup()
        if self.type == 1 :      
            self.characters()
        if self.type == 2 :
            self.words()
        self.convert_input()
        self.binary_data()
        for i in range(self.launches) :
            self.current_launch = i
            self.initialization()
            self.compute_spectral_radius()
            if self.compute_type == "offline" :
                self.run_and_record_network()
                self.train_output()
            elif self.compute_type == "online" :
                self.train_online()
            self.test() 
            self.compute_error()
            self.convert_output()
            self.record_output() # save output in a file
            if self.type == 1 :
                self.words_list(existing_words=False)
                self.words_list(existing_words=True)      

    def train_online(self, verbose=False) :
        '''Update network variable by applying a LMS algo'''
        percent = 0.1 # for the progression
        for t in range(self.trainLen):
#        for t in range(self.initLen+self.trainLen+self.testLen):
            self.u = self.data_b[t%len(self.data)] 
            
            ## update equations of reservoir and output units
            self.x = (1-self.a)*self.x + self.a*np.tanh( np.dot(self.Win, np.concatenate((np.array([1]),self.u)).reshape(len(self.input_units)+1,1) ) + np.dot( self.W, self.x ) )
            percent = self.progression(percent, t, self.trainLen)
            
            if t >= self.initLen : # we finished the "warming period", the reservoir is warm enough (i.e. the "intial transient states" are gone ...hopefully), we can start to train now!!!
                # In online mode we do not need to save (i.e. concatenate) all the values of x in X
#                self.X[:,t-self.initLen] = np.concatenate((np.array([1]),self.u,self.x[:,0])).reshape(len(self.input_units)+self.resSize+1,1)[:,0]
                if verbose:
                    print("np.concatenate((np.array([1]), self.u, self.x)).shape ", np.concatenate((np.array([1]), self.u, self.x[:,0])).shape)
                self.y = np.dot(self.Wout, np.concatenate((np.array([1]), self.u, self.x[:,0])))
#                np.concatenate((np.array([1]),self.u,self.x[:,0])).reshape(len(self.input_units)+self.resSize+1,1)[:,0]  
                
                ### compute error and update (reservoir to output) weights
                ##- compute current error
                if verbose:
                    print("self.Wout.shape ", self.Wout.shape)
#                    print("self.x ", self.x)
                    print("self.x.shape ", self.x.shape)
#                    print("self.y ", self.y)
                    print("self.y.shape ", self.y.shape)
                    print("corresponding char: ",self.output_units[np.argmax(self.y)])
#                    print("self.data_b[t+1] ", self.data_b[t+1])
                    print("self.data_b[t+1].shape ", self.data_b[t+1].shape)
                    print("corresponding char: ",self.output_units[np.argmax(self.data_b[t+1])])
                    print("initLen : ", self.initLen, self.trainLen-self.initLen)
                err = self.y - self.data_b[t+1] #.reshape(self.inSize,1) #TODO: should be equivalent to: err = self.y - self.Ytarget[t]
                ##- update reservoir to output weights
                if verbose:
                    print("err.shape ", err.shape)
                    print("err.reshape(self.inSize, 1).shape ", err.reshape(self.inSize, 1).shape)
                    print("np.concatenate((np.array([1]), self.u, self.x[:,0])).reshape(1, self.resSize+self.inSize+1)).shape ", np.concatenate((np.array([1]), self.u, self.x[:,0])).reshape(1, self.resSize+self.inSize+1).shape)
#                    print("np.dot(err.reshape(self.inSize, 1), self.x.reshape(1, self.resSize)).shape ", np.dot(err.reshape(self.inSize, 1), self.x.reshape(1, self.resSize)).shape)
#                self.Wout -= self.learning_rate * np.dot(err.reshape(self.inSize, 1), self.x.reshape(1, self.resSize)) #TODO: check this equation
                self.Wout -= self.learning_rate * np.dot(err.reshape(self.inSize, 1), np.concatenate((np.array([1]), self.u, self.x[:,0])).reshape(1, self.resSize+self.inSize+1))
#                np.concatenate((np.array([1]), self.u, self.x))
                # stop the program if the learning is diverging               
                if np.max(err) > 10**9 or np.max(err)=='nan':
                    raise(Exception, "LMS error is too big (more than 10**42): "+str(err)+". The algorithm is diverging because of a too high learning rate. You should decrease the learning rate !!!")

if __name__ == '__main__':
    nw = Network()
    nw.compute_network()
