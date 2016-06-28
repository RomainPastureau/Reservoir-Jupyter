import numpy as np
import matplotlib.pyplot as plt
import re
from scipy import linalg
from ipywidgets import *
from IPython.display import *
from collections import Counter

class Network(object):

    def __init__(self, trainLen=0, testLen=0, initLen=100) :
        """Initialization of the network."""
        self.initLen = initLen
        self.trainLen = trainLen
        self.testLen = testLen
        self.resSize = 0
        self.a = 0.3
        self.spectral_radius = 0.25
        self.input_scaling = 1.
        self.reg =  1e-8
        self.mode = 'prediction'
        self.seed = None #42
        self.set_seed()
        self.file = open("shakespeare_input.txt", "r").read()

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

    def characters_setup(self) :
        """Only in advanced mode.
        Allows the user to configure the character filter."""

        keep_upper, keep_punctuation, keep_numbers = "", "", ""

        dico_yn = {"Y" : True, "O" : True, "T" : True,
                   "N" : False, "F" : False}

        while keep_upper not in [True, False] :
            keep_upper = input("Keep upper case letters? Y/N ")
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
        
    def filter_characters(self, keep_upper=True, keep_punctuation=True, keep_numbers=True) :
        """Filters the characters of the input :
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
        """Creates the input units according to all the different characters
        in the text, and prints the corresponding list and its length."""
        self.input_units = list(set(self.input_text))
        print("Existing characters in the text :", sorted(self.input_units),"\nNumber of different characters :", len(self.input_units), "\n")

    def words(self) :
        """Creates the input units according to all the different words
        in the text, and prints the number of different words."""
        print(type(self.input_text))
        self.input_text = re.findall(r"[\w']+|[.,!?;]", "".join(self.input_text))
        print(len(self.input_text))
        self.input_units = list(set(self.input_text))
        print(len(self.input_units))
        print(self.input_units[0:5])
        #print("Existing words in the text :", sorted(self.input_units),"\nNumber of different words :", len(self.input_units), "\n")
        print("Number of different words :", len(self.input_units), "\n")

    def convert_input(self) :
        print("Converting input into ID numbers...", end=" ")
        percent = 0.1
        self.data = np.array([])
        for i in range(len(self.input_text)) :
            percent = self.progression(percent, i, len(self.input_text))
            np.append(self.data, self.input_units.index(self.input_text[i]))
        #self.data = np.array([self.input_units.index(i) for i in self.input_text])
        self.inSize = self.outSize = len(self.input_units)
        print("done.")

    def plot_inputs(self, f) :
        print("Plotting characters ID throughout the first", str(f), "characters of the text.")
        plt.figure(1).clear()
        limit = np.arange(int(f))   
        plt.plot(limit,self.data[0:int(f)], 'ro')
        plt.title('Characters throughout the text')
        plt.show()

    def binary_data(self) :
        print("Creating the input binary matrix...", end=" ")
        self.data_b = np.zeros((len(self.input_text), len(self.input_units)))
        for i in range(len(self.data)) :
            self.data_b[i][self.data[i]] = 1
        print("done.\n")

    def type_setup(self) :

        self.type = 0
        while self.type not in [1, 2, 3, 4] :
            print("Type of input/output?\n 1. Characters\n 2. Words\n 3. Pixels (not working)\n 4. Picture (not working)\n > ", end="")
            self.type = int(input())

    def network_setup(self) :
        
        while not 0 < self.resSize :
            print("Reservoir Size?", end=" ")
            self.resSize = int(input())

        while not 0 < self.trainLen < len(self.input_text) :
            print("Training length? (0-", str(len(self.input_text)), ")", sep="", end=" ")
            self.trainLen = int(input())
        
        while not 0 < self.testLen < len(self.input_text)-self.trainLen :
            print("Testing length? (0-", str(len(self.input_text)-self.trainLen), ")", sep="", end=" ")
            self.testLen = int(input())
        
        probamodes = ["filter0", "filter01", "add_min", "max"]
        self.probamode = 0
        while self.probamode not in [1, 2, 3, 4] :
            print("Probability mode of calculation?\n 1. Filter negative (ReLu)\n 2. Filter negative and > 1\n 3. Normalization\n 4. Maximum value\n > ", end="")
            self.probamode = int(input())
        self.probamode = probamodes[self.probamode-1]

        self.launches = 0
        while 0 < self.launches :
            print("How many launches of the network? ")
            self.lauches = int(input())
        
    def initialization(self) :
        print("\nInitializing the network matrices...", end=" ")
        self.Win = (np.random.rand(self.resSize,1+self.inSize)-0.5) * self.input_scaling
        self.W = np.random.rand(self.resSize,self.resSize)-0.5 
        self.X = np.zeros((1+self.inSize+self.resSize,self.trainLen-self.initLen))
        self.Ytarget = self.data_b[self.initLen+1:self.trainLen+1].T
        self.x = np.zeros((self.resSize,1))  
        print("done.")

    def compute_spectral_radius(self):
        print('Computing spectral radius...',end=" ")
        rhoW = max(abs(linalg.eig(self.W)[0]))
        print('done.')
        self.W *= self.spectral_radius / rhoW

    def train_input(self) :
        print('Training the input...', end=" ")
        percent = 0.1
        for t in range(self.trainLen):
            percent = self.progression(percent, t, self.trainLen)
            self.u = self.data_b[t]
            self.x = (1-self.a)*self.x + self.a*np.tanh( np.dot(self.Win, np.concatenate((np.array([1]),self.u)).reshape(len(self.input_units)+1,1) ) + np.dot( self.W, self.x ) )
            if t >= self.initLen :
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
        self.u = self.data_b[self.trainLen]
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
                self.u[self.data[self.trainLen+t+1]] = 1
            else:
                raise(Exception, "ERROR: 'mode' was not set correctly.")
        print('done.\n')

    def compute_error(self) :
        print("Computing the error...", end=" ")
        errorLen = 500
        mse = sum( np.square( self.data[self.trainLen+1:self.trainLen+errorLen+1] - self.Y[0,0:errorLen] ) ) / errorLen
        print(self.data[self.trainLen+1:self.trainLen+errorLen+1][0])
        print(self.Y[0,0:errorLen][0])
        print('MSE = ' + str( mse ))

    def probabilities(self, i) :
        if self.probamode == "filter0" :
            proba_weights = abs((self.Y.T[i] > 0)*self.Y.T[i])
        elif self.probamode == "filter01" :
            proba_weights = abs((self.Y.T[i] > 0)*self.Y.T[i])
            proba_weights = proba_weights-((proba_weights > 1)*proba_weights) + (proba_weights > 1)*1
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
                self.output_text += np.random.choice(self.input, p=proba_weights)
            else :
                self.output_text += self.input[np.argmax(proba_weights)]
        print("done.")

    def record_output(self) :
        print("Saving the output as a text file.")
        record_file = open("output.txt", "w")
        record_file.write(self.output_text)
        record_file.close()

    def words_list(self, existing_words=True, language="EN", nb_words=20) :
        
        if existing_words == True :
            if language == "EN" : words_dict = open("words_list_EN.txt", "r").read()
            elif language == "FR" : words_dict = open("words_list_FR.txt", "r").read()
            words_dict = words_dict.split()

        alphabet = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ")
        self.allwords = "".join([i for i in self.output_text if i in alphabet])
        self.allwords = self.allwords.lower().split()
        words_occurences = Counter(self.allwords)
        
        if existing_words == False :
            print("\nMost common words (real or not) in the generated text", end="\n\n")
            longest_size = len(sorted(list(words_occurences.most_common(nb_words)), key=len)[-1])
            print("| Word", end=" "*(max(longest_size,4)-3))
            print("| Occurences ")
            print("-"*(longest_size+17))
            words_occurences = words_occurences.most_common(nb_words)
            
            for i in range(nb_words) :
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
            
            while i < nb_words :
                if words_occurences[j][0] in words_in_dictionary :
                    real_words_occurences.append(words_occurences[j])
                    i += 1
                j += 1
                if j == len(words_occurences) :
                    break

            longest_size = len(sorted(real_words_occurences, key=len)[-1])
            print("| Word", end=" "*(max(longest_size,4)-3))
            print("| Occurences ")
            print("-"*(longest_size+17))
            
            for k in range(i) :
                w = str(real_words_occurences[k][0])
                n = str(real_words_occurences[k][1])
                print("| " + w + " "*(max(longest_size,3)-len(w)+2) + "| " + n)
            print("-"*(longest_size+17))
            
    def search_word(self, word) :
        pass

    def progression(self, percent, i, total) :
        if i == 0 :
            print("Progress :", end= " ")
            percent = 0.1
        elif (i/total) > percent :
            print(round(percent*100), end="")
            print("%", end=" ")
            percent += 0.1
        if total-i == 1 :
            print("100%")

        return(percent)

    def compute_network(self) :
        self.type_setup()
        if self.type == 1 :
            self.characters_setup()
            self.characters()
        if self.type == 2 :
            self.characters_setup()
            self.words()
        self.convert_input()
        #self.plot_inputs(500)
        self.binary_data()
        self.network_setup()
        self.initialization()
        self.compute_spectral_radius()
        self.train_input()
        self.train_output()
        self.test() 
        self.compute_error()
        self.convert_output()
        self.record_output()
        self.words_list(existing_words=False)
        self.words_list(existing_words=True)

if __name__ == '__main__':
    nw = Network()
    nw.compute_network()
    print("_"*50)
    print("Output text :")
    print(nw.output_text)
