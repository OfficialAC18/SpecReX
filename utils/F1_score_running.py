class F1_score_running:
    def __init__(self,classes = None):
        self.tp = [] if not classes else [0]*classes
        self.fp = [] if not classes else [0]*classes
        self.fn = [] if not classes else [0]*classes
        self.counts = [] if not classes else [0]*classes
        self.classes = classes
        
    def log(self, predicted, labels):
        #Get the total number of classes that exist from the data
        if not self.classes:
            self.classes = len(np.unique(labels))
            self.fp = [0] * self.classes
            self.fn = [0] * self.classes
            self.tp = [0] * self.classes
            self.counts = [0] * self.classes
        
        #We need to calculate the FP and FN for each class 
        for idx in range(self.classes):
            #In a copy of the predicted and label arrays
            #Change all the indices with val = i to 1, and everything else to 0
            pred_idx = np.array(list(map(lambda x: 1 if x == idx else 0, predicted)))
            label_idx = np.array(list(map(lambda x: 1 if x == idx else 0, labels)))
            
            #Count the number of 1s and -1s and the support for the class
            unique, counts = np.unique(pred_idx - label_idx, return_counts = True)
            counts = dict(zip(unique, counts))
            self.fp[idx] += counts[1] if 1 in counts.keys() else 0
            self.fn[idx] += counts[-1] if -1 in counts.keys() else 0
            self.tp[idx] += np.logical_and(pred_idx, label_idx).sum().item()
            self.counts[idx] += np.sum(len(np.argwhere(label_idx)))
    
    def calc(self,average = None):
        #Convert to numpy array for vectorisation
        self.tp = np.array(self.tp,dtype=np.float32)
        self.fp = np.array(self.fp,dtype=np.float32)
        self.fn = np.array(self.fn,dtype=np.float32)
        ratio = np.array(self.counts)/sum(self.counts)

        #Calculate Precision and Recall
        self.precision = self.tp/(self.tp + self.fp)
        self.recall = self.tp/(self.tp + self.fn)
        
        #Calculate the F1-Score
        f1 = 2*self.tp
        f1 /= 2*self.tp + self.fn + self.fp

        if not self.average:
            return f1
        elif self.average == 'binary':
            return f1[-1]
        elif self.average == 'macro':
            return np.mean(f1)
        elif self.average == 'weighted':
            return sum(ratio * f1)
        return f1