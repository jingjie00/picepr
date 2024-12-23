

class PrEvaluation():

    def __init__(self,name="PrEvaluation"):
        self.name = name
        self.preds_dict = {}
        self.labels_dict = {}
        self.confusion_matrix = {}


    def push(self, preds, labels):
        def verify_input(preds,labels):
            preds_flag = isinstance(preds, list) and all(isinstance(sublist, list) and (all(isinstance(item, float)or isinstance(item, int)) for item in sublist) for sublist in preds)
            labels_flag = isinstance(preds, list) and all(isinstance(sublist, list) and (all(isinstance(item, float)or isinstance(item, int)) for item in sublist) for sublist in preds)
            if preds_flag and labels_flag:
                if (len(preds) == len(labels)):
                    if len(self.preds_dict) == 0:
                        if len(preds[0])==len(labels[0]):
                            return True
                        else:
                            raise ValueError("Error, should same")
                    elif  (len(preds)==len(self.preds_dict)):
                        return True
                    else:
                        raise ValueError("The length of preds and labels should be the same a")
                raise ValueError("The length of preds and labels should be the same b")
            raise ValueError("preds and labels should be lists of lists of integers or floats")

        if verify_input(preds, labels):
            if len(self.preds_dict) == 0:
                for i in range(len(preds)):
                    self.preds_dict[i] = preds[i]
            else:
                for i in range(len(preds)):
                        self.preds_dict[i].extend(preds[i])
                    
            if len(self.labels_dict) == 0:
                for i in range(len(labels)):
                    self.labels_dict[i] = labels[i]
            else:
                for i in range(len(labels)):
                    self.labels_dict[i].extend(labels[i])


    def get_confusion_matrix(self):
        def get_single_confusion_matrix(preds, labels):
            true_positive = 0
            false_positive = 0
            true_negative = 0
            false_negative = 0
            for pred, label in zip(preds, labels):
                if pred == 1 and label == 1:
                    true_positive += 1
                elif pred == 1 and label == 0:
                    false_positive += 1
                elif pred == 0 and label == 0:
                    true_negative += 1
                elif pred == 0 and label == 1:
                    false_negative += 1
            
            single_confusion_matrix = (true_positive, false_positive, true_negative, false_negative)

            return single_confusion_matrix

        # loop the keys 
        for a,b in zip(self.preds_dict, self.labels_dict):
            if a != b:
                raise ValueError("The keys of preds and labels should be the same")
            
            preds = self.preds_dict[a]
            labels = self.labels_dict[b]

            single_confusion_matrix = get_single_confusion_matrix(preds, labels)
            self.confusion_matrix[a] = single_confusion_matrix



    def get_performance_metrics(self):
        def get_single_performance(cm):
            true_positive, false_positive, true_negative, false_negative = cm
            negative = true_negative + false_positive
            positive = true_positive + false_negative

            if negative == 0 or positive == 0:
                print("There is no positive or negative in the dataset")
                return -1,-1,-1,cm

            sensitivity = true_positive / positive
            specificity = true_negative / negative
            
            regular_accuracy = (true_negative + true_positive)/(true_positive + false_negative + true_negative + false_positive)
            balanced_accuracy = (sensitivity + specificity) / 2

            f1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative)

            # assert positive == 0 or negative == 0, "There is no positive or negative in the dataset"
            return balanced_accuracy, regular_accuracy, f1, cm

        self.get_confusion_matrix()

        performance = {}
        for key in self.confusion_matrix.keys():
            ba,ra, f1, cm = get_single_performance(self.confusion_matrix[key])
            performance[key] = (ba, ra, f1, cm)

        return performance
    
    def print_performance(self):
        performance = self.get_performance_metrics()
        for p in performance:
            ba,ra,f1,cm = performance[p]
            print(f"{self.name}\t   BA: {ba:.4f}, RA: {ra:.4f}, F1: {f1:.4f}, CM: {cm}")

def print_performance(performance):
    for p in performance:
        ba,ra,f1,cm = performance[p]
        print(f"\t   BA: {ba:.4f}, RA: {ra:.4f}, F1: {f1:.4f}, CM: {cm}")
    return

if __name__ == "__main__":
    preds1 = [[1,1,1,1,0,0,0], [1,1,1,0,0,0,0],[1,1,1,1,0,0,0]]
    labels1 = [[1,1,1,1,0,0,0], [1,1,1,0,0,0,0],[1,1,1,1,0,0,0]]

    preds2 = [[1,1,1,1,0,0,0], [1,1,1,0,0,0,0],[1,1,1,1,0,0,0]]
    labels2 = [[0,0,0,0,1,1,1], [0,0,0,1,1,1,1],[1,1,1,1,0,0,0]]



    # Instantiate the class and test
    evaluator = PrEvaluation()
    evaluator.push(preds1, labels1)
    evaluator.push(preds2, labels2)
    performance = evaluator.get_performance_metrics()
    print(print_performance(performance))
