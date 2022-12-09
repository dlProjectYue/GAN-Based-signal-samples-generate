import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
class confusMatrix(object):
    def __init__(self,num_classes:int,labels:int):
        self.matrix=np.zeros((num_classes,num_classes))
        self.num_classes=num_classes
        self.labels=labels
    def update(self,preds,labels):
        for p,t in zip(preds,labels):
            self.matrix[p.astype(int),t.astype(int)]+=1
    def summary(self):#计算指标函数？
        sum_TP=0
        n=np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP+=self.matrix[i,i]#混淆矩阵对角线的元素之和，也就是分类正确的数量
        acc=sum_TP/n
        print("the model accuracy is ",acc)

        sum_po=0
        sum_pe=0
        for i in range(len(self.matrix[0])):
            sum_po+=self.matrix[i][i]
            row=np.sum(self.matrix[i,:])
            col=np.sum(self.matrix[:,i])
            sum_pe+=row*col
        po=sum_po/n
        pe=sum_pe/(n*n)
        kappa=round((po-pe)/(1-pe),3)
        table=PrettyTable()
        table.field_names=["","Precision","Recall","Specificity"]
        for i in range(self.num_classes):
            TP=self.matrix[i,i]
            FP=np.sum(self.matrix[i,:])-TP
            FN=np.sum(self.matrix[:,i])-TP
            TN=np.sum(self.matrix)-TP-FP-FN
            Precision=round(TP/(TP+FP),3) if TP+FP!=0 else 0
            Recall=round(TP/(TP+FN),3) if TP+FN!=0 else 0
            Specificity=round(TN/(TN+FP),3)if TN+FP!=0 else 0
            table.add_row([self.labels[i],Precision,Recall,Specificity])
        print(table)
        return str(acc)
    def plot(self):
        matrix=self.matrix
        print(matrix)
        plt.imshow(matrix,cmap=plt.cm.Blues)
        plt.xticks(range(self.num_classes),range(self.labels),rotation=45)
        plt.yticks(range(self.num_classes),range(self.labels))
        plt.colorbar(),
        plt.xlabel('True Labels')
        plt.title('Confusion matrix(acc=' +self.summary()+')')

        thresh=matrix.max()/2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                info=int(matrix[y,x])
                plt.text(x,y,info,verticalalignment='center',horizontalalignment='center',color="white" if info >thresh else "black")
        plt.tight_layout()
        plt.show()
