from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.optimizers import SGD
from matplotlib import pyplot as plt
#koriste se za klasifikatore
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def Convert(Y):
    Yint=[]
    for i in range (0,len(Y)):
            #Y[i]=Y[i]-1
            if(Y[i]=='A'):
                Y[i]=0;
            elif(Y[i]=='B'):
                Y[i]=1;
            elif(Y[i]=='C'):
                Y[i]=2;
            elif(Y[i]=='D'):
                Y[i]=3;     
            elif(Y[i]=='E'):
                Y[i]=4;
            elif(Y[i]=='F'):
                Y[i]=5;
            elif(Y[i]=='G'):
                Y[i]=6;
            elif(Y[i]=='H'):
                Y[i]=7;
            elif(Y[i]=='I'):
                Y[i]=8;
            elif(Y[i]=='J'):
                Y[i]=9;
            elif(Y[i]=='K'):
                Y[i]=10;
            elif(Y[i]=='L'):
                Y[i]=11; 
            elif(Y[i]=='M'):
                Y[i]=12;
            elif(Y[i]=='N'):
                Y[i]=13;
            elif(Y[i]=='O'):
                Y[i]=14;
            elif(Y[i]=='P'):
                Y[i]=15;     
            elif(Y[i]=='Q'):
                Y[i]=16;
            elif(Y[i]=='R'):
                Y[i]=17;
            elif(Y[i]=='S'):
                Y[i]=18;
            elif(Y[i]=='T'):
                Y[i]=19;
            elif(Y[i]=='U'):
                Y[i]=20;
            elif(Y[i]=='V'):
                Y[i]=21;
            elif(Y[i]=='W'):
                Y[i]=22;
            elif(Y[i]=='X'):
                Y[i]=23; 
            elif(Y[i]=='Y'):
                Y[i]=24;
            elif(Y[i]=="Z"):
                Y[i]=25;
            else:
                Y[i]=Y[i]
            #Y[i]=int(Y[i])
            Yint.append(int(Y[i]))
    return Yint  

def reportStats(TP,TN,FP,FN,klasa):
    print()
    print("klasa " + str(klasa) + ":")
    
    senzitivnost = TP/(TP+FN) #recall
    specificnost = TN/(FP+TN)
    ppv = TP/(TP+FP) #precision
    npv = TN/(TN+FN)
    f1 = 2*(ppv*senzitivnost)/(ppv+senzitivnost)
    acc = (TP+TN)/(TP+FP+TN+FN)
    print("senzitivnost: " + str(round(senzitivnost,2)) + " specificnost: " + str(round(specificnost,2)))
    print("PPV: " + str(round(ppv,2)) + " NPV: " + str(round(npv,2)))
    print("f1 score: " + str(round(f1,2)))
    print("preciznost: " + str(round(acc,2)))

def UporediRezultate(Ytest,Ypredict,method):
    print()
    print("Dobijeni rezultati")
    print(Ypredict)
    print("Trazeni rezultati")
    print(Ytest)

    counter=0;
    
    for i in range (0,len(Ytest)):
        if(Ytest[i]!=Ypredict[i]):
            counter+=1
    #print(counter)
    print("Velicina tening skupa je: " + str(len(Ytrain)))
    print("Velicina test skupa je:" + str(len(Ytest)))
    print("Broj pogodaka: " + str(len(Ytest)-counter))
    print("Broj promasaja: " + str(counter))
    
    title = "confusion matrix : " + method
    
    #classes = np.unique(Ytest)
    #classes=np.unique(Ytrain);
    
    ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25];
    classes=np.unique(ticks);
    fig, ax = plt.subplots()
    cm = metrics.confusion_matrix(Ytest, Ypredict, labels=classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, cbar=False)
    ax.set_xticks(ticks);
    ax.set_yticks(ticks);
    #ax.set_xticks(classes);
    #ax.set_xticks(range(0,len(classes)),classes.index)
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    ax.set_yticklabels(labels=classes, rotation=0)
    plt.show()
    
    #klasa 1
    TP = cm[0,0]
    TN = 0 
    for i in range(1,25):
        for j in range(1,25):
            TN = TN + cm[i,j]
    FP = 0
    for i in range(1,25):
        FP = FP + cm[i,0]
        
    FN = 0
    for i in range(1,25):
        FN = FN + cm[0,i]

    print(TP)
    print(TN)
    print(FP)
    print(FN)
    reportStats(TP,TN,FP,FN,1)
    
    
    
    
def SupportVectorMachine(Xtrain,Ytrain,Xtest,Ytest):
    print()
    print("Support vector machine: ")
    print()
    
    param_grid={'kernel':['linear','rbf'], 'C':[1,5,10,25,100,175,250]} 
    #radi brze za rbf i samo jedan C logicno
    
    svc = SVC()
    svc.fit(Xtrain,Ytrain)
    Ypredict = svc.predict(Xtest)
    method = "Support Vector Machine"
    Ytest=Convert(Ytest)
    Ypredict=Convert(Ypredict)
    UporediRezultate(Ytest,Ypredict,method)
    
    grid = GridSearchCV(svc,param_grid,cv=5)
    grid.fit(Xtrain,Ytrain)
    
    print()
    print("grid rezultat:")
    print(grid.best_params_)
    print()
    
    c=grid.best_params_.get('C')
    kernel=grid.best_params_.get('kernel')
    
    gsvc=SVC(C=c,kernel=kernel)
    gsvc.fit(Xtrain,Ytrain)
    Ypredict=gsvc.predict(Xtest)
    method= "Grid, Support vector machine"
    #Ytest=Convert(Ytest)
    Ypredict=Convert(Ypredict)
    UporediRezultate(Ytest,Ypredict,method)
    
    
    
def RandomForest(Xtrain,Ytrain,Xtest,Ytest):
     print()
    print("Random forest: ")
    print()
    
    param_grid = {'bootstrap':[True,False],'max_depth':[10,20,30,40,50], 'max_features':['auto','sqrt'], 'min_samples_leaf':[1,2,3,4]}
    
    rf = RandomForestClassifier()
    rf.fit(Xtrain,Ytrain)
    
    Ypredict = rf.predict(Xtest)
    method = "Random forest"
    Ytest=Convert(Ytest)
    Ypredict=Convert(Ypredict)
    UporediRezultate(Ytest,Ypredict,method)    
    
    grid = GridSearchCV(rf, param_grid, cv=5)
    grid.fit(Xtrain,Ytrain)
    
    print()
    print("grid rezultat:")
    print(grid.best_params_)
    print()
    
    bootstrap=grid.best_params_.get('bootstrap')
    depth=grid.best_params_.get('max_depth')
    features = grid.best_params_.get('max_features')
    leaf = grid.best_params_.get('min_samples_leaf')
    
    grf = RandomForestClassifier(bootstrap=bootstrap, max_depth=depth, max_features=features, min_samples_leaf=leaf)
    grf.fit(Xtrain,Ytrain)
    Ypredict=grf.predict(Xtest)
    method="Grid, Random forest"
    #Ytest=Convert(Ytest)
    Ypredict=Convert(Ypredict)
    UporediRezultate(Ytest,Ypredict,method)
    
def StabloOdlucivanja(Xtrain,Ytrain,Xtest,Ytest):
    print()
    print("Stablo odlucivanja: ")
    print()
    
    dtc = DecisionTreeClassifier()
    dtc.fit(Xtrain,Ytrain)
    
    Ypredict = dtc.predict(Xtest)
    method = "Stablo odlucivanja"
    Ytest=Convert(Ytest)
    Ypredict=Convert(Ypredict)
    UporediRezultate(Ytest,Ypredict,method)

def K_NearestNeighbors(Xtrain,Ytrain,Xtest,Ytest):
    print()
    print("K-nearest neighbors: ")
    print()
    
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(Xtrain,Ytrain)
    
    Ypredict = knn.predict(Xtest)
    method = "KNearestNeighbors"
    Ytest=Convert(Ytest)
    Ypredict=Convert(Ypredict)
    UporediRezultate(Ytest,Ypredict,method)

def StochasticGradientDescent(Xtrain,Ytrain,Xtest,Ytest):
    print()
    print("Stocastic gradient descent: ")
    print()
    
    #param_grid={'alpha':[0.0001,0.001,0.01,0.1,1,10],'loss':['hinge','log','modified_huber','squared_hinge','perceptron'],
    #            'penalty':['l2','l1','elasticnet'],'max_iter':[1000,1500,2000,2500]}
    
    sgdc = SGDClassifier()
    sgdc.fit(Xtrain,Ytrain)
    
    Ypredict=sgdc.predict(Xtest)
    method = "Stocastic Gradient Descent"
    Ytest=Convert(Ytest)
    Ypredict=Convert(Ypredict)
    UporediRezultate(Ytest,Ypredict,method)
    
    
def NaivniBayes(Xtrain,Ytrain,Xtest,Ytest):
    print()
    print("Naivni bayes: ")
    print()
    
    param_grid = {'var_smoothing':[1e-7,1e-8,1e-9,1e-10,1e-11]}
                  #'priors':[[0.4,0.2,0.4],[0.2,0.4,0.4],[0.4,0.4,0.2],[0.3,0.4,0.3],[0.4,0.3,0.3],[0.3,0.3,0.4],
                  #          [0.35,0.35,0.3],[0.3,0.35,0.35],[0.35,0.3,0.35]]}

    gnb = GaussianNB()
    gnb.fit(Xtrain,Ytrain)
    Ypredict = gnb.predict(Xtest)
    method = "Naivni Bayes"
    Ytest=Convert(Ytest)
    Ypredict=Convert(Ypredict)
    UporediRezultate(Ytest,Ypredict,method)
    
    grid = GridSearchCV(gnb,param_grid,cv=5)
    grid.fit(Xtrain,Ytrain)
    
    print()
    print("grid rezultat:")
    print(grid.best_params_)
    print()
    
    smoothing = grid.best_params_.get('var_smoothing')
    #priority = grid.best_params_.get('priors')
    
    ggnb = GaussianNB(var_smoothing=smoothing)
    ggnb.fit(Xtrain,Ytrain)
    Ypredict = ggnb.predict(Xtest)
    method = "Grid, Naivni Bayes"
    #Ytest=Convert(Ytest)
    Ypredict=Convert(Ypredict)
    UporediRezultate(Ytest,Ypredict,method)
    
    
def NeuralNetworkCC(neurons=26,act='relu', Nepochs=100, NBatchSize=200, optAlg='adam', enable=0, learnRate=0.01, momentum=0.9):    
    if(optAlg=='SGD'):
        opt = SGD(lr=learnRate, momentum=momentum)
    else: opt='adam'
    model = Sequential()
    model.add(Dense(neurons, input_dim=16, activation=act))
    #model.add(Dense(neurons, activation=act))
    model.add(Dense(neurons, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])    
    #sparse_categorical_crossentropy se koristi zato sto je svaki zeljeni izlaz ceo broj i postoji vise od 2 izlazne klase

    if(enable==1):
        print("training neural network")
        #data=loadtxt('letter.csv', delimiter=',',skiprows=1, dtype=(int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,"U1"))
        #data=np.genfromtxt('letter1.csv', delimiter=',', dtype=(int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,"U1"))
        #data=np.genfromtxt('letter1.csv', delimiter=',', dtype='unicode')
        #data.reshape(20000,17)
        #data=pd.read_csv('letter1.csv', sep='\t')
        #jedva proradila ovo gore ne funkcionise sa mojim podacima
        col_list=["i1","i2","i3","i4","i5","i6","i7","i8","i9","i10","i11","i12","i13","i14","i15","i16","o1"]
        data = pd.read_csv("letter.csv",usecols=col_list)
        data = data.sample(frac=1)
        testS = 4000
        a=20000
        trainS = a-testS
        
        Xcols = train[["i1","i2","i3","i4","i5","i6","i7","i8","i9","i10","i11","i12","i13","i14","i15","i16"]]
        Ycols = train[["o1"]]
    
        X = Xcols[0:trainS-1]
        X = X.to_numpy()
        Y = Ycols[0:trainS-1]
        Xtest = Xcols[trainS:a]
        Xtest = Xtest.to_numpy()
        Ytest = Ycols[trainS:a]

        Y=np.ravel(Y)
        Ytest = np.ravel(Ytest)
       
        #np.random.shuffle(data)
        #X=data[0:16000,0:16]
        #Y=data[0:16000,16]
        #Xtest = data[16001:20000,0:16]
        #Ytest = data[16001:20000,16]
        YtestSave = []
        Yint=[]
        Ytestint=[]
       
        #slova moras da konvertujes u brojeve
        for i in range (0,len(Y)):
            #Y[i]=Y[i]-1
            if(Y[i]=='A'):
                Y[i]=0;
            elif(Y[i]=='B'):
                Y[i]=1;
            elif(Y[i]=='C'):
                Y[i]=2;
            elif(Y[i]=='D'):
                Y[i]=3;     
            elif(Y[i]=='E'):
                Y[i]=4;
            elif(Y[i]=='F'):
                Y[i]=5;
            elif(Y[i]=='G'):
                Y[i]=6;
            elif(Y[i]=='H'):
                Y[i]=7;
            elif(Y[i]=='I'):
                Y[i]=8;
            elif(Y[i]=='J'):
                Y[i]=9;
            elif(Y[i]=='K'):
                Y[i]=10;
            elif(Y[i]=='L'):
                Y[i]=11; 
            elif(Y[i]=='M'):
                Y[i]=12;
            elif(Y[i]=='N'):
                Y[i]=13;
            elif(Y[i]=='O'):
                Y[i]=14;
            elif(Y[i]=='P'):
                Y[i]=15;     
            elif(Y[i]=='Q'):
                Y[i]=16;
            elif(Y[i]=='R'):
                Y[i]=17;
            elif(Y[i]=='S'):
                Y[i]=18;
            elif(Y[i]=='T'):
                Y[i]=19;
            elif(Y[i]=='U'):
                Y[i]=20;
            elif(Y[i]=='V'):
                Y[i]=21;
            elif(Y[i]=='W'):
                Y[i]=22;
            elif(Y[i]=='X'):
                Y[i]=23; 
            elif(Y[i]=='Y'):
                Y[i]=24;
            elif(Y[i]=="Z"):
                Y[i]=25;
            #Y[i]=int(Y[i])
            Yint.append(int(Y[i]))
        Yint=np.array(Yint)
        Yint=Yint.reshape(15999,1)
      
        for i in range (0,len(Ytest)):
            #Y[i]=Y[i]-1
            if(Ytest[i]=='A'):
                Ytest[i]=0;
            elif(Ytest[i]=='B'):
                Ytest[i]=1;
            elif(Ytest[i]=='C'):
                Ytest[i]=2;
            elif(Ytest[i]=='D'):
                Ytest[i]=3;     
            elif(Ytest[i]=='E'):
                Ytest[i]=4;
            elif(Ytest[i]=='F'):
                Ytest[i]=5;
            elif(Ytest[i]=='G'):
                Ytest[i]=6;
            elif(Ytest[i]=='H'):
                Ytest[i]=7;
            elif(Ytest[i]=='I'):
                Ytest[i]=8;
            elif(Ytest[i]=='J'):
                Ytest[i]=9;
            elif(Ytest[i]=='K'):
                Ytest[i]=10;
            elif(Ytest[i]=='L'):
                Ytest[i]=11; 
            elif(Ytest[i]=='M'):
                Ytest[i]=12;
            elif(Ytest[i]=='N'):
                Ytest[i]=13;
            elif(Ytest[i]=='O'):
                Ytest[i]=14;
            elif(Ytest[i]=='P'):
                Ytest[i]=15;     
            elif(Ytest[i]=='Q'):
                Ytest[i]=16;
            elif(Ytest[i]=='R'):
                Ytest[i]=17;
            elif(Ytest[i]=='S'):
                Ytest[i]=18;
            elif(Ytest[i]=='T'):
                Ytest[i]=19;
            elif(Ytest[i]=='U'):
                Ytest[i]=20;
            elif(Ytest[i]=='V'):
                Ytest[i]=21;
            elif(Ytest[i]=='W'):
                Ytest[i]=22;
            elif(Ytest[i]=='X'):
                Ytest[i]=23; 
            elif(Ytest[i]=='Y'):
                Ytest[i]=24;
            elif(Ytest[i]=='Z'):
                Ytest[i]=25;
            Ytestint.append(int(Ytest[i]))
        Ytestint=np.array(Ytestint)
        Ytestint=Ytestint.reshape(4000,1)
      
            #Ytest[i]=int(Ytest[i])
                            
                
        #for i in range(0,len(Ytest)):
         #   if(Ytest[i]==1):YtestSave.append(1)
          #  elif(Ytest[i]==2):YtestSave.append(2)
           # else:YtestSave.append(3)
            #Ytest[i]=Ytest[i]-1
            
        #Yin=to_categorical(Y)   #ove dve linije se koriste ukoliko je loss funkcija podesena kao categorical_crossentropy
        #Yt=to_categorical(Ytest)
        model.fit(X,Yint,epochs=Nepochs,batch_size=NBatchSize, verbose=1)
        print()
        print("training data:")
        _ , accuracy = model.evaluate(X,Yint)
        print(str(round(accuracy*100,2))+ "%")
        print("testing data:")
        _ , accuracy = model.evaluate(Xtest,Ytestint)
        print(str(round(accuracy*100,2)) + "%")
        #izlazne klase su promenjene iz 1 2 3 u 0 1 2 zbog kategorizacije
        Ypredict = model.predict(Xtest)
        
        prediction =[]
        for i in range (0,len(Ypredict)):
            niz = Ypredict[i].tolist()
            value = niz.index(max(niz))
            prediction.append(value)
        
        method="neuronska mreza"
        UporediRezultate(Ytestint,prediction,method)
    return model
    
def vizuelizacija(x,y,output,xlabel,ylabel):
    x1=[]
    x2=[]
    x3=[]
    x4=[]
    x5=[]
    x6=[]
    x7=[]
    x8=[]
    x9=[]
    x10=[]
    x11=[]
    x12=[]
    x13=[]
    x14=[]
    x15=[]
    x16=[]
    x17=[]
    x18=[]
    x19=[]
    x20=[]
    x21=[]
    x22=[]
    x23=[]
    x24=[]
    x25=[]
    x26=[]
    y1=[]
    y2=[]
    y3=[]
    y4=[]
    y5=[]
    y6=[]
    y7=[]
    y8=[]
    y9=[]
    y10=[]
    y11=[]
    y12=[]
    y13=[]
    y14=[]
    y15=[]
    y16=[]
    y17=[]
    y18=[]
    y19=[]
    y20=[]
    y21=[]
    y22=[]
    y23=[]
    y24=[]
    y25=[]
    y26=[]
    #definisi do x26 i y26
    
    for i in range(0,len(output)):
        if(output[i]=="A"): 
            x1.append(x[i])
            y1.append(y[i])
        elif (output[i]=="B"):
            x2.append(x[i])
            y2.append(y[i])
        elif (output[i]=="C"):
            x3.append(x[i])
            y3.append(y[i])
        elif (output[i]=="D"):
            x4.append(x[i])
            y4.append(y[i])
        elif (output[i]=="E"):
            x5.append(x[i])
            y5.append(y[i])
        elif (output[i]=="F"):
            x6.append(x[i])
            y6.append(y[i])
        elif (output[i]=="G"):
            x7.append(x[i])
            y7.append(y[i])
        elif (output[i]=="H"):
            x8.append(x[i])
            y8.append(y[i])
        elif (output[i]=="I"):
            x9.append(x[i])
            y9.append(y[i])
        elif (output[i]=="J"):
            x10.append(x[i])
            y10.append(y[i])
        elif (output[i]=="K"):
            x11.append(x[i])
            y11.append(y[i])
        elif (output[i]=="L"):
            x12.append(x[i])
            y12.append(y[i])
        elif (output[i]=="M"):
            x13.append(x[i])
            y13.append(y[i])
        elif (output[i]=="N"):
            x14.append(x[i])
            y14.append(y[i])
        elif (output[i]=="O"):
            x15.append(x[i])
            y15.append(y[i])
        elif (output[i]=="P"):
            x16.append(x[i])
            y16.append(y[i])
        elif (output[i]=="Q"):
            x17.append(x[i])
            y17.append(y[i])
        elif (output[i]=="R"):
            x18.append(x[i])
            y18.append(y[i])
        elif (output[i]=="S"):
            x19.append(x[i])
            y19.append(y[i])
        elif (output[i]=="T"):
            x20.append(x[i])
            y20.append(y[i])
        elif (output[i]=="U"):
            x21.append(x[i])
            y21.append(y[i])
        elif (output[i]=="V"):
            x22.append(x[i])
            y22.append(y[i])
        elif (output[i]=="W"):
            x23.append(x[i])
            y23.append(y[i])
        elif (output[i]=="X"):
            x24.append(x[i])
            y24.append(y[i])
        elif (output[i]=="Y"):
            x25.append(x[i])
            y25.append(y[i])
        elif (output[i]=="Z"):
            x26.append(x[i])
            y26.append(y[i])
        
    #dodati slucajeve za sva slova od A do Z kao gore         
    a1 = plt.scatter(x1,y1,label='klasa1')
    a2 = plt.scatter(x2,y2,label='klasa2')
    a3 = plt.scatter(x3,y3,label='klasa3')
    a4 = plt.scatter(x4,y4,label='klasa4')
    a5 = plt.scatter(x5,y5,label='klasa5')
    a6 = plt.scatter(x6,y6,label='klasa6')
    a7 = plt.scatter(x7,y7,label='klasa7')
    a8 = plt.scatter(x8,y8,label='klasa8')
    a9 = plt.scatter(x9,y9,label='klasa9')
    a10 = plt.scatter(x10,y10,label='klasa10')
    a11 = plt.scatter(x11,y11,label='klasa11')
    a12 = plt.scatter(x12,y12,label='klasa12')
    a13 = plt.scatter(x13,y13,label='klasa13')
    a14 = plt.scatter(x14,y14,label='klasa14')
    a15 = plt.scatter(x15,y15,label='klasa15')
    a16 = plt.scatter(x16,y16,label='klasa16')
    a17 = plt.scatter(x17,y17,label='klasa17')
    a18 = plt.scatter(x18,y18,label='klasa18')
    a19 = plt.scatter(x19,y19,label='klasa19')
    a20 = plt.scatter(x20,y20,label='klasa20')
    a21 = plt.scatter(x21,y21,label='klasa21')
    a22 = plt.scatter(x22,y22,label='klasa22')
    a23 = plt.scatter(x23,y23,label='klasa23')
    a24 = plt.scatter(x24,y24,label='klasa24')
    a25 = plt.scatter(x25,y25,label='klasa25')
    a26 = plt.scatter(x26,y26,label='klasa26')
    
    #definisati za a26 
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend((a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26),('klasa1','klasa2','klasa3','klasa4','klasa5','klasa6','klasa7','klasa8','klasa9','klasa10','klasa11','klasa12','klasa13','klasa14','klasa15','klasa16','klasa17','klasa18','klasa19','klasa20','klasa21','klasa22','klasa23','klasa24','klasa25','klasa26'))
    plt.legend(bbox_to_anchor=(1.28,0.9),loc='center right')
    #dodati u legendu
    plt.show()
    

#main rutina
col_list=["i1","i2","i3","i4","i5","i6","i7","i8","i9","i10","i11","i12","i13","i14","i15","i16","o1"]
read = pd.read_csv("letter.csv",usecols=col_list)
a = len(read)
train = read.sample(frac=1)

#print(read)
#print(train)

testS = 4000
trainS = a-testS

Xcols = train[["i1","i2","i3","i4","i5","i6","i7","i8","i9","i10","i11","i12","i13","i14","i15","i16"]]
Ycols = train[["o1"]]

Y=Ycols[0:20000];

Xtrain = Xcols[0:trainS-1]
Ytrain = Ycols[0:trainS-1]
Xtest = Xcols[trainS:a]
Ytest = Ycols[trainS:a]

Ytrain=np.ravel(Ytrain)
Ytest = np.ravel(Ytest)

#NaivniBayes(Xtrain,Ytrain,Xtest,Ytest) #los je zato sto podaci nisu uslovno nezavisni 
#StochasticGradientDescent(Xtrain,Ytrain,Xtest,Ytest)
#K_NearestNeighbors(Xtrain, Ytrain, Xtest, Ytest)
#StabloOdlucivanja(Xtrain, Ytrain, Xtest, Ytest)
#RandomForest(Xtrain,Ytrain,Xtest,Ytest)
#SupportVectorMachine(Xtrain,Ytrain,Xtest,Ytest)

#odkomentarisati onaj koji koristis

NeuralNetworkCC(enable=1) 

#ovaj deo sluzi samo za vizuelizaciju podataka
ulaz1 = train[["i1"]].to_numpy() 
ulaz2 = train[["i2"]].to_numpy()
ulaz3 = train[["i3"]].to_numpy()
ulaz4 = train[["i4"]].to_numpy()
ulaz5 = train[["i5"]].to_numpy()
ulaz6 = train[["i6"]].to_numpy() 
ulaz7 = train[["i7"]].to_numpy()
ulaz8 = train[["i8"]].to_numpy()
ulaz9 = train[["i9"]].to_numpy()
ulaz10 = train[["i10"]].to_numpy()
ulaz11 = train[["i11"]].to_numpy() 
ulaz12 = train[["i12"]].to_numpy()
ulaz13 = train[["i13"]].to_numpy()
ulaz14 = train[["i14"]].to_numpy()
ulaz15 = train[["i15"]].to_numpy()
ulaz16 = train[["i16"]].to_numpy()

izlaz = train[["o1"]].to_numpy()  

vizuelizacija(ulaz1, ulaz2, izlaz, "ulaz1", "ulaz2")
vizuelizacija(ulaz3, ulaz6, izlaz, "ulaz3", "ulaz6")
vizuelizacija(ulaz14, ulaz16, izlaz, "ulaz14", "ulaz16")
