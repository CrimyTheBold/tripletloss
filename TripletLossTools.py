import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import wandb
from GenericTools import *

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform

from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

#################################### To build model
class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)
    
    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square(anchor-positive), axis=-1)
        n_dist = K.sum(K.square(anchor-negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)
    
    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss
    
def build_model3(input_shape, network, margin=0.2):
    '''
    Define the Keras Model for training 
        Input : 
            input_shape : shape of input images
            network : Neural network to train outputing embeddings
            margin : minimal distance between Anchor-Positive and Anchor-Negative for the lossfunction (alpha)
    
    '''
     # Define the tensors for the three input images
    anchor_input = Input(input_shape, name="anchor_input")
    positive_input = Input(input_shape, name="positive_input")
    negative_input = Input(input_shape, name="negative_input") 
    
    # Generate the encodings (feature vectors) for the three images
    encoded_a = network(anchor_input)
    encoded_p = network(positive_input)
    encoded_n = network(negative_input)
    
    #TripletLoss Layer
    loss_layer = TripletLossLayer(alpha=margin,name='3xLoss')([encoded_a,encoded_p,encoded_n])
    
    # Connect the inputs with the outputs
    network_train = Model(inputs=[anchor_input,positive_input,negative_input],outputs=loss_layer)
    
    # return the model
    return network_train

####################### EVALUATION
def compute_l2_dist(a,b):
    return np.sum(np.square(a-b))

def compute_probs(network,X,Y):
    '''
    Input
        network : current NN to compute embeddings
        X : tensor of shape (m,w,h,1) containing pics to evaluate
        Y : tensor of shape (m,) containing true class
        
    Returns
        probs : array of shape (m,m) containing distances
    
    '''
    m = X.shape[0]
    nbevaluation = int(m*(m-1)/2)
    probs = np.zeros((nbevaluation))
    y = np.zeros((nbevaluation))
    
    #Compute all embeddings for all pics with current network
    embeddings = network.predict(X)
    
    size_embedding = embeddings.shape[1]
    
    #For each pics of our dataset
    k = 0
    for i in range(m):
            #Against all other images
            for j in range(i+1,m):
                #compute the probability of being the right decision : it should be 1 for right class, 0 for all other classes
                probs[k] = -compute_l2_dist(embeddings[i,:],embeddings[j,:])
                if (Y[i]==Y[j]):
                    y[k] = 1
                    #print("{3}:{0} vs {1} : {2}\tSAME".format(i,j,probs[k],k))
                else:
                    y[k] = 0
                    #print("{3}:{0} vs {1} : \t\t\t{2}\tDIFF".format(i,j,probs[k],k))
                k += 1
    return probs,y

def compute_interdist_L2(network,dataset_test,nbclasseval=10, nbsampleperclass=100):
    '''
    Computes L2 distances between all images of classes vs each other 
        
    Returns:
        list of shape (nb_classes) each element i containing a list of all the distances between class i images and other classes
    '''

    embeddings = []
    #generates embeddings for test images
    for i in range(nbclasseval):
        m_i = dataset_test[i].shape[0]
        print("Encoding test images for class {0}               ".format(i),end="\r")
        emb = network.predict(dataset_test[i].reshape((m_i,dataset_test[i].shape[1],dataset_test[i].shape[2],1)))
        size_embedding = emb.shape[1]
        #print(emb.shape)
        embeddings.append(emb)
    
    res = []
    for i in range(nbclasseval):
        res.append([])
    step=0
    nstep = int(nbclasseval*(nbclasseval-1)/2)
    
    for i in range(nbclasseval):
        for j in range(i+1,nbclasseval):
            print("Computing L2 distance of class {} vs {} {:.0%}".format(i,j,step/nstep),end = "                          \r")
            step+=1
            m_i = min(dataset_test[i].shape[0],nbsampleperclass)
            m_j = min(dataset_test[j].shape[0],nbsampleperclass)
            nbevaluation = m_i*m_j
            A = np.zeros((nbevaluation,size_embedding))
            B = np.zeros((nbevaluation,size_embedding))
            
            k = 0
            for iidx in range(m_i):
                for jidx in range(m_j):
                    #store embeddings1
                    A[k,:] = embeddings[i][iidx]
                    #store embeddings2
                    B[k,:] = embeddings[j][jidx]
                    k +=1
                    
            ij_distances = np.sum(np.square(A-B),axis=-1)
            
            #this is all the distances between class i and class j
            #So we add them to i vs all
            res[i].extend(ij_distances)
            #But also to j vs all
            res[j].extend(ij_distances)
         
    
    print("Computing L2 distance done                                                       ")  
    return res

def draw_interdist(network,n_iteration, dataset_test,savewandb=False, titleprefix="",nbclasseval=10, nbsampleperclass=100, folder="./"):
    data = compute_cache("interdist_"+titleprefix,lambda :compute_interdist_L2(network,dataset_test,nbclasseval, nbsampleperclass), \
                         parameter_dict={'n_iteration':n_iteration, 'interdistnbclass':nbclasseval,'interdistnbsample':nbsampleperclass}, folder=folder)
    if type(data)==np.ndarray: data=data.tolist()    
    nbmaxclass_to_display=40
    n=min(len(data),nbmaxclass_to_display)

    fig, ax = plt.subplots()
    fulltitle = titleprefix+'Interdistance after {0} iterations'.format(n_iteration)
    ax.set_title(fulltitle)
    ax.set_ylim([0,4])
    plt.xlabel('Classes')
    plt.ylabel('Distance')
    ax.boxplot(data[:n],showfliers=False,showbox=True)
    locs, labels = plt.xticks()
    plt.xticks(locs,np.arange(nbclasseval))

    fig.set_size_inches(900/80.0, 400/80.0)
    plt.show()
    if savewandb:
        wandb.log({fulltitle:wandb.Image(fig)})
          
    #Agregration de toutes les stats de classes
    fig2, ax2 = plt.subplots()
    fulltitle = titleprefix+'Interdistance (all classes) after {0} iterations'.format(n_iteration)
    ax2.set_title(fulltitle)
    ax2.set_ylim([0,4])
    plt.xlabel('Classes')
    plt.ylabel('Distance')
    
    alldata = np.concatenate(data)
    ax2.boxplot(alldata,showfliers=True,showbox=True)

    plt.show()
    if savewandb:
        wandb.log({fulltitle:wandb.Image(fig2)})
          
         
        
def DrawTestImage(network, images, dataset_test, refidx=0,nb_test_class=10):
    
    _, w,h,c = dataset_test[0].shape
    nbimages=images.shape[0]
    
    #generates embedings for given images
    image_embedings = network.predict(images)
    
    #generates embedings for reference images
    ref_images = np.zeros((nb_test_class,w,h,c))
    for i in range(nb_test_class):
        ref_images[i,:,:,:] = dataset_test[i][refidx,:,:,:]
    ref_embedings = network.predict(ref_images)
            
    for i in range(nbimages):
        #Prepare the figure
        fig=plt.figure(figsize=(16,2))
        subplot = fig.add_subplot(1,nb_test_class+1,1)
        axis("off")
        plotidx = 2
            
        #Draw this image    
        plt.imshow(images[i,:,:,0],vmin=0, vmax=1,cmap='Greys')
        subplot.title.set_text("Test image")
            
        for ref in range(nb_test_class):
            #Compute distance between this images and references
            dist = compute_l2_dist(image_embedings[i,:],ref_embedings[ref,:])
            #Draw
            subplot = fig.add_subplot(1,nb_test_class+1,plotidx)
            axis("off")
            plt.imshow(ref_images[ref,:,:,0],vmin=0, vmax=1,cmap='Greys')
            subplot.title.set_text(("Class {0}\n{1:.3e}".format(ref,dist)))
            plotidx += 1

def DrawTestImageWithRank(network, images, dataset_test, threshold, classindicator=-1, refidx=0,nb_test_class=400 ):
    
    _, w,h,c = dataset_test[0].shape
    nbimages=images.shape[0]
    nb_display=10
    #generates embedings for given images
    image_embedings = network.predict(images)
    
    #generates embedings for reference images
    ref_images = np.zeros((nb_test_class,w,h,c))
    for i in range(nb_test_class):
        ref_images[i,:,:,:] = dataset_test[i][refidx,:,:,:]
    ref_embedings = network.predict(ref_images)
            
    for i in range(nbimages):
        if nbimages>1:
            trueclass=i
        else:
            trueclass=classindicator
        
        #Prepare the figure
        fig=plt.figure(figsize=(16,2))
        subplot = fig.add_subplot(1,nb_display+1,1)
        axis("off")
        plotidx = 2
            
        #Draw this image    
        plt.imshow(images[i,:,:,0],vmin=0, vmax=1,cmap='Greys')
        subplot.title.set_text("Test image")
            
        distdtype=[('class', int), ('dist', float)]
        dist = np.zeros(nb_test_class, dtype=distdtype)
        
        #Compute distances
        for ref in range(nb_test_class):
            #Compute distance between this images and references
            dist[ref] = (ref,compute_l2_dist(image_embedings[i,:],ref_embedings[ref,:]))
            
        #sort
        sorted_dist = np.sort(dist, order='dist')
        
        #Draw
        for j in range(min(10,nb_test_class)):
            subplot = fig.add_subplot(1,nb_display+1,plotidx)

            axis("off")
            #subplot.get_xaxis().set_visible(False)
            #subplot.get_yaxis().set_visible(False)
            #subplot.set_facecolor((0,0,1))
            
            plt.imshow(ref_images[sorted_dist['class'][j],:,:,0],vmin=0, vmax=1,cmap='Greys')
            
            #Red for sample above threshold
            if (sorted_dist['dist'][j] > threshold):
                if (trueclass == sorted_dist['class'][j]):
                    color = (1,0,0)
                    label = "TRUE"
                else:
                    color = (0.5,0,0)
                    label = "Class {0}".format(sorted_dist['class'][j])
            else:
                if (trueclass == sorted_dist['class'][j]):
                    color = (0, 1, 0)
                    label = "TRUE"
                else:
                    color = (0, .5, 0)
                    label = "Class {0}".format(sorted_dist['class'][j])
                
            subplot.set_title("{0}\n{1:.3e}".format(label,sorted_dist['dist'][j]),color=color)
            plotidx += 1
                        
            
def ComputeCMCScoresL2(network, dataset_test, idxcatalog=0, idxcandidate=1, nb_test_class=400 ):
    
    _, w,h,c = dataset_test[0].shape
    
    #generates embeddings for candidate images
    candidate_images = np.zeros((nb_test_class,w,h,c))
    for i in range(nb_test_class):
        candidate_images[i,:,:,:] = dataset_test[i][idxcandidate,:,:,:]
    candidate_embeddings = network.predict(candidate_images)
    
    #generates embeddings for catalog images
    catalog_images = np.zeros((nb_test_class,w,h,c))
    for i in range(nb_test_class):
        catalog_images[i,:,:,:] = dataset_test[i][idxcatalog,:,:,:]
    catalog_embeddings = network.predict(catalog_images)
       
    #ranks[i] will store the number of time the candidate matched the right class at rank at least i
    ranks = np.zeros(nb_test_class)
        
    #for each candidate
    for i in range(nb_test_class):

        #Compute distances
        predictionsdtype=[('class', int), ('dist', float)]
        predictions = np.zeros(nb_test_class, dtype=predictionsdtype)
        for ref in range(nb_test_class):
            #Compute distance between the candidate and catalog
            predictions[ref] = (ref,compute_l2_dist(candidate_embeddings[i,:],catalog_embeddings[ref,:]))
        #print("predictions",predictions)   

        #sort : now all predictions are ranked from the smallest distance from the candidate to the biggest
        sorted_predictions = np.sort(predictions, order='dist')
        #print("sorted_predictions",sorted_predictions)
        rankedPredictions = sorted_predictions['class']
        #print("rankedPredictions",rankedPredictions)
        
        #if i is in the predictions
        if i in rankedPredictions :
            #lets find at which rank
            firstOccurance = np.argmax(rankedPredictions == i)        
            
            #update ranks from firstOccurance to the end
            for j in range(firstOccurance, nb_test_class) :            
                ranks[j] +=1
        #print("ranks",ranks)
    
    #Computes CMC Scores from ranks
    cmcScores = ranks / nb_test_class
    
    return cmcScores

def drawTriplets(tripletbatch, nbmax=None):
    """display the three images for each triplets in the batch
    """
    labels = ["Anchor", "Positive", "Negative"]

    if (nbmax==None):
        nbrows = tripletbatch[0].shape[0]
    else:
        nbrows = min(nbmax,tripletbatch[0].shape[0])
                 
    for row in range(nbrows):
        fig=plt.figure(figsize=(16,2))
    
        for i in range(3):
            subplot = fig.add_subplot(1,3,i+1)
            axis("off")
            plt.imshow(tripletbatch[i][row,:,:,0],vmin=0, vmax=1,cmap='Greys')
            subplot.title.set_text(labels[i])