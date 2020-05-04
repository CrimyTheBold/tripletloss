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

class QuadrupletLossLayer(Layer):
    def __init__(self, alpha, beta, **kwargs):
        self.alpha = alpha
        self.beta = beta
        self.debugeric = 1
  
        super(QuadrupletLossLayer, self).__init__(**kwargs)
    
    def quadruplet_loss(self, inputs):
        ap_dist,an_dist,nn_dist = inputs
        
        #square
        ap_dist2 = K.square(ap_dist)
        an_dist2 = K.square(an_dist)
        nn_dist2 = K.square(nn_dist)
        
        return K.sum(K.maximum(ap_dist2 - an_dist2 + self.alpha, 0), axis=0) + K.sum(K.maximum(ap_dist2 - nn_dist2 + self.beta, 0), axis=0)
    
    def call(self, inputs):
        loss = self.quadruplet_loss(inputs)
        self.add_loss(loss)
        return loss
    
def build_model4(input_shape, network, metricnetwork,margin=0.1, margin2=0.01):
    '''
    Define the Keras Model for training 
        Input : 
            input_shape : shape of input images
            network : Neural network to train outputing embeddings
            metricnetwork : Neural network to train the learned metric
            margin : minimal distance between Anchor-Positive and Anchor-Negative for the lossfunction (alpha1)
            margin2 : minimal distance between Anchor-Positive and Negative-Negative2 for the lossfunction (alpha2)
    
    '''
     # Define the tensors for the four input images
    anchor_input = Input(input_shape, name="anchor_input")
    positive_input = Input(input_shape, name="positive_input")
    negative_input = Input(input_shape, name="negative_input") 
    negative2_input = Input(input_shape, name="negative2_input")
    
    # Generate the encodings (feature vectors) for the four images
    encoded_a = network(anchor_input)
    encoded_p = network(positive_input)
    encoded_n = network(negative_input)
    encoded_n2 = network(negative2_input)
    
    #compute the concatenated pairs
    encoded_ap = Concatenate(axis=-1,name="Anchor-Positive")([encoded_a,encoded_p])
    encoded_an = Concatenate(axis=-1,name="Anchor-Negative")([encoded_a,encoded_n])
    encoded_nn = Concatenate(axis=-1,name="Negative-Negative2")([encoded_n,encoded_n2])
    
    #compute the distances AP, AN, NN
    ap_dist = metricnetwork(encoded_ap)
    an_dist = metricnetwork(encoded_an)
    nn_dist = metricnetwork(encoded_nn)
    
    #QuadrupletLoss Layer
    loss_layer = QuadrupletLossLayer(alpha=margin,beta=margin2,name='4xLoss')([ap_dist,an_dist,nn_dist])
    
    # Connect the inputs with the outputs
    network_train = Model(inputs=[anchor_input,positive_input,negative_input,negative2_input],outputs=loss_layer)
    
    # return the model
    return network_train

def simulateQuadLoss(quadruplets, network, metricnetwork, alpha1, alpha2):
    '''
    WARNING BUGGY
    Try to simulate the loss computation. For some unknown reason, it doesnt find the same result as keras does.
    I used it anyway to look specificaly at the balance between the strong push vs the weak push while still verifying that the error between 
    the real loss adn the result of this function remains low (like <20%).
    
    '''
   
    # Define the tensors for the four input images
    anchor_input,positive_input,negative_input,negative2_input = quadruplets

    # Generate the encodings (feature vectors) for the four images
    encoded_a = network.predict(anchor_input)
    encoded_p = network.predict(positive_input)
    encoded_n = network.predict(negative_input)
    encoded_n2 = network.predict(negative2_input)
   
    size_embedding = encoded_a.shape[0]
    size_pair = size_embedding*2
   
    #compute the concatenated pairs
   
    encoded_ap = np.concatenate(([encoded_a,encoded_p]),axis=-1)
    #print("encoded_a",encoded_a,"encoded_p",encoded_p,"encoded_ap",encoded_ap)
    encoded_an = np.concatenate(([encoded_a,encoded_n]),axis=-1)
    encoded_nn = np.concatenate(([encoded_n,encoded_n2]),axis=-1)
   
    #compute the distances AP, AN, NN
    ap_dist = metricnetwork.predict(encoded_ap)
    an_dist = metricnetwork.predict(encoded_an)
    nn_dist = metricnetwork.predict(encoded_nn)

    #Loss terms
    #print("ap_dist",ap_dist,"an_dist",an_dist,"nn_dist",nn_dist)
    #print("T1",np.maximum(ap_dist - an_dist + alpha1, 0),"T2",np.maximum(ap_dist - nn_dist + alpha2, 0))
   
    strongpush = np.sum(np.maximum(ap_dist - an_dist + alpha1, 0), axis=0)
    weakpush = np.sum(np.maximum(ap_dist - nn_dist + alpha2, 0), axis=0)
    totalloss = strongpush+weakpush

    return totalloss,strongpush,weakpush

####################### TRAINING
#@profile
def get_batch_random(batch_size,X):
    """
    Create batch of APN quadruplets with a complete random strategy
    
    Arguments:
    batch_size -- integer 
    X          -- list containing n tensors of shape (?,w,h,c) to draw the batch from

    Returns:
    quadruplets -- list containing 4 tensors A,P,N,N2 of shape (batch_size,w,h,c)
    """
    n = len(X)
    m, w, h,c = X[0].shape
    
    # initialize result
    quadruplets=[np.zeros((batch_size,h, w,c)) for i in range(4)]
    
    for i in range(batch_size):
        #Pick one random class for anchor
        anchor_class = np.random.randint(0, n)
        nb_sample_available_for_class_AP = X[anchor_class].shape[0]
        
        #Pick two different random pics for this class => A and P
        [idx_A,idx_P] = np.random.choice(nb_sample_available_for_class_AP,size=2,replace=False)
        
        #Pick another class for N, different from anchor_class
        negative_class = (anchor_class + np.random.randint(1,n)) % n
        nb_sample_available_for_class_N = X[negative_class].shape[0]
        
        #Pick a random pic for this negative class => N
        idx_N = np.random.randint(0, nb_sample_available_for_class_N)
        
        #Pick another class for N2, different from anchor_class and negative_class
        #remainingClasses = np.setdiff1d(range(n),[anchor_class,negative_class])
        remainingClasses = np.arange(n)
        np.delete(remainingClasses,[anchor_class,negative_class],axis=None)
        negative2_class = np.random.choice(remainingClasses,1)[0]
        nb_sample_available_for_class_N2 = X[negative2_class].shape[0]
                                        
        #Pick a random pic for this negative class => N2
        idx_N2 = np.random.randint(0, nb_sample_available_for_class_N2)

        quadruplets[0][i,:,:,:] = X[anchor_class][idx_A,:,:,:]
        quadruplets[1][i,:,:,:] = X[anchor_class][idx_P,:,:,:]
        quadruplets[2][i,:,:,:] = X[negative_class][idx_N,:,:,:]
        quadruplets[3][i,:,:,:] = X[negative2_class][idx_N2,:,:,:]

    return quadruplets

def drawQuadriplets(quadripletbatch, nbmax=None):
    """display the four images for each quadriplets in the batch
    """
    labels = ["Anchor", "Positive", "Negative", "Negative2"]

    if (nbmax==None):
        nbrows = quadripletbatch[0].shape[0]
    else:
        nbrows = min(nbmax,quadripletbatch[0].shape[0])
                 
    for row in range(nbrows):
        fig=plt.figure(figsize=(16,2))
    
        for i in range(4):
            subplot = fig.add_subplot(1,4,i+1)
            axis("off")
            plt.imshow(quadripletbatch[i][row,:,:,0],vmin=0, vmax=1,cmap='Greys')
            subplot.title.set_text(labels[i])

#@profile            
def get_batch_hard(draw_batch_size,hard_batchs_size,norm_batchs_size,network3, network4, metric_network4, X):
    """
    Create batch of APN "hard" triplets/quadruplets
    
    Arguments:
    draw_batch_size -- integer : number of initial randomly taken samples   
    hard_batchs_size -- interger : select the number of hardest samples to keep
    norm_batchs_size -- interger : number of random samples to add

    Returns:
    triplets -- list containing 3 tensors A,P,N of shape (hard_batchs_size+norm_batchs_size,w,h,c)
    quadruplets -- list containing 4 tensors A,P,N,N2 of shape (hard_batchs_size+norm_batchs_size,w,h,c)
    """
    m, w, h,c = X[0].shape
    
    #Step 1 : pick a random batch to study
    studybatch = get_batch_random(draw_batch_size,X)
    
    #Step 2 : compute the loss with current network : d(A,P)-d(A,N). The alpha parameter here is omited here since we want only to order them
    studybatchloss = np.zeros((draw_batch_size))
    
    #Compute embeddings for anchors, positive and negatives
    A = network3.predict(studybatch[0])
    P = network3.predict(studybatch[1])
    N = network3.predict(studybatch[2])
    
    #Compute d(A,P)-d(A,N)
    studybatchtripletloss = np.sum(np.square(A-P),axis=1) - np.sum(np.square(A-N),axis=1)
    
    #Sort by distance (high distance first) and take hard_batchs_size smaples from it
    selectiontriplet = np.argsort(studybatchtripletloss)[::-1][:hard_batchs_size]
    
    #Draw other random samples from the batch
    selection2triplet = np.random.choice(np.delete(np.arange(draw_batch_size),selectiontriplet),norm_batchs_size,replace=False)
    selectiontriplet = np.append(selectiontriplet,selection2triplet)
    
    triplets = [studybatch[0][selectiontriplet,:,:,:], studybatch[1][selectiontriplet,:,:,:], studybatch[2][selectiontriplet,:,:,:]]
    
    #Compute same with 4xloss
    #========================
    
    #Embeddings
    A = network4.predict(studybatch[0])
    P = network4.predict(studybatch[1])
    N = network4.predict(studybatch[2])
    N2 = network4.predict(studybatch[3])
    
    APNN2 = network4.predict(np.concatenate(studybatch, axis=0))
    
    #compute the concatenated pairs
    encoded_ap = np.concatenate([A,P], axis=-1)
    encoded_an = np.concatenate([A,N], axis=-1)
    encoded_nn = np.concatenate([N,N2], axis=-1)
    
    encoded_all = np.concatenate([encoded_ap,encoded_an,encoded_nn],axis=0)
    
    #compute the distances AP, AN, NN
    ap_dist = metric_network4.predict(encoded_ap)
    an_dist = metric_network4.predict(encoded_an)
    nn_dist = metric_network4.predict(encoded_nn)
    
    all_dist = metric_network4.predict(encoded_all)
    
    #compute d(A,P)-d(A,N) + d(A,P)-d(N,N2)
    studybatchquadrupletloss = 2*ap_dist - an_dist - nn_dist
    
    #Sort by distance
    selectionquadruplet = np.argsort(studybatchquadrupletloss)[::-1][:hard_batchs_size]
    
    #Draw other random samples from the batch
    selection2quadruplet = np.random.choice(np.delete(np.arange(draw_batch_size),selectionquadruplet),norm_batchs_size,replace=False)
    selectionquadruplet = np.append(selectionquadruplet,selection2quadruplet)
    
    quadruplets = [studybatch[0][selectionquadruplet,:,:,:], studybatch[1][selectionquadruplet,:,:,:], studybatch[2][selectionquadruplet,:,:,:],studybatch[3][selectionquadruplet,:,:,:] ]
    
    return triplets,quadruplets

#@profile
def get_batch_hardOptimized(draw_batch_size,hard_batchs_size,norm_batchs_size,network3, network4, metric_network4, X):
    """
    Create batch of APN "hard" triplets/quadruplets
    
    Arguments:
    draw_batch_size -- integer : number of initial randomly taken samples   
    hard_batchs_size -- interger : select the number of hardest samples to keep
    norm_batchs_size -- interger : number of random samples to add

    Returns:
    triplets -- list containing 3 tensors A,P,N of shape (hard_batchs_size+norm_batchs_size,w,h,c)
    quadruplets -- list containing 4 tensors A,P,N,N2 of shape (hard_batchs_size+norm_batchs_size,w,h,c)
    """
    m, w, h,c = X[0].shape
    
    #Step 1 : pick a random batch to study
    studybatch = get_batch_random(draw_batch_size,X)
    
    #Step 2 : compute the loss with current network : d(A,P)-d(A,N). The alpha parameter here is omited here since we want only to order them
    studybatchloss = np.zeros((draw_batch_size))
    
    #Compute embeddings for anchors, positive and negatives
    #A = network3.predict(studybatch[0])
    #P = network3.predict(studybatch[1])
    #N = network3.predict(studybatch[2])
    APN= network3.predict(np.concatenate(studybatch[:3], axis=0))
    A,P,N = np.split(APN,3,axis=0)           
    
    #Compute d(A,P)-d(A,N)
    studybatchtripletloss = np.sum(np.square(A-P),axis=1) - np.sum(np.square(A-N),axis=1)
    
    #Sort by distance (high distance first) and take hard_batchs_size smaples from it
    selectiontriplet = np.argsort(studybatchtripletloss)[::-1][:hard_batchs_size]
    
    #Draw other random samples from the batch
    selection2triplet = np.random.choice(np.delete(np.arange(draw_batch_size),selectiontriplet),norm_batchs_size,replace=False)
    selectiontriplet = np.append(selectiontriplet,selection2triplet)
    
    triplets = [studybatch[0][selectiontriplet,:,:,:], studybatch[1][selectiontriplet,:,:,:], studybatch[2][selectiontriplet,:,:,:]]
    
    #Compute same with 4xloss
    #========================
    
    #Embeddings
    APNN2 = network4.predict(np.concatenate(studybatch, axis=0))
    A,P,N,N2 = np.split(APNN2,4,axis=0)
    
    #compute the concatenated pairs
    encoded_ap = np.concatenate([A,P], axis=-1)
    encoded_an = np.concatenate([A,N], axis=-1)
    encoded_nn = np.concatenate([N,N2], axis=-1)
    encoded_all = np.concatenate([encoded_ap,encoded_an,encoded_nn],axis=0)
    
    #compute the distances AP, AN, NN
    #ap_dist = metric_network4.predict(encoded_ap)
    #an_dist = metric_network4.predict(encoded_an)
    #nn_dist = metric_network4.predict(encoded_nn)
    all_dist = metric_network4.predict(encoded_all)
    ap_dist,an_dist,nn_dist = np.split(all_dist,3,axis=0)
    
    #compute d(A,P)-d(A,N) + d(A,P)-d(N,N2)
    studybatchquadrupletloss = 2*ap_dist - an_dist - nn_dist
    
    #Sort by distance
    selectionquadruplet = np.argsort(studybatchquadrupletloss)[::-1][:hard_batchs_size]
    
    #Draw other random samples from the batch
    selection2quadruplet = np.random.choice(np.delete(np.arange(draw_batch_size),selectionquadruplet),norm_batchs_size,replace=False)
    selectionquadruplet = np.append(selectionquadruplet,selection2quadruplet)
    
    quadruplets = [studybatch[0][selectionquadruplet,:,:,:], studybatch[1][selectionquadruplet,:,:,:], studybatch[2][selectionquadruplet,:,:,:],studybatch[3][selectionquadruplet,:,:,:] ]
    
    return triplets,quadruplets


#to test normal vs optimized
#for i in range (10):
#    np.random.seed(i)
#    microtask_start =time.time() 
#    hardtriplets,hardquadruplets = get_batch_hard(100,16,16,network3,network4,metric_network4,dataset_train)
#    t1 = time.time()-microtask_start
#    np.random.seed(i)
#    microtask_start =time.time() 
#    hardtriplets2,hardquadruplets2 = get_batch_hardOptimized(100,16,16,network3,network4,metric_network4,dataset_train)
#    t2 = time.time()-microtask_start
#    print("hardtriplets equals\t",array_equal(hardtriplets,hardtriplets2), \
#          "hardquadruplets equals\t",array_equal(hardquadruplets,hardquadruplets2), \
#         t1,t2,"{:.0%}".format( (t2-t1)/t1))
    
####################### EVALUATION

def compute_probs3i(network,metricnetwork,X,Y):
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
            #print(i)
            #Against all other images
            for j in range(i+1,m):
                #compute the probability of being the right decision : it should be 1 for right class, 0 for all other classes
                probs[k] = -compute_learned_dist(embeddings[i,:],embeddings[j,:],metricnetwork)
                if (Y[i]==Y[j]):
                    y[k] = 1
                    #print("{3}:{0} vs {1} : {2}\tSAME".format(i,j,probs[k],k))
                else:
                    y[k] = 0
                    #print("{3}:{0} vs {1} : \t\t\t{2}\tDIFF".format(i,j,probs[k],k))
                k += 1
    return probs,y

def compute_probs3iVectorized(network,metricnetwork,X,Y):
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
    
    #Array to store embeddings pair to be fed into metricnetwork
    embeddingpairs = np.zeros((nbevaluation,size_embedding*2))
    
    #For each pics of our dataset
    k = 0
    for i in range(m):
            #print(i)
            #Against all other images
            for j in range(i+1,m):
                #store embeddings1
                embeddingpairs[k,:size_embedding] = embeddings[i,:]
                #store embeddings2
                embeddingpairs[k,size_embedding:] = embeddings[j,:]
                
                #compute the probability of being the right decision : it should be 1 for right class, 0 for all other classes
                if (Y[i]==Y[j]):
                    y[k] = 1
                else:
                    y[k] = 0
                k += 1
    
    #Take inverted distance as probabilities
    probs = metricnetwork.predict(embeddingpairs) * -1
    return probs,y

def compute_interdist_learnedmetric(network,metricnetwork,dataset_test,nbclasseval=10, nbsampleperclass=100):
    size_embedding=0
    embeddings = []
    
    #generates embeddings for reference images (images at index 0)
    for i in range(nbclasseval):
        m_i = dataset_test[i].shape[0]
        print("Encoding test images for class {0}".format(i),end="                      \r")
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
            print("Computing learned distance of class {} vs {} {:.0%}".format(i,j,step/nstep),end = "              \r")
            step+=1
            m_i = min(dataset_test[i].shape[0],nbsampleperclass)
            m_j = min(dataset_test[j].shape[0],nbsampleperclass)
            nbevaluation = m_i*m_j
            embeddingpairs = np.zeros((nbevaluation,size_embedding*2))
            k = 0
            for iidx in range(m_i):
                for jidx in range(m_j):
                    #store embeddings1
                    embeddingpairs[k,:size_embedding] = embeddings[i][iidx]
                    #store embeddings2
                    embeddingpairs[k,size_embedding:] = embeddings[j][jidx]
                    k +=1
            ij_distances = metricnetwork.predict(embeddingpairs)
            
            #this is all the distances between class i and class j
            #So we add them to i vs all
            res[i].extend(ij_distances)
            #But also to j vs all
            res[j].extend(ij_distances)
    print("Computing learned distance done                                  ")  
    return res
        
def draw_interdist_learnedmetric(network,metricnetwork,n_iteration, dataset_test,savewandb=False, titleprefix="",nbclasseval=10, nbsampleperclass=100, folder="./"):
    data = compute_cache("interdist_"+titleprefix, lambda :compute_interdist_learnedmetric(network,metricnetwork,dataset_test,nbclasseval,nbsampleperclass), \
                         parameter_dict={'n_iteration':n_iteration, 'interdistnbclass':nbclasseval,'interdistnbsample':nbsampleperclass}, folder=folder)
    if type(data)==np.ndarray: data=data.tolist()    
        
    nbmaxclass_to_display=40 
    n=min(len(data),nbmaxclass_to_display)
    
    fig, ax = plt.subplots()
    fulltitle = titleprefix+'Interdistance after {0} iterations'.format(n_iteration)
    ax.set_title(fulltitle)
    ax.set_ylim([0,1.1])
    plt.xlabel('Classes')
    plt.ylabel('Distance')
    ax.boxplot(data[:n],showfliers=True,showbox=True)
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
    ax2.set_ylim([0,1.1])
    plt.xlabel('Classes')
    plt.ylabel('Distance')
    
    alldata = np.concatenate(data)
    ax2.boxplot(alldata,showfliers=True,showbox=True)

    plt.show()
    if savewandb:
        wandb.log({fulltitle:wandb.Image(fig2)})
        
def compute_learned_dist(a,b,metricnetwork):
    c = np.concatenate((a,b), axis=-1)
    d = np.reshape(c,(1,c.shape[0]))
    return metricnetwork.predict(d)

def DrawTestImageLearnedMetric(network, metricnetwork, images, dataset_test, refidx=0, nb_test_class=10):
    '''
    Evaluate some pictures vs some samples in the test set
        image must be of shape(1,w,h,c)
    
    Returns
        scores : resultat des scores de similarités avec les images de base => (N)
    
    '''
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
            dist = compute_learned_dist(image_embedings[i,:],ref_embedings[ref,:],metricnetwork)
            #Draw
            subplot = fig.add_subplot(1,nb_test_class+1,plotidx)
            axis("off")
            plt.imshow(ref_images[ref,:,:,0],vmin=0, vmax=1,cmap='Greys')
            subplot.title.set_text(("Class {0}\n{1:0.3}".format(ref,dist[0])))
            plotidx += 1
            
def ComputeCMCScoresLearnedMetricSlow(network, metric_network, dataset_test, idxcatalog=0, idxcandidate=1, nb_test_class=400 ):
    ''' Reference version for ComputeCMCScoresLearnedMetric. This one is one order of magnitude slower'''
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
            
            #concatenate condidate and catalog element
            candidate_catalogelem_embeddingpair = np.concatenate([candidate_embeddings[i,:],catalog_embeddings[ref,:]], axis=-1)
            candidate_catalogelem_embeddingpair = np.reshape(candidate_catalogelem_embeddingpair,(1,candidate_catalogelem_embeddingpair.shape[0]))
            #use the metric network to compute distance
            dist = metric_network.predict(candidate_catalogelem_embeddingpair)
            
            predictions[ref] = (ref,dist[0])
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

def ComputeCMCScoresLearnedMetric(network, metric_network, dataset_test, idxcatalog=0, idxcandidate=1, nb_test_class=400 ):
    '''Computes Cumulative Match Characteristic (CMC) score, using an encoding network and a learned metric network
    Inputs:
        network : encoding neural network. Should outputs embeddings of size embeddings_size
        metric_network : metric network. inputs a concatenated pair of embeddings, outputs distance
        idxcatalog : Catalog index to select samples from each class from the dataset
        idxcandidate : Index to select candidate samples from the dataset
        nb_test_class : number of class to compute. Should be <len(dataset_test)
    '''
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
    
    embbeding_size = catalog_embeddings.shape[1]
       
    #ranks[i] will store the number of time the candidate matched the right class at rank at least i
    ranks = np.zeros(nb_test_class)
        
    #for each candidate
    for i in range(nb_test_class):

        #Compute distances
        predictionsdtype=[('class', int), ('dist', float)]
        predictions = np.zeros(nb_test_class, dtype=predictionsdtype)
        
        #first compute all candidate vs catalog pairs
        candidate_catalog_embeddingpair = np.zeros((nb_test_class,embbeding_size*2))
        for ref in range(nb_test_class):
            #concatenate condidate and catalog element
            candidate_catalog_embeddingpair[ref] = np.concatenate([candidate_embeddings[i,:],catalog_embeddings[ref,:]], axis=-1)

         #use the metric network to compute all distances in one go
        dist = metric_network.predict(candidate_catalog_embeddingpair)

        for ref in range(nb_test_class):
            predictions[ref] = (ref,dist[ref])
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

def DrawTestImageLearnedMetricWithRank(network, metric_network, images, dataset_test, threshold, classindicator=-1, refidx=0,nb_test_class=400 ):
    
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
            
    embbeding_size = ref_embedings.shape[1]
    
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
        #first compute all candidate vs catalog pairs
        candidate_catalog_embeddingpair = np.zeros((nb_test_class,embbeding_size*2))
        for ref in range(nb_test_class):
            #concatenate condidate and catalog element
            candidate_catalog_embeddingpair[ref] = np.concatenate([image_embedings[i,:],ref_embedings[ref,:]], axis=-1)

        #use the metric network to compute all distances in one go
        alldist = metric_network.predict(candidate_catalog_embeddingpair)
        
        for ref in range(nb_test_class):
            dist[ref] = (ref,alldist[ref])
        
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
                
            subplot.set_title("{0}\n{1:0.3}".format(label,sorted_dist['dist'][j]),color=color)
            plotidx += 1
 