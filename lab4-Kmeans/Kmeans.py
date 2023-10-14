from sklearn import preprocessing
import PIL.Image as Image
import numpy as np
import sys
import pandas as pd
import random
import matplotlib.pyplot as plt

def load_pic(file_path):
    '''
    Generate numpy matrix according to input file path.
    Return values include list, pic_width & pic_height. 
    '''
    with open(file_path,'rb') as f: # open picture file
        data = []
        position = []
        image = Image.open(f)   
        width, height = image.size
        for x in range(width):
            for y in range(height):
                r,g,b = image.getpixel((x,y)) # get rgb at(x,y)
                position.append((x,y))
                data.append([r/255.0,g/255.0,b/255.0]) # normalize
    print("Picuture successfully loaded.")
    return data, width, height, position

def cal_dist(dataset, centroids):
    '''Calculate Euclidean distance '''
    dist_list = []
    for data in dataset:
        # fill the data k times to minus centroids
        diff = np.tile(data,(len(centroids),1))-centroids
        dist_square = np.sum(np.square(diff),axis=1)
        dist_list.append(dist_square)
    return np.array(dist_list)

def cal_centroids(dataset,centroids):
    '''calculate new centroids'''
    dist_list = cal_dist(dataset,centroids)
    # calculate the centroids index of min distance
    min_idx = np.argmin(dist_list,axis=1) 
    new_centroids = pd.DataFrame(dataset).groupby(min_idx).mean().values

    # calculate the changes
    changes = new_centroids-centroids

    return changes,new_centroids

def Kmeans(X,k, position):
    centroids = random.sample(X,k) # randomly sample k centroids
    
    changes, centroids = cal_centroids(X,centroids)
    # continue updating centroids until changes equals zero
    # or stop when epoch reaches 100000
    epoch = 1
    while np.any(changes != 0) or epoch > 50:
        changes, centroids = cal_centroids(X,centroids)
        print("Finished: k=%(k)i, epoch=%(e)i"% {"k":k,"e":epoch})
        epoch += 1

    # calculate clusters according to centroids
    cluster = []
    rgb_cluster = [] # calculate rgb clusters to estimate results(SSE)
    for i in range(k):
        cluster.append([])
        rgb_cluster.append([])
    dist_list = cal_dist(X,centroids)
    min_idx = np.argmin(dist_list,axis=1)
    for i,j in enumerate(min_idx):
        cluster[j].append(position[i])
        rgb_cluster[j].append(X[i])

    print("Finish Kmeans: K=%i"% k)
    return centroids,cluster,rgb_cluster

def draw_result(pic_path,results,width,height):
    plt.title("Picture Segment using Kmeans") # title for the pic
    plt.subplot(231)
    plt.imshow(Image.open(pic_path)) # display original pic
    plt.axis('off') # shut down the axis
    plt.xticks([])
    plt.yticks([]) # shut down x,y values

    colors = [(0, 0, 255),(255, 0, 0),(0, 255, 0),(60, 0, 220),(167, 255, 167)]
    
    n = 2
    for res in results:
        new_pic = Image.new("RGB",(width,height))
        for i in range(len(res[0])):
            for j in res[1][i]:
                new_pic.putpixel(j,colors[i])
        plt.subplot(230+n)
        plt.title("K=%i"% n)
        plt.imshow(new_pic)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        n += 1
    
    plt.show()
    
path = "./flowers.jpg"
pic_data, width, height, position = load_pic(path)

res = []
all_rgb = []
for k in range(2,6):
    centroids, cluster, rgb_cluster = Kmeans(pic_data,k,position)
    res.append((centroids,cluster))
    all_rgb.append(rgb_cluster)

draw_result(path,res,width,height)
