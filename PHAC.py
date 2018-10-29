"""
 hierarchial agglomerative clustering  
"""
import multiprocessing as mp
import math as m   
from matplotlib import pyplot as plt  
from scipy.cluster.hierarchy import dendrogram, linkage  
import csv  # Importing csv library
from multiprocessing import Process,Manager,current_process
import datetime

class ClusterPrototype:
    sum_avg_attribute = None

    def __init__(self, data=None, label=None):
        self.data = data
        self.id_list = []   
        self.sum_avg_attribute = [] 
        self.label = label
        if data != None:
            self.size = len(data)
            for col in range(1,len(self.data[0])):
                summ = 0
                for row in self.data:
                    summ += row[col]
                self.sum_avg_attribute.append(summ / self.size)
            for row in data:
                self.id_list.append(row[0])

    def merge_cluster(self, pro):
        for row in pro.data:
            self.data.append(row)
            self.id_list.append(row[0])
        self.size = len(self.data)
        self.sum_avg_attribute = []
        for col in range(1,len(self.data[0])):
            summ = 0
            for row in self.data:
                summ += row[col]
            self.sum_avg_attribute.append(summ / self.size)

    def get_data(self):
        return self.data

    def get_label(self):

        return self.label

    def get_size(self):
        return self.size

    def get_id_list(self):
        return self.id_list

    def get_com(self):
        return self.sum_avg_attribute

def calculate_euclidean_dist( com_1, com_2):
    squared_diff_sum = 0
    for d in range(len(com_1)):
        squared_diff_sum += (com_1[d] - com_2[d])**2

    return m.sqrt(squared_diff_sum)

def assign_cluster( i, y,min_distance,clusters,min_distances,cluster1,cluster2,cluster1_ind,cluster2_ind):
    cluster_1 = None
    cluster_2 = None
    cluster_1_ind = None
    cluster_2_ind = None
    for j in range(i+1, y):
        dist = calculate_euclidean_dist (clusters[i].get_com (), clusters[j].get_com ())
        if dist < min_distance:
            min_distance = dist
            cluster_1 = clusters[i]
            cluster_2 = clusters[j]
            cluster_1_ind = i
            cluster_2_ind = j
    if cluster_1!= None:
        min_distances.append(min_distance)
        cluster1.append(cluster_1)
        cluster2.append(cluster_2)
        cluster1_ind.append(cluster_1_ind)
        cluster2_ind.append(cluster_2_ind)

id_data_map = {}  # Id and data mapping
data_list = []  # entire data
clusters = []  # List for clusters
merged_small_clusters = []  # list of the small clusters merged
min_distances = []  # entire data
min_distance = 10000

cluster1 = []  # entire data
cluster2 = []  # entire data
cluster1_ind = []  # entire data
cluster2_ind = []  # entire data


def cluster_creation( ):
        no_clusters__required = 7   
        cluster_1 = None
        cluster_2 = None
        cluster_1_ind = None
        cluster_2_ind = None

        count=0
        print(datetime.datetime.now().time())
        while len(clusters) > no_clusters__required:
            min_distance=10000
            prev_min_dist = 10000
            l = len(clusters)
            n= int((len(clusters)-1)/7)
            k=1
            z=n
            jobs = []
            manager = Manager ()
            min_distances = manager.list()
            cluster1 = manager.list()
            cluster2 = manager.list()
            cluster1_ind = manager.list()
            cluster2_ind = manager.list()
            clu = manager.list ()
            clu = clusters
            for i in range(7):
                #self.assign_cluster( k,z,self.min_distance )
				#*************************************7 Core***********************
				#*********************************************7 individaul Process ***************** 
                p = mp.Process(target=assign_cluster, args=(k,z,min_distance,clusters,min_distances,cluster1,cluster2,cluster1_ind,cluster2_ind))
                jobs.append(p)

                print('process %s',str(i))
                k = n * (i + 1)+1
                z = n * (i + 2)
                if i+1==4-1:
                    z= len(clusters)-1

            for proc in jobs:
                proc.start ()
                print ('process %s',  current_process().name )

            for proc in jobs:
                print (current_process().name)
                proc.join ()

            print (len(clusters))
            dist=0
            if len(min_distances)>0:
                dist=min_distances[0]
                dist = min_distances[0]
                cluster_1 = cluster1[0]
                cluster_2 = cluster2[0]
                cluster_1_ind = cluster1_ind[0]
                cluster_2_ind = cluster2_ind[0]

            for i in range(len(min_distances)-1):
                if min_distances[i]<dist:
                    dist=min_distances[i]
                    cluster_1 = cluster1[i]
                    cluster_2 = cluster2[i]
                    cluster_1_ind = cluster1_ind[i]
                    cluster_2_ind = cluster2_ind[i]

            if min_distance>dist:
                min_distance=dist

            del min_distances[:]
            del cluster1[:]
            del cluster2[:]
            del cluster1_ind[:]
            del cluster2_ind[:]



            if prev_min_dist != min_distance:
                if clusters[cluster_1_ind].get_label() <= clusters[cluster_2_ind].get_label():
                    if clusters[cluster_1_ind].get_size() <= clusters[cluster_2_ind].get_size():
                        merged_small_clusters.append(clusters[cluster_1_ind].get_size())
                    else:
                        merged_small_clusters.append(clusters[cluster_2_ind].get_size())
                    clusters[cluster_1_ind].merge_cluster(clusters[cluster_2_ind])

                    clusters.remove(clusters[cluster_2_ind])
                else:
                    if clusters[cluster_1_ind].get_size() <= clusters[cluster_2_ind].get_size():
                        merged_small_clusters.append(clusters[cluster_1_ind].get_size())
                    else:
                        merged_small_clusters.append(clusters[cluster_2_ind])
                    clusters[cluster_2_ind].merge_cluster(clusters[cluster_1_ind].get_size())
                    clusters.remove(clusters[cluster_1_ind])
                prev_min_dist = min_distance
        for i in range (len(clusters)):
            make_csv(clusters[i], 'cluster'+str(i)+'.csv')

        print (datetime.datetime.now ().time ())
        print("The size of smaller cluster that was merged in are: ")
        print(merged_small_clusters[-11:-1])
        draw_dendogram()

def clusters_info(cluster, file):
    print("\n\n")
    print("Information for : "+ file)
    print("Cluster label: "+str(cluster.get_label()))
    data = cluster.get_data()
    sum_avg_attribute = cluster.get_com()
    column_header = ['Milk', 'PetFood', 'Veggies', 'Cereal', 'Nuts', 'Rice', 'Meat', 'Eggs', 'Yogurt',
                     'Chips', 'Cola', 'Fruit']

    for i in range(len(sum_avg_attribute)):
        print(column_header[i]+": "+str(sum_avg_attribute[i]) + ",")

def make_csv( cluster, file_name):
    column_header = ['ID', 'Milk', 'PetFood', 'Veggies', 'Cereal', 'Nuts'	,'Rice', 'Meat', 'Eggs', 'Yogurt', 'Chips',	'Cola',	'Fruit']
    data = cluster.get_data()
    with open(file_name, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=column_header)
        writer.writeheader()
        for row in data:

            writer.writerow({column_header[0]: row[0], column_header[1]: row[1], column_header[2]: row[2],
                             column_header[3]: row[3], column_header[4]: row[4], column_header[5]: row[5],
                             column_header[6]: row[6], column_header[7]: row[7], column_header[8]: row[8],
                             column_header[9]: row[9], column_header[10]: row[10], column_header[11]: row[11]
                             ,column_header[12]: row[12]})

def draw_dendogram():
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Index')
    plt.ylabel('Distance')
    Z = linkage(data_list, method='centroid')
    dendrogram(Z, truncate_mode='lastp', p=100)
    plt.show()

def main():

    file_name ='SHOPPING_CART_v500.csv'
    with open (file_name) as f:
        count = 0
        for row in f:
            if count != 0:
                data_list.append ([int (_) for _ in row.split (',')])
            count += 1
        for row in data_list:
            id_data_map[row[0]] = row
    for key, row in id_data_map.items ():  # Initial clustering
        clusters.append (ClusterPrototype ([row], key))

    cluster_creation()

if __name__ == "__main__":
    print(mp.cpu_count())
    t1=datetime.datetime.now ()
    t2 = datetime.datetime.now ()
    elapsed =  t2  -  t1
    print (elapsed)
    main()
