import time
import faiss
import numpy as np

num_vectors = 100_000
dimension = 768

vectors = np.random.random((num_vectors, dimension)).astype("float32")
query_vector = np.random.random((1, dimension)).astype("float32")

quantizer = faiss.IndexFlatL2(dimension)
flatL2_index = faiss.IndexFlatL2(dimension)

ivf_index = faiss.IndexIVFFlat(quantizer, dimension, 100, faiss.METRIC_L2)

hnsw_index = faiss.IndexHNSWFlat(dimension, 32)

ivf_index.train(vectors)

flatL2_index.add(vectors)
ivf_index.add(vectors)
hnsw_index.add(vectors)

ivf_index.nprobe = 8
hnsw_index.hnsw.efSearch = 128

start = time.time()
Ldistances, Lindexes = flatL2_index.search(query_vector, 5)
L2_time = time.time() - start

start = time.time()
Idistances, Iindexes = ivf_index.search(query_vector, 5)
ivf_time = time.time() - start

start = time.time()
Hdistances, Hindexes = hnsw_index.search(query_vector, 5)
hnsw_time = time.time() - start


print("\n Summary of FlatIndexL2")
print(Lindexes, Ldistances)
print(f"Time = {L2_time} \n")

print("\n Summary of IVF")
print(Iindexes, Idistances)
print(f"Time = {ivf_time} \n")

print("\n Summary of HNSW")
print(Hindexes, Hdistances)
print(f"Time = {hnsw_time}")
