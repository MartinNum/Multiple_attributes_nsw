from n2 import HnswIndex
import random

f = 3
t = HnswIndex(f, "L2")  # HnswIndex(f, "L2 or angular")
for i in range(20):
    v = [random.gauss(0, 1) for z in range(f)]
    t.add_data(v)

t.build(m=5, max_m0=10, n_threads=4)
t.save('test.n2')

u = HnswIndex(f, "L2")
u.load('test.n2')

search_id = 1
k = 10
neighbor_ids = u.search_by_id(search_id, k)
print(
    "[search_by_id]: Nearest neighborhoods of id {}: {}".format(
        search_id,
        neighbor_ids))

example_vector_query = [random.gauss(0, 1) for z in range(f)]
example_vector_art = ['高校', '企业']
nns = u.search_by_vector(example_vector_query, k, include_distances=True)
print(
    "[search_by_vector]: Nearest neighborhoods of vector {}: {}".format(
        example_vector_query,
        nns))
