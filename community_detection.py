from pyspark import SparkContext
import pyspark
import itertools
import sys
import time

def flatten_user_pair(i):
    for pair in itertools.combinations(i[1], 2):
        yield (tuple(sorted(pair)), 1)
        
def get_shortest_path(user, neighbors_map, result):
    queue = [user, None]
    d = dict() #key = child, value = (parent & #shortest path) for all parents
    path_count = dict() # key = node, value = #shortest path
    visited = set(user)
    while queue:
        curr_user = queue.pop(0)
        if curr_user == None:
            if d:
                result.append(list(d.items()))
                for key in d:
                    visited.add(key)
                    queue.append(key)
            queue.append(None)
            d = dict()
            curr_user = queue.pop(0)
        if curr_user in neighbors_map:
            if curr_user not in path_count:
                curr_user = (curr_user, 1)
            else:
                curr_user = (curr_user, path_count[curr_user])
            for neighbor in neighbors_map[curr_user[0]]:
                if neighbor not in visited:
                    if neighbor in d:
                        d[neighbor] += (curr_user,)
                    else:
                        d[neighbor] = (curr_user,)
                    path_count[neighbor] = path_count.get(neighbor, 0) + curr_user[1]
    return (result, path_count)

def get_betweenness(user, neighbors_map):
    shortest_path, path_count = get_shortest_path(user, neighbors_map, [])
    d = {} #key = parent node, value = sum of betweenness of children
    while shortest_path:
        curr_level = shortest_path.pop(-1)
        for t in curr_level:
            child = t[0]
            for parent in t[1]:
                prev = 0
                if child in d:
                    prev += d[child]
                btn = (prev + 1) * (parent[1] / path_count[child])
                d[parent[0]] = d.get(parent[0], 0) + btn
                yield (tuple(sorted((parent[0], child))), btn)

def get_communities(users, neighbors_map):
    communities = []
    user_visited = set()
    for user in users:
        if user not in user_visited:
            queue = [user]
            visited = set([user])
            user_visited.add(user)
            while queue:
                curr_user = queue.pop(0)
                if curr_user in neighbors_map:
                    for neighbor in neighbors_map[curr_user]:
                        if neighbor not in visited:
                            user_visited.add(neighbor)
                            visited.add(neighbor)
                            queue.append(neighbor)
            communities.append(tuple(sorted(visited)))
    return communities            


def get_modularity(pair, A, m, degrees_map):
    if pair in A:
        Aij = 1
    else:
        Aij = 0
    useri, userj = pair
    ki, kj = degrees_map[useri], degrees_map[userj]
    return Aij - ((ki * kj) / (2 * m))

def get_best_communities(user1, user2, users, neighbors_map, degrees_map, m, A):

    max_modularity = -1
    user_count = users.count()
    pre_modularity = -1

    while True:

        # delete neighbors in map
        neighbors_map[user1].remove(user2)
        neighbors_map[user2].remove(user1)

        # find communities and calculate modularity
        communities = get_communities(users.collect(), neighbors_map)
        modularity = 0

        for community in communities:
            for pair in itertools.combinations(community, 2):
                modularity += get_modularity(pair, A, m, degrees_map)
        modularity /= (2 * m)

        if modularity > max_modularity:
            max_modularity = modularity
            best_communities = communities

        if len(communities) == user_count:
            break

        # calculate new betweenness
        new_betweenness = users \
                    .flatMap(lambda user: get_betweenness(user, neighbors_map)) \
                    .reduceByKey(lambda x, y: x + y) \
                    .mapValues(lambda value: value / 2) \
                    .sortBy(lambda x: (-x[1], x[0]))
        
        # cut the edge between user1 and user2 (max_betweenness) 
        user1, user2 = new_betweenness.take(1)[0][0]

    return best_communities


def main():
    start = time.time()

    threshold = int(sys.argv[1])
    input_file = sys.argv[2]
    btn_output_file = sys.argv[3]
    com_output_file = sys.argv[4]

    sc = SparkContext()
    sc.setLogLevel("ERROR")

    edges = sc.textFile(input_file) \
            .map(lambda line: line.split(',')) \
            .map(lambda line: (line[1], line[0])) \
            .groupByKey() \
            .filter(lambda line: len(line[1]) >= 2) \
            .flatMap(flatten_user_pair) \
            .reduceByKey(lambda x, y: x + y) \
            .filter(lambda line: line[1] >= threshold)

    users = edges \
            .flatMap(lambda line: [line[0][0], line[0][1]]) \
            .distinct()
    edges = edges \
            .flatMap(lambda line: [line[0], (line[0][1], line[0][0])]) 

    user_neighbors = edges.groupByKey()
    neighbors_map = dict(user_neighbors \
                        .mapValues(set) \
                        .collect())

    # Betweenness Caecueation
    betweenness = users \
                .flatMap(lambda user: get_betweenness(user, neighbors_map)) \
                .reduceByKey(lambda x, y: x + y) \
                .mapValues(lambda value: value / 2) \
                .sortBy(lambda x: (-x[1], x[0])) 


    # Community Detection
    degrees_map = dict(user_neighbors \
                .mapValues(len) \
                .collect())

    A = set(betweenness.map(lambda line: line[0]).collect())
    m = betweenness.count()
    user1, user2 = betweenness.take(1)[0][0]

    best_communities = get_best_communities(user1, user2, users, neighbors_map, degrees_map, m, A)

    result_communities = sorted(best_communities, key = lambda community: (len(community), community))

    # output results
    result_betweenness = betweenness.collect()
    fh = open(btn_output_file, 'w')
    for line in result_betweenness:
        fh.write(str(line).rstrip(')').replace('(', '', 1))
        fh.write('\n')
    fh.close()

    fh = open(com_output_file, 'w')
    for line in result_communities:
        fh.write(str(line).replace('(', '').replace(',)', '').replace(')', ''))
        fh.write('\n')
    fh.close()

    print("Duration: %s" % (time.time() - start))

if __name__ == "__main__":
    main()