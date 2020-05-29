import matplotlib.pyplot as plt
import math
import time
from queue import Queue 
import numpy as np
import random
# --------------------------------------------------------------------------------
# Segment an image:
# Returns a color image representing the segmentation.
#
# Inputs:
#           in_image: image to segment.
#           sigma: to smooth the image.
#           k: constant for threshold function.
#           min_size: minimum component size (enforced by post-processing stage).
#
# Returns:
#           num_ccs: number of connected components in the segmentation.
# --------------------------------------------------------------------------------
def segment(in_image, k, min_size):
    start_time = time.time()
    assert(len(in_image.shape) == 2)
    height, width = in_image.shape
    print("Height:  " + str(height))
    print("Width:   " + str(width))
    
    # build graph
    # 8 neighbor
    workQueue = Queue()
    workQueue.put((0,0))
    visitedRecord = np.zeros
    visitedPoint = np.zeros_like(in_image)
    visitedPoint[0, 0] = 1

    
    validateX = lambda x : (x > -1 and x < width)
    validateY = lambda y : (y > -1 and y < height)
    seq2dCoor = lambda x, y :  y * width + x
    num = 0
    edges_size = width * height * 4
    edges = np.zeros(shape=(edges_size, 3), dtype=object)
    for y in range(height):
        for x in range(width):
            if x < width - 1:
                edges[num, 0] = int(seq2dCoor(x, y))
                edges[num, 1] = int(seq2dCoor(x+1, y))
                edges[num, 2] = diff(in_image, x, y, x + 1, y)
                num += 1
            if y < height - 1:
                edges[num, 0] = int(seq2dCoor(x, y))
                edges[num, 1] = int(seq2dCoor(x, y+1))
                edges[num, 2] = diff(in_image, x, y, x, y + 1)
                num += 1

            if (x < width - 1) and (y < height - 2):
                edges[num, 0] = int(seq2dCoor(x, y))
                edges[num, 1] = int(seq2dCoor(x+1, y+1))
                edges[num, 2] = diff(in_image, x, y, x + 1, y + 1)
                num += 1

            if (x < width - 1) and (y > 0):
                edges[num, 0] = int(seq2dCoor(x, y))
                edges[num, 1] = int(seq2dCoor(x+1, y-1))
                edges[num, 2] = diff(in_image, x, y, x + 1, y - 1)
                num += 1
    # Segment
    vertexNum = width * height
    edgeNum = num
    u = segment_graph(vertexNum, edgeNum, edges, k)

    # post process small components
    for i in range(edgeNum):
        a = u.find(edges[i, 0])
        b = u.find(edges[i, 1])
        if (a != b) and ((u.size(a) < min_size) or (u.size(b) < min_size)):
            u.join(a, b)

    num_cc = u.num_sets()
    output = np.zeros(shape=(height, width, 3))

    # pick random colors for each component
    colors = np.zeros(shape=(height * width, 3), dtype=np.float)
    for i in range(height * width):
        colors[i, :] = random_rgb()

    for y in range(height):
        for x in range(width):
            comp = u.find(y * width + x)
            output[y, x, :] = colors[comp, :]
            #print("pick color:{}".format(colors[comp, :]))

    elapsed_time = time.time() - start_time
    print(
        "Execution time: " + str(int(elapsed_time / 60)) + " minute(s) and " + str(
            int(elapsed_time % 60)) + " seconds")

    # displaying the result
    print("blocks: {}".format(u.num_sets()))
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    plt.imshow(in_image, cmap='gray')
    a.set_title('Original Image')
    a = fig.add_subplot(1, 2, 2)
    plt.imshow(output/255)
    a.set_title('Segmented Image')
    plt.show()
    return u, output

def plotU():
    pass


# disjoint-set forests using union-by-rank and path compression (sort of).
class universe:
    def __init__(self, n_elements):
        self.num = n_elements
        self.elts = np.empty(shape=(n_elements, 3), dtype=int)
        for i in range(n_elements):
            self.elts[i, 0] = 0  # rank
            self.elts[i, 1] = 1  # size
            self.elts[i, 2] = i  # p

    def size(self, x):
        return self.elts[x, 1]

    def num_sets(self):
        return self.num

    def find(self, x):
        y = int(x)
        x = int(x)
        while y != self.elts[y, 2]:
            y = self.elts[y, 2]
        try:
            self.elts[x, 2] = y
        except Exception as e:
            print("now x: {}  type:{}".format(x, type(x)))
            raise Exception("Error")
        return y

    def join(self, x, y):
        # x = int(x)
        # y = int(y)
        if self.elts[x, 0] > self.elts[y, 0]:
            self.elts[y, 2] = x
            self.elts[x, 1] += self.elts[y, 1]
        else:
            self.elts[x, 2] = y
            self.elts[y, 1] += self.elts[x, 1]
            if self.elts[x, 0] == self.elts[y, 0]:
                self.elts[y, 0] += 1
        self.num -= 1


# ---------------------------------------------------------
# Segment a graph:
# Returns a disjoint-set forest representing the segmentation.
#
# Inputs:
#           num_vertices: number of vertices in graph.
#           num_edges: number of edges in graph
#           edges: array of edges.
#           c: constant for threshold function.
#
# Output:
#           a disjoint-set forest representing the segmentation.
# ------------------------------------------------------------
def segment_graph(num_vertices, num_edges, edges, c):
    # sort edges by weight (3rd column)
    edges[0:num_edges, :] = edges[edges[0:num_edges, 2].argsort()]
    # make a disjoint-set forest
    u = universe(num_vertices)
    # init thresholds
    threshold = np.zeros(shape=num_vertices, dtype=float)
    for i in range(num_vertices):
        threshold[i] = get_threshold(1, c)

    # for each edge, in non-decreasing weight order...
    for i in range(num_edges):
        pedge = edges[i, :]
        edgesSrc = int(pedge[0])
        edgesDst = int(pedge[1]) 
        # components connected by this edge
        a = u.find(edgesSrc)
        b = u.find(edgesDst)
        if a != b:
            if (pedge[2] <= threshold[a]) and (pedge[2] <= threshold[b]):
                u.join(a, b)
                a = u.find(a)
                threshold[a] = pedge[2] + get_threshold(u.size(a), c)

    return u


def get_threshold(size, c):
    return c / size


# returns square of a number
def square(value):
    return value * value


# randomly creates RGB
def random_rgb():
    rgb = np.zeros(3, dtype=int)
    rgb[0] = random.randint(0, 255)
    rgb[1] = random.randint(0, 255)
    rgb[2] = random.randint(0, 255)
    return rgb


# dissimilarity measure between pixels
def diff(img, x1, y1, x2, y2):
    result = math.sqrt(square(img[y1, x1] - img[y2, x2]))
    return result


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

if __name__ == "__main__":
    sigma = 0.5
    k = 500
    min_ = 50
    input_path = "paris.jpg"

    # Loading the image
    input_image = plt.imread(input_path)
    input_image = rgb2gray(input_image)
    print("Loading is done.")
    print("processing...")
    segment(input_image, k, min_)
