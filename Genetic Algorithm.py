import random
import cv2
import numpy as np
from operator import itemgetter
import copy

# import matplotlib.pyplot as plt

# read the input image filename from user input
inputFileName = input(
    "Input file name with the extension (or file path if the input image is not in the same folder): ")

# origin is the given image. Variable is used for comparing
origin = cv2.imread(inputFileName)
height, width, channels = origin.shape

# Resize ratio
resizeRatio = 8
resizedHeight = int(height / resizeRatio)
resizedWidth = int(width / resizeRatio)

# Resize the origin image for faster calculation
resizedOrigin = cv2.resize(origin, (resizedHeight, resizedWidth), interpolation=cv2.INTER_AREA)

# Transparency limits for circles
minTran = 0.25
maxTran = 0.75

# Limits as constants
shapeLimit = 4000
genLimit = 5000
populationLimit = 512

# An all white image
blank = np.ones((resizedHeight, resizedWidth, channels), np.uint8) * 255
blankFullSize = np.ones((height, width, channels), np.uint8) * 255

# read all colors of the image
# originColors global colors list of color tuples
originColors = []
for i in range(0, height):
    for j in range(0, width):
        b, g, r = (origin[i, j])
        originColors.append((b, g, r))


# def circle to create a list of attributes for cv2.circle function
# Attributes: center 0 as a tuple (x,y), radius 1, color 2 as a (b,g,r) tuple, transparency 3
def circle():
    color = random.choice(originColors)
    return [(random.randint(0, resizedHeight), random.randint(0, resizedWidth)),
            random.randint(1, int(resizedHeight / 4)),
            (int(color[0]), int(color[1]), int(color[2])), random.uniform(minTran, maxTran)]


# Calculating fitness by counting the difference with the im by each pixel
# Fitness is the Mean Square Error, therefore the less is fitness, the closer to the origin is the given image
# Use resized images for optimized mean square error calculation
def calculateFitness(image):
    resized = cv2.resize(image, (int(height / 8), int(width / 8)), interpolation=cv2.INTER_AREA)
    fitness = np.sum((resizedOrigin.astype(np.float32) - resized.astype(np.float32)) ** 2)
    fitness /= float(resizedOrigin.shape[0] * resized.shape[1])
    return fitness


# Draw with given circles
def makeSketch(im, circles):
    # background is created with given circles and added on top of given image
    for cir in circles:
        background = im.copy()
        background = cv2.circle(background, cir[0], cir[1], cir[2], -1)
        im = cv2.addWeighted(background, cir[3], im, 1 - cir[3], 0)
    return im


# Population is a collection of sketches.
class Population:
    currSketch = blank
    currCircles = []

    def __init__(self, newIms):
        # list storing all sketches in the 512 sized population
        self.sketches = []
        if not newIms:
            # Preparing tools to draw with: the canvas, circles
            self.currSketch = blank
            self.sketches = []
        else:
            self.sketches = newIms

    # Finding the images with the best (least) fitness.
    def bestFit(self):
        bestFitIm = min(self.sketches, key=itemgetter(2))
        if calculateFitness(self.currSketch) > bestFitIm[2]:
            # Append new gen to curr
            self.currCircles += bestFitIm[0]
            self.currSketch = makeSketch(blank, self.currCircles)
        return self.currSketch

    # Allowing a mutation to occur, changing genes
    def mutation(self):
        newPopulation = []
        for p in range(populationLimit):
            mutationSubject = [[], Generation, 0]
            # Two new circles mutated!
            for a in range(2):
                mutationSubject[0].append(circle())
            pic = makeSketch(self.currSketch, mutationSubject[0])
            mutationSubject[2] = calculateFitness(pic)
            # Append the results of mutation
            newPopulation.append(mutationSubject)
        self.sketches = newPopulation
        return newPopulation

    # Crossover between two parents, randomly choosing their attributes
    def crossover(self):
        self.bestFit()
        self.mutation()
        return self

    def resizedCircles(self):
        resizedCircles = []
        for c in self.currCircles:
            c = [(c[0][0] * resizeRatio, c[0][1] * resizeRatio), c[1] * resizeRatio, c[2], c[3]]
            resizedCircles.append(c)
        return resizedCircles


# Generation 0
Generation = 0

population = Population([])
outputFullSize = Population([])
population.mutation()

# Fitness list for graph plotting, x axis is the number of the current generation, y axis is ths corresponding fitness
# fitness = []
# x = np.arange(0, genLimit)

while True:
    # Making new gen
    population = population.crossover()
    # fitness.append(calculateFitness(population.currSketch))
    # saving every 100th generation sample to disk to see progress
    if (Generation % 100) == 0:
        outputFullSize = copy.deepcopy(population)
        outputFullSize.currSketch = cv2.resize(outputFullSize.currSketch, (height, width), interpolation=cv2.INTER_AREA)
        outputFullSize.currCircles = outputFullSize.resizedCircles()
        filename = 'Generation ' + str(Generation) + ' sample with ' + str(
            calculateFitness(population.currSketch)) + ' fitness.png '
        cv2.imwrite(filename, makeSketch(blankFullSize, outputFullSize.currCircles))
    print("Made gen ", Generation)
    Generation += 1
    # End loop condition
    if Generation > genLimit:
        break

# # Plotting fitness graph for the report
# y = fitness
# plt.plot(x, y)
# plt.xlabel('Number of Generation')
# plt.ylabel('Fitness')
# plt.title('Dependence of Fitness on Generation')
# plt.show()
