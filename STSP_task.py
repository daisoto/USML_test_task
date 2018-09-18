import numpy
import random
import itertools


n = int(input())


def get_length(matrix, list_): # получаем длину пути, представленного списком - последовательностью вершин
    length = 0
    for i in range(len(list_) - 1):
        # в цикле вершин суммируем путь из одной в другую
        length += matrix[list_[i]][list_[i + 1]]

    return int(length)


def get_matrix():

    def floyd_warshall(matrix):
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if matrix[i][j] > matrix[i][k] + matrix[k][j] and i != j:
                        matrix[i][j] = matrix[i][k] + matrix[k][j]

    def _get_string():
        raw_input = input().split()
        string = numpy.array(raw_input, dtype=float)
        string[string == 0] = float('inf')

        return string

    matrix = numpy.array([_get_string() for _ in range(n)])

    floyd_warshall(matrix)

    return matrix


def greedy_algorithm(matrix):

    nodes = set(range(1, n))  # множество необходимых для посещения вершин
    visited = [0]  # список посещенных вершин
    currentNode = 0  # начинаем путь из "нулевой" вершины

    while True:
        if len(nodes) == 0:  # если все вершины посещены
            visited.append(0)  # идем в начальную
            return get_length(matrix, visited)

        minWeight = float('inf')  # ищем ближайшую вершину в цикле
        for node in nodes:
            if matrix[currentNode][node] < minWeight:
                minWeight = matrix[currentNode][node]
                nextNode = node  # и идем в нее

        currentNode = nextNode
        visited.append(currentNode)
        if currentNode in nodes:
            nodes.remove(currentNode)


def modified_greedy_algorithm(matrix):
    # небольшое улучшение жадног алгоритма, смысл которого заключается в том, чтобы
    # искать ближайшую вершину для двух активных вершин сразу (а не одной, как в классическом жадном)

    def get_length_of_edges(edges):
        length = 0
        for edge in edges:
            length += matrix[edge[0]][edge[1]]
        return int(length)

    nodes = set(range(1, n))  # множество необходимых для посещения вершин
    visited = []  # список пройденных ребер
    firstNode = 0  # начинаем путь из "нулевой" вершины

    minWeight = float('inf')
    for node in nodes:
        if matrix[firstNode][node] < minWeight:
            minWeight = matrix[firstNode][node]
            secondNode = node  # и идем в нее
    visited.append((firstNode, secondNode))
    nodes.remove(secondNode)

    while True:

        if len(visited) == n:  # если все вершины посещены
            return get_length_of_edges(visited)

        minWeight = float('inf')
        for node in nodes:
            if matrix[firstNode][node] < minWeight:
                minWeight = matrix[firstNode][node]
                nextNode = node  # и идем в нее
        visited.append((firstNode, nextNode))
        firstNode = nextNode
        if firstNode in nodes:
            nodes.remove(firstNode)

        if len(visited) == n:  # если все вершины посещены
            return get_length_of_edges(visited)

        minWeight = float('inf')
        for node in nodes:
            if matrix[secondNode][node] < minWeight:
                minWeight = matrix[secondNode][node]
                nextNode = node  # и идем в нее
        visited.append((secondNode, nextNode))
        secondNode = nextNode
        if secondNode in nodes:
            nodes.remove(secondNode)


def little_algorithm(matrix):

    def coercion(new_mat):
        # в ходе приведения расчитываем коэффициент приведения

        # выполняем приведение по строкам
        def row_coercion(new_mat):
            coef = 0
            for i in range(n):
                min_ele = float('inf')
                for j in range(n):
                    if new_mat[i][j] < min_ele:
                        min_ele = new_mat[i][j]
                if min_ele != 0 and min_ele != float('inf'):
                    new_mat[i] -= min_ele
                    coef += min_ele
            return coef
        # выполняем приведение по столбцам
        def column_coercion(new_mat):
            coef = 0
            for j in range(n):
                min_ele = float('inf')
                for i in range(n):
                    if new_mat[i][j] < min_ele:
                        min_ele = new_mat[i][j]
                if min_ele != 0 and min_ele != float('inf'):
                    for k in range(n):
                        new_mat[k][j] -= min_ele
                    coef += min_ele
            return coef
        a = row_coercion(new_mat) + column_coercion(new_mat)
        return a


    def get_min_zero(new_mat):
    # получаем нуль с минимальным коэффициентом

        def get_coef(new_mat, i, j):
            coef = 0
            min_ele = float('inf')
            for k in range(n):
                if new_mat[i][k] < min_ele and k != j:
                    min_ele = new_mat[i][k]
            if min_ele != float('inf') and min_ele != 0:
                coef += min_ele
            min_ele = float('inf')
            for k in range(n):
                if new_mat[k][j] < min_ele and k != i:
                    min_ele = new_mat[k][j]
            if min_ele != float('inf') and min_ele != 0:
                coef += min_ele
            return coef

        zeros = {}
        for i in range(len(new_mat[0])):
            for j in range(len(new_mat[0])):
                if new_mat[i][j] == 0:
                    zeros[(i, j)] = get_coef(new_mat, i, j)

        inverted_zeros = {v: k for k, v in zeros.items()}
        if len(inverted_zeros) == 0:
            return 0, (0, 0)
        need = sorted(inverted_zeros, reverse=True)[0]
        return need, inverted_zeros[need]


    def delete_(new_mat, row, column):
    # "удаляем" столбец, присваивая всем элементам строки и столбца неугодного элемента значения беспонечности
        for i in range(n):
            for j in range(n):
                if i == row or j == column:
                    new_mat[i][j] = float('inf')


    def step(new_mat, border, edges):
        # на каждом шаге алгоритма увеличиваем границу и включаем ребка в список пути

        if len(edges) == n and border != 0:
            return int(border)

        border += coercion(new_mat)
        weight, edge = get_min_zero(new_mat)

        border_1 = border + coercion(new_mat)
        border_2 = border + weight

        if border_2 < border_1:
            new_mat[edge[0]][edge[1]] = float('inf')  # исключаем ребро с макс. штрафом
            return step(new_mat, border_2, edges)
        else:
            delete_(new_mat, edge[0], edge[1])
            new_mat[edge[1]][edge[0]] = float('inf')
            edges.append(edge)
            return step(new_mat, border_1, edges)

    a = numpy.array(matrix)
    return step(a, 0, [])


def genetic_algorithm(matrix):
    def crossover(parent1, parent2):
        # операция кроссинговера
        child = [None for i in range(len(parent1))]
        # случайно выбираем две позиции
        firstPos = int(random.random() * len(parent1))
        secondPos = int(random.random() * len(parent1))
        for i in range(len(parent1)):
            if firstPos < secondPos and i > firstPos and i < secondPos:
                child[i] = parent1[i]
            elif firstPos > secondPos:
                if not (i < firstPos and i > secondPos):
                    child[i] = parent1[i]
        for i in range(len(parent1)):
            if not parent2[i] in child:
                for j in range(len(parent1)):
                    if child[j] == None:
                        child[j] = parent2[i]
                        break
        return child

    def mutate(tour):
       # операция мутации - с заданной вероятностью
        # меняем узлы маршрута местами
        #mutationProb = 1/(len(bin(max(tour)).lstrip('0b')))
        for Pos1 in range(1, len(tour)):
            if random.random() < mutationProb:
                Pos2 = int(len(tour) * random.random())
                tour[Pos1], tour[Pos2] = tour[Pos2], tour[Pos1]

    def fitness(matrix, tour):
        # фитнесс-функция маршрута
        # обратно пропорциональна длине пути
        length = 0
        for i in range(len(tour)-1):
            # в цикле вершин суммируем путь из одной в другую
            length += matrix[tour[i]][tour[i+1]]
        # добавляем путь из первой вершины в последнюю
        length += matrix[tour[len(tour)-1]][tour[0]]
        return 1/length

    def tournamentSelection(pop):
        # турнирная селекция
        # отбираем особь с наибольшей пригодностью
        # используем случайное "заполнение" для получения различных
        # результатов - с целью получения различных родителей в будущем
        tournament = []
        for i in range(len(pop)):
            randomId = int(random.random() * len(pop))
            tournament.append(pop[randomId])
        fittest = tournament[0]
        for i in range(len(pop)):
            if fitness(matrix, fittest) <= fitness(matrix, tournament[i]):
                fittest = tournament[i]
        return fittest

    def startPopulation(matrix):
        pop = []
        for i in range(populationSize):
            newTour = list(range(1, len(matrix[0])))
            random.shuffle(newTour)
            newTour.insert(0, 0)
            pop.append(newTour)
        return pop

    def evolvePopulation(pop):
        newPopulation = []
        for i in range(populationSize):
            parent1 = tournamentSelection(pop)
            parent2 = tournamentSelection(pop)
            child = crossover(parent1, parent2)
            newPopulation.append(child)
        for i in range(len(newPopulation)):
            mutate(newPopulation[i])
        return newPopulation

    def finalTour(matrix, pop):
        fittest = pop[0]
        length = 0
        for i in range(len(pop)):
            if fitness(matrix, fittest) <= fitness(matrix, pop[i]):
                fittest = pop[i]
        for i in range(len(fittest)-1):
            # в цикле вершин суммируем путь из одной в другую
            length += matrix[fittest[i]][fittest[i+1]]
        # добавляем путь из первой вершины в последнюю
        length += matrix[fittest[len(fittest)-1]][fittest[0]]
        return int(length)

    mutationProb = 0.05
    populationSize = len(matrix[0]) * 5
    pop = startPopulation(matrix)
    for i in range(0, int(populationSize * 5)):
        evolvePopulation(pop)
    return finalTour(matrix, pop)


def brute(matrix):
    # решаем задачу методом брутфорса
    # создаем (n-1)! решений и ищем наименьшее
    a = numpy.array(list(itertools.permutations(range(1, n))))
    m = []

    for i in range(len(a)):
        b = [0]
        for j in range(n-1):
            b.append(a[i][j])
        b.append(0)
        m.append(get_length(matrix, b))
    return min(m)


matrix = get_matrix()
print('Литтла -', little_algorithm(matrix))
print('Жадный -', greedy_algorithm(matrix))
print('Улучшенный жадный -', modified_greedy_algorithm(matrix))
print('Брутфорс -', brute(matrix))
print('Генетический -', brute(matrix))
