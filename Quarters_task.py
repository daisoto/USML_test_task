import numpy

inp = input().split()
M, N = int(inp[0]), int(inp[1])


def get_points():

    def _get_point_and_name():

        raw_input = input().split()
        name = raw_input.pop(0)
        point = numpy.array(raw_input, dtype=float)

        return point, name

    return [tuple(_get_point_and_name()) for _ in range(M)]


def get_place():
    return numpy.array(input().split(), dtype=float)


def get_matrix(place, points):

    def metric(point, place):
        point, name = abs(place - point[0]), point[1]
        dist = numpy.sum(point)

        return (dist, name)

    matrix = []
    for point in points:
        matrix.append(metric(point, place))

    return sorted(matrix)


def print_final(matrix):
    for i in range(N):
        print(matrix[i][1], end=' ')


points = get_points()
place = get_place()
matrix = get_matrix(place, points)
print_final(matrix)
