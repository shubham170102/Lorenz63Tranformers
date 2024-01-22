def matrix_multiplication(a, b):
    column_a = len(a[0])
    row_a = len(a)
    column_b = len(b[0])
    row_b = len(b)

    result_matrix = [[j for j in range(column_b)] for i in range(row_a)]

    if column_a == row_b:
        for x in range(row_a):
            for y in range(column_b):
                total_sum = 0
                for k in range(column_a):
                    total_sum += a[x][k] * b[k][y]
                result_matrix[x][y] = total_sum
        return result_matrix
    else:
        print("Error! the number of columns of the first matrix needs to be equal to the number of rows in the second "
              "matrix")
        return None
