def remove_object(img, mask):
    n = img.shape[0]
    m = img.shape[1]
    for i in range(n):
        for j in range(m):
            if mask[i][j]:
                sum_neighbours = [0, 0, 0]
                num_neighbours = 0
                for ki in range(max(0, i - 2), min(n, i + 3)):
                    for kj in range(max(0, j - 2), min(m, j + 3)):
                        sum_neighbours += (not mask[ki][kj]) * img[ki][kj]
                        num_neighbours += not mask[ki][kj]

                img[i][j] = [s_i / num_neighbours for s_i in sum_neighbours]
                mask[i][j] = False
