def is_inside(a, b):

    # calculate overlapping length
    x_overlap = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    y_overlap = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))

    a_area = a[2] * a[3]
    b_area = b[2] * b[3]

    # calculate the overlap rectangle area
    overlaps = x_overlap * y_overlap

    if overlaps > 0:
        if overlaps == a_area:
            return -1  # B contains a A
        elif overlaps == b_area:
            return 1  # A contains a B
        else:
            return 0  # A and B partially overlap
    else:
        return 0  # A and B do not overlap


def rect_grouper(rect_list):

    i = 0

    if len(rect_list) > 1:
        j = i+1
    else:
        return rect_list

    while 1:
        while 1:
            # modify the rectangle list using the comparison results
            result = is_inside(rect_list[i], rect_list[j])
            if result == 1:
                # i contains j
                rect_list.pop(j)  # remove j
                j -= 1
            elif result == -1:
                # j contains i
                rect_list.pop(i)  # remove i
                i -= 1
                break

            j += 1

            if j == len(rect_list):
                break
        i += 1
        j = i + 1

        if i+1 >= len(rect_list):
            break

    return rect_list

