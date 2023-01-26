def kaplan(survtimes):
    h_coords = []
    v_coords = []
    lost = []
    y = 1
    for i in survtimes:
        if i[1] != 1:
            lost.append([i[0], y])
        else:
            h_coords.append([i[0], y])
            y = len(survtimes[survtimes.index(i) + 1:]) / len(survtimes[survtimes.index(i):])
            v_coords.append([i[0], h_coords[-1][-1], y])
            break
    newsurv = survtimes[survtimes.index(i) + 1:]
    while len(newsurv) > 0:
        newsurv, y, h_coords, v_coords, lost = loop(newsurv, y, h_coords, v_coords, lost)
    return h_coords, v_coords, lost

def loop(newsurv, y, h_coords, v_coords, lost):
    for j in newsurv:
        if j[1] != 1:
            lost.append([j[0], y])
        else:
            h_coords.append([j[0], y])
            y = y * len(newsurv[newsurv.index(j) + 1:]) / float(len(newsurv[newsurv.index(j):]))
            v_coords.append([j[0], h_coords[-1][-1], y])
            break
    newsurv = newsurv[newsurv.index(j)+1:]
    return newsurv, y, h_coords, v_coords, lost
