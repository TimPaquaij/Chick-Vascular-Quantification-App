def get_line(x1, y1, x2, y2): # Script for  extracting every point of a certain line
    points = []
    issteep = abs(y2-y1) > abs(x2-x1)
    if issteep: # If the y distance is  bigger then x the coirdinates will be switch. 
        x1, y1 = y1, x1 
        x2, y2 = y2, x2
    rev = False
    if x1 > x2: #If the first point is bigger then sceccond they will be switched
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    deltax = x2 - x1 #Determine horizontal distance
    deltay = abs(y2-y1) #Determine vertical distance
    error = int(deltax / 2) #determine if horizontal distance is even
    y = y1 
    ystep = None
    if y1 < y2: #determine if vertcal distance if positive or negative
        ystep = 1
    else:
        ystep = -1
    for x in range(x1, x2 + 1):  #Goes for every x value from starting point to end point with step one
        if issteep:
            points.append((y, x)) # appends point in list is issteep
        else:
            points.append((x, y))    # appends point in list is not issteep
        error -= deltay             # if vertical height is  much higher then horizontal error stays negative this is for detection a horizontal line
        if error < 0:               
            y += ystep              #and y will increase or decrease
            error += deltax
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()
    return points

