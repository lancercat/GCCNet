import numpy as np;
import math;
import cv2;

def fit_line(p1, p2):
    # fit a line ax+by+c = 0
    if np.abs(p1[0]-p1[1]) < 1e-3:
        return [1., 0., -p1[0]]
    else:
        [k, b] = np.polyfit(p1, p2, deg=1)
        return [k, -1., b]

def l2(p):
    return np.sqrt(p[0] * p[0]+ p[1]*p[1]);

def neko_point_dist_to_line(p1, p2, p3):
    # compute the distance from p3 to p1-p2
    d=p2-p1;
    return abs(d[1]*p3[0]-d[0]*p3[1]+p2[0]*p1[1]-p2[1]*p1[0]) / (l2(d)+0.000009);

def neko_points_to_line(p1,p2,p3s):
    d = p2 - p1;
    l2d=l2(d)+0.00000009;
    p3xs = p3s[:, 0];
    p3ys = p3s[:, 1];
    return abs(d[1] * p3xs - d[0] * p3ys + p2[0] * p1[1] - p2[1] * p1[0]) / l2d;


def neko_points_to_each_line(p1s,p2s,p3s):
    ds = p2s - p1s;
    dx = ds[:, 0];
    dy = ds[:, 1];
    p1xs = p1s[:, 0];
    p1ys = p1s[:, 1];
    p2xs = p2s[:, 0];
    p2ys = p2s[:, 1];
    l2ds = (np.sqrt(dx * dx + dy * dy) + 0.0000009);
    magic_terms=p2xs * p1ys - p2ys * p1xs;
    magic_terms/=l2ds;
    dy/=l2ds;
    dx/=l2ds;
    p3xs = p3s[:, 0];
    p3ys = p3s[:, 1];
    t1=np.matmul(np.expand_dims(p3xs,1),np.expand_dims(dy,0) );
    t2=np.matmul( np.expand_dims(p3ys,1),np.expand_dims(dx,0))
    return np.abs(
        ((t1-t2)[:] + magic_terms));


def neko_points_dist_to_lines(p1s,p2s,p3s):
    ds = p2s - p1s;
    dx = ds[:,0];
    dy=  ds[:,1];
    p1xs=p1s[:,0];
    p1ys=p1s[:,1];
    p2xs=p2s[:,0];
    p2ys=p2s[:,1];
    p3xs = p3s[:, 0];
    p3ys = p3s[:, 1];
    l2ds = np.sqrt(dx * dx + dy* dy)+0.0000009;
    return np.abs(dy * p3xs - dx * p3ys + p2xs * p1ys - p2ys * p1xs) / l2ds;


def point_dist_to_line(p1, p2, p3):
    # compute the distance from p3 to p1-p2
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / (np.linalg.norm(p2 - p1)+0.000009);


def line_cross_point(line1, line2):
    # line1 0= ax+by+c, compute the cross point of line1 and line2
    if line1[0] != 0 and line1[0] == line2[0]:
        print('Cross point does not exist')
        return None
    if line1[0] == 0 and line2[0] == 0:
        print('Cross point does not exist')
        return None
    if line1[1] == 0:
        x = -line1[2]
        y = line2[0] * x + line2[2]
    elif line2[1] == 0:
        x = -line2[2]
        y = line1[0] * x + line1[2]
    else:
        k1, _, b1 = line1
        k2, _, b2 = line2
        x = -(b1-b2)/(k1-k2)
        y = k1*x + b1
    return np.array([x, y], dtype=np.float32)





def line_verticle(line, point):
    # get the verticle line from line across point
    if line[1] == 0:
        verticle = [0, -1, point[1]]
    else:
        if line[0] == 0:
            verticle = [1, 0, -point[0]]
        else:
            verticle = [-1./line[0], -1, point[1] - (-1/line[0] * point[0])]
    return verticle



def sort_rectangle(poly):
    # sort the four coordinates of the polygon, points in poly should be sorted clockwise
    # First find the lowest point
    p_lowest = np.argmax(poly[:, 1])
    if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
        # 底边平行于X轴, 那么p0为左上角
        p0_index = np.argmin(np.sum(poly, axis=1))
        p1_index = (p0_index + 1) % 4
        p2_index = (p0_index + 2) % 4
        p3_index = (p0_index + 3) % 4
        return poly[[p0_index, p1_index, p2_index, p3_index]], 0.
    else:
        # 找到最低点右边的点
        p_lowest_right = (p_lowest - 1) % 4
        p_lowest_left = (p_lowest + 1) % 4
        magic=poly[p_lowest][0] - poly[p_lowest_right][0];
        if(magic ==0):
            magic=1e-9;
        angle = np.arctan(-(poly[p_lowest][1] - poly[p_lowest_right][1])/(magic))
        # assert angle > 0
        if angle <= 0:
            print(angle, poly[p_lowest], poly[p_lowest_right])
        if angle/np.pi * 180 > 45:
            # 这个点为p2
            p2_index = p_lowest
            p1_index = (p2_index - 1) % 4
            p0_index = (p2_index - 2) % 4
            p3_index = (p2_index + 1) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], -(np.pi/2 - angle)
        else:
            # 这个点为p3
            p3_index = p_lowest
            p0_index = (p3_index + 1) % 4
            p1_index = (p3_index + 2) % 4
            p2_index = (p3_index + 3) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], angle



def check_and_validate_polys(polys, tags, xxx_todo_changeme):
    '''
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    :param polys:
    :param tags:
    :return:
    '''
    (h, w) = xxx_todo_changeme
    if polys.shape[0] == 0:
        return polys
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w-1)
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h-1)

    validated_polys = []
    validated_tags = []
    for poly, tag in zip(polys, tags):
        p_area = polygon_area(poly)
        if abs(p_area) < 1:
            # print(poly)
            print('invalid poly')
            continue
        if p_area > 0:
            print('poly in wrong direction')
            poly = poly[(0, 3, 2, 1), :]
        validated_polys.append(poly)
        validated_tags.append(tag)
    return np.array(validated_polys), np.array(validated_tags)


##############fissshhhhhyyyyy.
def polygon_area(poly):
    '''
    compute area of a polygon
    :param poly:
    :return:
    '''
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge)/2.


def shrink_poly(poly, r, shrink_ratio=0.3):
    '''
    fit a poly inside the origin poly, maybe bugs here...
    used for generate the score map
    :param poly: the text poly
    :param r: r in the paper
    :return: the shrinked poly
    '''
    # shrink ratio
    R = shrink_ratio
    # find the longer pair
    if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) > \
                    np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
        # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
        ## p0, p3
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
    else:
        ## p0, p3
        # print poly
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
    return poly

def welf_rrect_to_4pts(center, size, angle):
    _angle = angle
    b = np.cos(_angle) * 0.5
    a = np.sin(_angle) * 0.5

    pt = [0 for _ in range(8)]
    pt[0] = center[0] - a * size[1] - b * size[0]
    pt[1] = center[1] + b * size[1] - a * size[0]
    pt[2] = center[0] + a * size[1] - b * size[0]
    pt[3] = center[1] - b * size[1] - a * size[0]
    pt[4] = 2 * center[0] - pt[0]
    pt[5] = 2 * center[1] - pt[1]
    pt[6] = 2 * center[0] - pt[2]
    pt[7] = 2 * center[1] - pt[3];

    return sort_rectangle(np.array(pt, dtype=np.float).reshape(4, 2))[0];

def kot_vertice_distance(a,b):
    d=a-b;
    d*=d;
    d=np.sum(d,-1);
    d=np.sqrt(d);
    return np.sum(d);

# You need to setup master HDD and slave HDD with jumpers back in 90's
def kot_min_l2_match(master,slave):
    kps=master.shape[0]
    cords=slave.copy();
    best=slave;
    best_val=kot_vertice_distance(master,cords);
    for i in range(1,kps):
        cords=np.roll(cords,1,0);
        val=kot_vertice_distance(master,cords);
        if(val<best_val):
            best_val=val;
            best=cords.copy();
    return best;


def kot_min_area_quadrilateral(points):
    rrect=cv2.minAreaRect(points.astype(np.float32));
    return sort_rectangle(cv2.boxPoints(rrect));

    # angle/=180;
    # angle*=np.pi;
    #
    # return

def welf_crop_textline(im, corners):
    rect = cv2.minAreaRect(corners.reshape((-1, 2)).astype("float32"))
    angle = rect[2]
    rect_size = rect[1]

    if rect_size[0] > rect_size[1]:
        if (angle < -75):
            angle += 90
            s1 = rect_size[1]
            s0 = rect_size[0]
            rect_size = (s1, s0)
    elif rect_size[0] < rect_size[1]:
        if (angle < -15):
            angle += 90
            s1 = rect_size[1]
            s0 = rect_size[0]
            rect_size = (s1, s0)
    else:
        if (angle < -55):
            angle += 90
            s1 = rect_size[1]
            s0 = rect_size[0]
            rect_size = (s1, s0)

    short_side = min(rect_size[0], rect_size[1])
    pad_len = 7
    rect_size = (int(rect_size[0] + pad_len), int(rect_size[1]) + pad_len)
    center = (int(rect[0][0]), int(rect[0][1]))

    angle = angle * np.pi / 180.0
    rotate_matrix = np.array([np.cos(angle), -np.sin(angle), center[0],
                              np.sin(angle), np.cos(angle), center[1]], dtype=np.float64)
    dx = (rect_size[0] - 1) * 0.5
    dy = (rect_size[1] - 1) * 0.5
    rotate_matrix[2] -= (rotate_matrix[0] * dx + rotate_matrix[1] * dy)
    rotate_matrix[5] -= (rotate_matrix[3] * dx + rotate_matrix[4] * dy)
    patch = cv2.warpAffine(im, rotate_matrix.reshape(2, 3), rect_size,
                           flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)

    pad_points = welf_rrect_to_4pts(center, rect_size, angle)
    if patch.shape[0] < 5 or patch.shape[1] < 5:
        print('crop small than 5 {}'.format(patch.shape[0], patch[1]))
        return None, None
    return patch, pad_points