"""
    Taken from https://github.com/fnozarian/CARLA-KITTI/blob/main/bounding_box.py
"""
import copy

import numpy as np
from scipy import stats

WINDOW_WIDTH = 512
WINDOW_HEIGHT = 512
OCCLUDED_VERTEX_COLOR = (255, 0, 0)
VISIBLE_VERTEX_COLOR = (0, 255, 0)
MIN_VISIBLE_VERTICES_FOR_RENDER = 4
MIN_BBOX_AREA_IN_PX = 100


def crop_boxes_in_canvas(cam_bboxes):
    output = copy.deepcopy(cam_bboxes)
    neg_x_inds = np.where(cam_bboxes[:, 0] < 0)[0]
    out_x_inds = np.where(cam_bboxes[:, 2] > WINDOW_WIDTH)[0]
    neg_y_inds = np.where(cam_bboxes[:, 1] < 0)[0]
    out_y_inds = np.where(cam_bboxes[:, 3] > WINDOW_HEIGHT)[0]
    output[neg_x_inds, 0] = 0
    output[neg_y_inds, 1] = 0
    output[out_x_inds, 2] = WINDOW_WIDTH
    output[out_y_inds, 3] = WINDOW_HEIGHT

    return output

def project_3d_points_in_4D_format(p2, points_4d, pad_ones= False):
    """
    Projects 3d points appened with ones to 2d using projection matrix
    :param p2:       np array 4 x 4
    :param points:   np array 4 x N
    :return: coord2d np array 4 x N
    """
    N = points_4d.shape[1]
    z_eps = 1e-2

    if type(points_4d) == np.ndarray:
        if pad_ones:
            points_4d = np.vstack((points_4d, np.ones((1, N))))

        coord2d = np.matmul(p2, points_4d)
        ind = np.where(np.abs(coord2d[2]) > z_eps)

    coord2d[:2, ind] /= coord2d[2, ind]

    return coord2d

def ego_to_cam(points, rot, trans, intrins):
    """Transform points (3 x N) from ego frame into a pinhole camera
    """
    print(points.dtype, rot.dtype, trans.dtype, intrins.dtype)
    if type(points) == np.ndarray:
        points -= trans[:, np.newaxis]
        points = np.matmul(rot.transpose(1, 0), points)
        points = np.matmul(intrins, points)

    # elif type(points) == torch.tensor:
    #     points = points - trans.unsqueeze(1)
    #     points = rot.permute(1, 0).matmul(points)
    #     points = intrins.matmul(points)

    points[:2] /= points[2:3]

    return points


def cam_to_ego(points, rot, trans, intrins):
    """Transform points (3 x N) from pinhole camera with depth
    to the ego frame
    """
    points = torch.cat((points[:2] * points[2:3], points[2:3]))
    points = intrins.inverse().matmul(points)

    points = rot.matmul(points)
    points += trans.unsqueeze(1)

    return points


def calc_projected_2d_bbox(vertices_pos2d):
    """ Takes in all vertices in pixel projection and calculates min and max of all x and y coordinates.
        Returns left top, right bottom pixel coordinates for the 2d bounding box as a list of four values.
        Note that vertices_pos2d contains a list of (y_pos2d, x_pos2d) tuples, or None
    """
    x_coords = vertices_pos2d[:, 0]
    y_coords = vertices_pos2d[:, 1]
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    return [min_x, min_y, max_x, max_y]

def point_in_canvas(pos):
    """Return true if point is in canvas"""
    if (pos[0] >= 0) and (pos[0] < WINDOW_HEIGHT) and (pos[1] >= 0) and (pos[1] < WINDOW_WIDTH):
        return True
    return False

def point_is_occluded(point, vertex_depth, depth_map):
    """ Checks whether or not the four pixels directly around the given point has less depth than the given vertex depth
        If True, this means that the point is occluded.
    """
    y, x = map(int, point)
    from itertools import product
    neigbours = product((1, -1), repeat=2)
    is_occluded = []
    for dy, dx in neigbours:
        if point_in_canvas((dy + y, dx + x)):
            # If the depth map says the pixel is closer to the camera than the actual vertex
            if depth_map[y + dy, x + dx] < vertex_depth:
                is_occluded.append(True)
            else:
                is_occluded.append(False)
    # Only say point is occluded if all four neighbours are closer to camera than vertex
    return all(is_occluded)


def calculate_occlusion(bbox, agent, depth_map):
    """Calculate the occlusion value of a 2D bounding box.
    Iterate through each point (pixel) in the bounding box and declare it occluded only
    if the 4 surroinding points (pixels) are closer to the camera (by using the help of depth map)
    than the actual distance to the middle of the 3D bounding boxe and some margin (the extent of the object)
    """
    bbox_3d_mid = np.mean(bbox[:, 2])
    min_x, min_y, max_x, max_y = calc_projected_2d_bbox(bbox)
    height, width, length = agent.bounding_box.extent.z, agent.bounding_box.extent.x, agent.bounding_box.extent.y

    # depth_margin should depend on the rotation of the object but this solution works fine
    depth_margin = np.max([2 * width, 2 * length])
    is_occluded = []

    for x in range(int(min_x), int(max_x)):
        for y in range(int(min_y), int(max_y)):
            is_occluded.append(point_is_occluded(
                (y, x), bbox_3d_mid - depth_margin, depth_map))

    occlusion = ((float(np.sum(is_occluded))) / ((max_x - min_x) * (max_y - min_y)))

    # discretize the 0–1 occlusion value into KITTI’s {0,1,2,3} labels by equally dividing the interval into 4 parts
    occlusion = np.digitize(occlusion, bins=[0.25, 0.50, 0.75])

    return occlusion


def calculate_occlusion_semantics(bbox_2d, centers, hlw, seman_map, depth_map):
    num_boxes  = centers.shape[0]
    umin = bbox_2d[:, 0]
    vmin = bbox_2d[:, 1]
    umax = bbox_2d[:, 2]
    vmax = bbox_2d[:, 3]
    bbox_depth = centers[:, 2]
    depth_margin = np.max([2 * hlw[:, 2], 2 * hlw[:, 1]], axis= 0)
    occlusion = np.zeros((num_boxes, ))
    seman_ids = np.zeros((num_boxes, )).astype(np.uint8)

    for i in range(num_boxes):
        is_occluded = []
        box_seman   = []
        for x in range(int(umin[i]), int(umax[i])):
            for y in range(int(vmin[i]), int(vmax[i])):
                flag = point_is_occluded((y, x), bbox_depth[i] - depth_margin[i], depth_map)
                is_occluded.append(flag)
                # If the point inside the bbox is occluded, ignore its semantics
                if not flag:
                    box_seman.append(seman_map[y, x])

        occlusion[i] = ((float(np.sum(is_occluded))) / ((umax[i] - umin[i]) * (vmax[i] - vmin[i])))
        box_seman_array = np.array(box_seman)
        if box_seman_array.shape[0] > 0:
            seman_ids[i] = stats.mode(box_seman_array)[0][0]
        else:
            # no valid points, Unknown class
            seman_ids[i] = 0

    occlusion = np.digitize(occlusion, bins= [0.25, 0.50, 0.75])

    return occlusion, seman_ids

def calculate_occlusion_stats(image, bbox_points, depth_map, max_render_depth, draw_vertices=True):
    """ Draws each vertex in vertices_pos2d if it is in front of the camera
        The color is based on whether the object is occluded or not.
        Returns the number of visible vertices and the number of vertices outside the camera.
    """
    num_visible_vertices = 0
    num_vertices_outside_camera = 0

    for i in range(len(bbox_points)):
        x_2d = bbox_points[i, 0]
        y_2d = bbox_points[i, 1]
        point_depth = bbox_points[i, 2]

        # if the point is in front of the camera but not too far away
        if max_render_depth > point_depth > 0 and point_in_canvas((y_2d, x_2d)):
            is_occluded = point_is_occluded(
                (y_2d, x_2d), point_depth, depth_map)
            if is_occluded:
                vertex_color = OCCLUDED_VERTEX_COLOR
            else:
                num_visible_vertices += 1
                vertex_color = VISIBLE_VERTEX_COLOR
        else:
            num_vertices_outside_camera += 1
    return num_visible_vertices, num_vertices_outside_camera


def calc_bbox2d_area(bbox_2d):
    """ Calculate the area of the given 2d bbox
    Input is assumed to be xmin, ymin, xmax, ymax tuple
    """
    xmin = bbox_2d[:, 0]
    ymin = bbox_2d[:, 1]
    xmax = bbox_2d[:, 2]
    ymax = bbox_2d[:, 3]
    area = (ymax - ymin) * (xmax - xmin)
    return area.astype(np.float32)


def calculate_truncation(uncropped_bbox, cropped_bbox):
    "Calculate how much of the object’s 2D uncropped bounding box is outside the image boundary"

    area_cropped   = calc_bbox2d_area(cropped_bbox)
    area_uncropped = calc_bbox2d_area(uncropped_bbox)
    truncation     = 1.0 - area_cropped / area_uncropped
    return truncation


def create_datapoint(image, depth_map, player_transform, max_render_depth=70):
    num_visible_vertices, num_vertices_outside_camera = calculate_occlusion_stats(image,
                                                                                  camera_bbox,
                                                                                  depth_map,
                                                                                  max_render_depth,
                                                                                  draw_vertices=False)

    # At least N vertices has to be visible in order to draw bbox
    if num_visible_vertices >= MIN_VISIBLE_VERTICES_FOR_RENDER > num_vertices_outside_camera:

        uncropped_bbox_2d = calc_projected_2d_bbox(camera_bbox)

        # Crop vertices outside camera to image edges
        crop_boxes_in_canvas(camera_bbox)

        bbox_2d = calc_projected_2d_bbox(camera_bbox)

        area = calc_bbox2d_area(bbox_2d)
        if area < MIN_BBOX_AREA_IN_PX:
            logging.info("Filtered out bbox with too low area {}".format(area))
            return image, None, None

        occlusion = calculate_occlusion(camera_bbox, agent, depth_map)
        rotation_y = get_relative_rotation_y(agent, player_transform)
        alpha = get_alpha(agent, player_transform)
        truncation = calculate_truncation(uncropped_bbox_2d, bbox_2d)
        datapoint = KittiDescriptor()
        datapoint.set_truncated(truncation)
        datapoint.set_occlusion(occlusion)
        datapoint.set_type(obj_type)
        datapoint.set_bbox(bbox_2d)
        datapoint.set_3d_object_dimensions(ext)
        datapoint.set_3d_object_location(sensor_refpoint)
        datapoint.set_rotation_y(rotation_y)
        datapoint.set_alpha(alpha)

        return image, datapoint, camera_bbox