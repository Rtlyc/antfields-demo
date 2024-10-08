import numpy
import matplotlib.pyplot as plt
import math

class OccupancyGridMap:
    def __init__(self, data_array, cell_size, occupancy_threshold=0.8, offset=0.5):
        """
        Creates a grid map
        :param data_array: a 2D array with a value of occupancy per cell (values from 0 - 1)
        :param cell_size: cell size in meters
        :param occupancy_threshold: A threshold to determine whether a cell is occupied or free.
        A cell is considered occupied if its value >= occupancy_threshold, free otherwise.
        """

        self.data = data_array
        self.dim_cells = data_array.shape
        self.dim_meters = (self.dim_cells[0] * cell_size, self.dim_cells[1] * cell_size)
        self.cell_size = cell_size
        self.occupancy_threshold = occupancy_threshold
        # 2D array to mark visited nodes (in the beginning, no node has been visited)
        self.visited = numpy.zeros(self.dim_cells, dtype=numpy.float32)
        self.offset = offset

        # Stack to store the blocks of the map that have not been visited well
        self.block_cells = (self.dim_cells[0]//4, self.dim_cells[1]//4)
        self.block_meters = self.cell_size * 4
        self.block_stack = []

    def save(self, filepath):
        numpy.savez(filepath, data=self.data, visited=self.visited) 

    def load(self, filepath):
        f = numpy.load(filepath)
        self.data = f['data']
        self.visited = f['visited']
        

    def mark_visited_idx(self, point_idx, status=1):
        """
        Mark a point as visited with a status.
        :param point_idx: a point (x, y) in data array
        :param status: 1 for bad visited, 2 for good visited
        """
        x_index, y_index = point_idx
        if not self.is_inside_idx((x_index, y_index)):
            raise Exception('Point is outside map boundary')

        self.visited[y_index][x_index] = status

    def mark_visited(self, point, status):
        """
        Mark a point as visited.
        :param point: a 2D point (x, y) in meters
        """
        x, y = point
        x_index, y_index = self.get_index_from_coordinates(x, y)

        return self.mark_visited_idx((x_index, y_index), status)

        
    def is_visited_idx(self, point_idx):
        """
        Check whether the given point is visited.
        :param point_idx: a point (x, y) in data array
        :return: Status of the visit (0, 1, or 2)
        """
        x_index, y_index = point_idx
        if not self.is_inside_idx((x_index, y_index)):
            raise Exception('Point is outside map boundary')

        return self.visited[y_index][x_index]

    def is_visited(self, point):
        """
        Check whether the given point is visited.
        :param point: a 2D point (x, y) in meters
        :return: True if the given point is visited, false otherwise
        """
        x, y = point
        x_index, y_index = self.get_index_from_coordinates(x, y)

        return self.is_visited_idx((x_index, y_index))

    def get_data_idx(self, point_idx):
        """
        Get the occupancy value of the given point.
        :param point_idx: a point (x, y) in data array
        :return: the occupancy value of the given point
        """
        x_index, y_index = point_idx
        if x_index < 0 or y_index < 0 or x_index >= self.dim_cells[0] or y_index >= self.dim_cells[1]:
            raise Exception('Point is outside map boundary')

        return self.data[y_index][x_index]

    def get_data(self, point):
        """
        Get the occupancy value of the given point.
        :param point: a 2D point (x, y) in meters
        :return: the occupancy value of the given point
        """
        x, y = point
        x_index, y_index = self.get_index_from_coordinates(x, y)

        return self.get_data_idx((x_index, y_index))

    def set_data_idx(self, point_idx, new_value):
        """
        Set the occupancy value of the given point.
        :param point_idx: a point (x, y) in data array
        :param new_value: the new occupancy values
        """
        x_index, y_index = point_idx
        if x_index < 0 or y_index < 0 or x_index >= self.dim_cells[0] or y_index >= self.dim_cells[1]:
            raise Exception('Point is outside map boundary')

        self.data[y_index][x_index] = new_value

    def set_data(self, point, new_value):
        """
        Set the occupancy value of the given point.
        :param point: a 2D point (x, y) in meters
        :param new_value: the new occupancy value
        """
        x, y = point
        x_index, y_index = self.get_index_from_coordinates(x, y)

        self.set_data_idx((x_index, y_index), new_value)

    def is_inside_idx(self, point_idx):
        """
        Check whether the given point is inside the map.
        :param point_idx: a point (x, y) in data array
        :return: True if the given point is inside the map, false otherwise
        """
        x_index, y_index = point_idx
        if x_index < 0 or y_index < 0 or x_index >= self.dim_cells[0] or y_index >= self.dim_cells[1]:
            return False
        else:
            return True

    def is_inside(self, point):
        """
        Check whether the given point is inside the map.
        :param point: a 2D point (x, y) in meters
        :return: True if the given point is inside the map, false otherwise
        """
        x, y = point
        x_index, y_index = self.get_index_from_coordinates(x, y)

        return self.is_inside_idx((x_index, y_index))

    def is_occupied_idx(self, point_idx):
        """
        Check whether the given point is occupied according the the occupancy threshold.
        :param point_idx: a point (x, y) in data array
        :return: True if the given point is occupied, false otherwise
        """
        x_index, y_index = point_idx
        if self.get_data_idx((x_index, y_index)) >= self.occupancy_threshold:
            return True
        else:
            return False

    def is_occupied(self, point):
        """
        Check whether the given point is occupied according the the occupancy threshold.
        :param point: a 2D point (x, y) in meters
        :return: True if the given point is occupied, false otherwise
        """
        x, y = point
        x_index, y_index = self.get_index_from_coordinates(x, y)

        return self.is_occupied_idx((x_index, y_index))

    def get_index_from_coordinates(self, x, y):
        """
        Get the array indices of the given point.
        :param x: the point's x-coordinate in meters
        :param y: the point's y-coordinate in meters
        :return: the corresponding array indices as a (x, y) tuple
        """
        # print("get_index_from_coordinates pre: ", (x, y))
        x = x + self.offset
        y = y + self.offset
        # print("get_index_from_coordinates: ", (x, y))
        # print("cell_size: ", self.cell_size)
        x_index = int(round(x/self.cell_size))
        y_index = int(round(y/self.cell_size))

        return x_index, y_index

    def get_coordinates_from_index(self, x_index, y_index):
        """
        Get the coordinates of the given array point in meters.
        :param x_index: the point's x index
        :param y_index: the point's y index
        :return: the corresponding point in meters as a (x, y) tuple
        """
        x = x_index*self.cell_size
        y = y_index*self.cell_size
        x = x - self.offset
        y = y - self.offset
        return x, y

    def get_all_free_space_indices(self):
        mask = (self.visited == 2) & (self.data < self.occupancy_threshold)
        indices = numpy.where(mask)
        return indices
        
    def get_all_free_space_coordinates(self):
        y_indices, x_indices = self.get_all_free_space_indices()
        coordinates = []
        for i in range(len(x_indices)):
            x, y = x_indices[i], y_indices[i]
            coordinates.append(self.get_coordinates_from_index(x, y))
        return numpy.array(coordinates)

    def plot(self, curloc, targetloc, alpha=1, origin='lower', path=None):
        """
        Plot the grid map with different colors based on visit status.
        """
        # Create a color map: 0 - grey, 1 - red, 2 - black/white based on occupancy
        colored_map = numpy.empty((*self.dim_cells, 3))
        for j in range(self.dim_cells[0]):
            for i in range(self.dim_cells[1]):
                if self.visited[j, i] == 0:  # not visited
                    colored_map[j, i] = [0.5, 0.5, 0.5]  # grey
                elif self.visited[j, i] == 1:  # bad visited
                    colored_map[j, i] = [1, 0, 0]  # red
                else:  # good visited
                    if self.data[j, i] >= self.occupancy_threshold:
                        colored_map[j, i] = [0, 0, 0]  # black for occupied
                    else:
                        colored_map[j, i] = [1, 1, 1]  # white for not occupied
        
        # for current location and target location
        curi,curj = self.get_index_from_coordinates(curloc[0],curloc[1])
        print("curij: ", curi, curj)    
        colored_map[curj,curi] = [138/255,43/255,226/255]
        targeti,targetj = self.get_index_from_coordinates(targetloc[0],targetloc[1])
        print("targetij: ", targeti, targetj)
        colored_map[targetj,targeti] = [255/255,20/255,147/255]

        if True: # plot the centers of the blocks
            centers = self.get_block_centers()
            for center in centers:
                centeri,centerj = self.get_index_from_coordinates(center[0],center[1])
                colored_map[centerj,centeri] = [0,1,1]


        # Create a new RGBA image to hold the overlay
        overlay = numpy.zeros((*self.dim_cells, 4))

        # Highlight blocks in block_stack by drawing rectangles
        print("block_stack: ", self.block_stack)
        print("block_stack_size: ", len(self.block_stack))
        for block_idx in self.block_stack:
            col, row = block_idx
            overlay[col*4:(col+1)*4, row*4:(row+1)*4, :] = [0, 1, 0, 0.5]  # Highlighted in green with alpha

        # Combine the original colored_map and the overlay
        final_image = colored_map.copy()
        final_image = final_image.reshape(final_image.shape[0], final_image.shape[1], -1)
        overlay = overlay.reshape(overlay.shape[0], overlay.shape[1], -1)
        final_image = (1 - overlay[:, :, 3:4]) * final_image + overlay[:, :, :3] * overlay[:, :, 3:4]

        fig, ax = plt.subplots()
        # Set the extent of the axes if needed
        extent = [0, self.dim_cells[1], 0, self.dim_cells[0]]  # [xmin, xmax, ymin, ymax]
        ax.imshow(final_image, origin=origin, interpolation='none', alpha=alpha, extent=extent)

        # ax.imshow(final_image, origin=origin, interpolation='none', alpha=alpha)

        plt.draw()
        if path is not None:
            plt.savefig(path)
        plt.close()

    def update(self, position, frame_points, frame_bounds):
        """
        Update the grid map with the given frame points and bounds.
        :param frame_points: a list of points in the frame
        :param frame_bounds: the bounds of the frame

        Update strategy:
        1. Mark all points in the frame as visited, if they are far away from the robot, mark them as bad visited, otherwise good visited
        2. calculate the occupancy value of the points in the frame based on the bounds.If bounds smaller, we assume occupancy values are larger. More Specifically: bounds -> occupancy, [-1, 1] -> [0, 1], bounds valid range(0.01, 0.1), 0.01 -> 1, 0.1 -> 0
        """
        end_points = frame_points[:, 3:]
        end_bounds = frame_bounds[:, 1:]
        # print("end_points: ", end_points.shape)
        # print("end_bounds: ", end_bounds.shape)

        for i in range(end_points.shape[0]):
            cur_point = end_points[i]
            cur_bound = end_bounds[i][0]

            # Mark the point as visited
            dist = numpy.linalg.norm(cur_point - position)
            if dist > 0.2: # bad visited distance
                if self.is_visited(cur_point[:2]) == 0:
                    self.mark_visited(cur_point[:2], 1) # mark as bad visited
            else:
                self.mark_visited(cur_point[:2], 2) # mark as good visited

            # Update the occupancy value of the point
            # speeds = torch.clip((bounds - minimum) / (maximum - minimum), 0.1, 1)
            minimum = 0.01 # value we assume to be obstacle
            maximum = 0.1 # value we assume to be free
            occ_val = 1 - numpy.clip((cur_bound - minimum) / (maximum - minimum), 0, 1)
            self.set_data(cur_point[:2], occ_val)

        # TODO: Update the block stack, some hardcoding here, need to be tuned
        blocks = self.visited.reshape(self.block_cells[0], 4, self.block_cells[1], 4)
        pixels_not_well_visited_rate = ((blocks == 1).sum(axis=(1, 3)))/16
        indices_not_well_visited = numpy.argwhere(pixels_not_well_visited_rate > 0.25)
        # Swap the columns to get indices in [j, i] format
        # indices_not_well_visited = indices_not_well_visited[:, [1, 0]]
        self.block_stack = indices_not_well_visited.tolist()

    # def get_target_block(self):
    #     """
    #     Get the target block to explore.
    #     :return: the target block
    #     """
    #     # TODO: some hardcoding here, need to be tuned
    #     if len(self.block_stack) == 0:
    #         return None
    #     else:
    #         cur_block = self.block_stack.pop(0)
    #         cur_data = self.get_block_data(cur_block)
    #         while numpy.sum(cur_data == 1)/16 < 0.25:
    #             if len(self.block_stack) == 0:
    #                 return None
    #             else:
    #                 cur_block = self.block_stack.pop(0)
    #                 cur_data = self.get_block_data(cur_block)
    #         return cur_block
    

    def get_block_data(self, block):
        """
        Get the data of the given block.
        :param block: the block to get data from
        :return: the data of the given block
        """
        # TODO: some hardcoding here, need to be tuned
        x_index, y_index = block
        return self.visited[x_index*4:(x_index+1)*4, y_index*4:(y_index+1)*4]
    
    def get_block_center(self, block):
        """
        Get the center of the given block.
        :param block: the block to get center from
        :return: the center of the given block
        """
    
        y_index, x_index = block
        #temp_x = x_index*4 + 2
        #temp_y = y_index*4 + 2
        #x,y = self.get_coordinates_from_index(temp_x, temp_y)
        for i in range (-2,6):
            for j in range (-2,6):
                temp_x = x_index*4 + i
                temp_y = y_index*4 + j
                #print(self.visited[temp_y, temp_x])
                if temp_x>=0 and temp_y>=0 and temp_x<=99 and temp_y<=99:
                    if self.visited[temp_y, temp_x] == 2 and self.data[temp_y, temp_x] < self.occupancy_threshold:
                        x, y = self.get_coordinates_from_index(temp_x, temp_y)
                        return x, y
        # x = (x_index+0.5)*self.block_meters
        # y = (y_index+0.5)*self.block_meters
        # x = x - self.offset
        # y = y - self.offset
        return math.inf, math.inf
    
    # def get_block_centers(self):
    #     """
    #     Get the centers of all blocks.
    #     :return: the centers of all blocks
    #     """
    #     centers = []
    #     for block in self.block_stack:
            
    #         centers.append(self.get_block_center(block))
    #     return numpy.array(centers)
    
    def get_block_centers(self):
        """
        Get the centers of all blocks.
        :return: the centers of all blocks
        """
        centers = []
        for block in self.block_stack:
            x, y = self.get_block_center(block)
            if (x,y)!=(math.inf, math.inf):
                centers.append((x,y))
        return numpy.array(centers)
    
    def get_coverage(self):
        """
        Get the coverage of the map.
        :return: the coverage of the map
        """
        return numpy.sum(self.visited == 2)/self.visited.size


