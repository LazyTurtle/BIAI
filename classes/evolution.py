from resource import *
from server import *
from mesa.space import MultiGrid

if __name__ == '__main__':
    grid = MultiGrid(10, 10, True)
    instantiate_server(grid, ResourceModel)
