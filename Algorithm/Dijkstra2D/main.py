import heapq

# Dijkstra's Algorithm in 2D Grid
def dijkstra(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    
    # Priority queue to store (cost, row, col)
    pq = [(0, start[0], start[1])]
    
    # Distance table to store the shortest distance to each cell
    distances = [[float('inf')] * cols for _ in range(rows)]
    distances[start[0]][start[1]] = 0
    
    while pq:
        current_cost, x, y = heapq.heappop(pq)
        
        # If the goal is reached
        if (x, y) == goal:
            return current_cost
        
        # Explore all four possible directions
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Check if the next cell is within bounds and not blocked (-1)
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] != -1:
                new_cost = current_cost + grid[nx][ny]
                
                # If a shorter path to this cell is found
                if new_cost < distances[nx][ny]:
                    distances[nx][ny] = new_cost
                    heapq.heappush(pq, (new_cost, nx, ny))
    
    # If the goal is not reachable, return -1
    return -1

# Example grid (0 represents walkable cells, -1 represents blocked cells)
grid = [
    [1, 1, 1, 1],
    [1, -1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1]
]

start = (0, 0)  # Starting point (top-left corner)
goal = (3, 3)   # Goal (bottom-right corner)

# Running Dijkstra's algorithm on the grid
result = dijkstra(grid, start, goal)
print(f"Shortest path cost: {result}")
