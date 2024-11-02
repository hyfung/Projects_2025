# ROI Determination

- Given 2D space with `width * height`
- Given a point `(x, y)`
- Determine if it is within our Region Of Interest

## Solution

- Create a mat with `width` and `height` with `bool` as `type`
- Draw and fill a polygon with points [(x1, y1), (x2, y2), (x3, y3)]
- To determine if point `(x, y)` falls in ROI, return `mat[y][x] == 1`

## Complexity Analysis

- Time complexity: `O(1)`
- Space complexity: `O(MN)`
