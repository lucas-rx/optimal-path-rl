# Find an optimal path with RL

## Initialize the project

Create a virtual environment `python3 -m venv venv` and install the dependencies `pip install -r requirements.txt`.

## Change the input parameters

The project reads the `value-iteration.txt` file or the `Q-Learning.txt` file, depending of the algorithm.
You can change the grid dimensions.
0 is empty, 1 is victory, 2 is defeat, 3 is inaccessible.

For VI only : after the grid, the first row is gamma (0 <= gamma < 1, importance of the future rewards), the second row is the tolerance (early stops the algorithm if two politics are similar).

For QL only : after the grid, the first row is gamma (0 <= gamma < 1), the second row is alpha (learning rate), the third row is the number of iterations.

## Value Iteration

Run `python3 value_iteration.py`, you can see the results in `log-file_VI.txt`.

## Q-Learning

Run `python3 Q_learning.py`, you can see the results in `log-file_QL.txt`.
