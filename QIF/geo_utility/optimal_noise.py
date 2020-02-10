"""Usage: geo-optimal <goal> <width> <height> <cell_size> <constraint> <hard_max_loss> [<prior_file>|uniform|random]

goal:          one of
                 min_loss_given_min_bayesrisk
                 min_loss_given_min_georisk
                 max_bayesrisk_given_max_loss
                 max_georisk_given_max_loss
width:         width of grid
height:        height of grid
cell_size:     length of each cell
constraint:    the bayesrisk, georisk or loss constraint, depending on the goal
hard_max_loss: C_xy is forced to 0 when loss(x,y) > hard_max_loss (can greatly reduce the problem size)
prior_file:    file with a single line containing the prior. 'uniform' or 'random' prior can also be used"""

import os


def optimal_noise(optimal_noise_script_launcher,
                  goal,
                  width,
                  height,
                  cell_size,
                  constraint,
                  hard_max_loss,
                  prior_file,
                  solver,
                  result,
                  mv_result):
    cmd = optimal_noise_script_launcher + " " \
          + goal + " " \
          + width + " " \
          + height + " " \
          + cell_size + " " \
          + constraint + " " \
          + hard_max_loss + " " \
          + prior_file + " " \
          + solver

    print "\n\n", cmd
    os.system(cmd)

    cmd = "mv " + result + " " + mv_result
    print "\n\n", cmd
    os.system(cmd)
