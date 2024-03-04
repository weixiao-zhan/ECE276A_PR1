# usage

`code/grad_descent.py` and `code/grad_descent.ipynb` performs: importing IMU data , 
preprocessing, gradient descent, and plot the results and save Quaternions to `*_best_gd_Q.pt`

`code/rotplot.py` and `code/rotplot.ipynb` perform: importing camera data and gradient descent results,
projecting and output a panorama of the dataset.

# dependency
this implementation used pytorch (only cpu version). No CUDA or GPU accelerator needed.