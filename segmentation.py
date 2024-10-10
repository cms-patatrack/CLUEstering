import os
from pickle import FALSE

import CLUEstering as clue
import json
from tifffile import imread, imwrite 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def generate_colors(n):
    cm = plt.get_cmap('hsv')
    return [tuple((np.array(cm(1. * i / n)[:3]) * 255).astype(int)) for i in range(n)]


def prepare_image(f):
    numpy_data = imread(spim_file)  
    print('image shape: ',numpy_data.shape)

    image_min = numpy_data.min()
    image_max = numpy_data.max()

    if image_max > image_min:
        image_normalized = (numpy_data - image_min) / (image_max - image_min)
    else:
        image_normalized = np.zeros_like(numpy_data)

    return image_normalized


def prepare_clue(image):
    if image.ndim==3:
        x0, x1, x2 = image.shape
        x0_coords, x1_coords, x2_coords = np.meshgrid(
            np.arange(x0) * pixel_size_z,  # scale z-coordinates
            np.arange(x1) * pixel_size_x_y,  # scale x-coordinates
            np.arange(x2) * pixel_size_x_y,  # scale y-coordinates
            indexing='ij'
        )
        x0_flat = x0_coords.flatten()
        x1_flat = x1_coords.flatten()
        x2_flat = x2_coords.flatten()
        intensities = image.flatten()

        df = pd.DataFrame({
            'x0': x0_flat,
            'x1': x1_flat,
            'x2': x2_flat,
            'weight': intensities
        })
        return df

    elif image.ndim==2:
        x0, x1  = image.shape
        x0_coords, x1_coords = np.meshgrid(
            np.arange(x0) * pixel_size_z,  # scale z-coordinates
            np.arange(x1) * pixel_size_x_y,  # scale x-coordinates
            indexing='ij'
        )

        x0_flat = x0_coords.flatten()
        x1_flat = x1_coords.flatten()
        intensities = image.flatten()

        df = pd.DataFrame({
            'x0': x0_flat,
            'x1': x1_flat,
            'weight': intensities
        })
        return df



def plot_results(image, clust):
    coords = clust.clust_data.original_coords
    clust_ids = clust.clust_prop.cluster_ids
    clust_seeds = clust.clust_prop.is_seed

    print('n coords ', len(coords), ' n clust id ', len(clust_ids), '  n seeds ', len(clust_seeds))

    mask_points_list=[[] for i in range(clust.clust_prop.n_clusters)]
    mask = np.zeros((*image.shape, 3), dtype=np.uint8)

    if image.ndim==2:

        x_seeds = []
        y_seeds = []

        for index,s in enumerate(clust_seeds):
        #if clust_ids[index]<0:print(clust_ids[index])
       
            if s==1:
                x_seeds.append(coords[index][1])
                y_seeds.append(coords[index][0])
            if clust_ids[index]!=-1:
                mask_points_list[clust_ids[index]].append((coords[index][1], coords[index][0]))

        colors = generate_colors(len(mask_points_list))  # Generate as many colors as there are masks

        for mask_points, color in zip(mask_points_list, colors):
            for x0, x1 in mask_points:

                mask[int(x1), int(x0)] = color

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Original grayscale image slice
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title(f"Original Image")

        # Overlay mask slice
        ax[1].imshow(image, cmap='gray')
        ax[1].imshow(mask, alpha=0.3)  # semi-transparent overlay
        ax[1].set_title(f"Image with Mask Overlay")

        #plt.imshow(numpy_data, cmap="gray")
        ax[1].scatter(x_seeds, y_seeds, s=25, color='r', marker='*')
        plt.show()

    elif image.ndim==3:
        #import napari
        #viewer = napari.Viewer()
        #viewer.add_image(image, colormap='gray', name='original')
        #viewer.add_labels(mask, name='mask')
        #napari.run()
        #import pyvista as pv
        #grid = pv.ImageData()
        #grid.dimensions = np.array(image.shape)+1
        #grid.spacing = (1,1,1)
        #grid.cell_data['intensities'] = image.flatten(order="F")

        #plotter = pv.Plotter()
        #plotter.add_volume(grid, cmap="gray", opacity="sigmoid", shade=True)
        #plotter.show()
        x_seeds = []
        y_seeds = []
        z_seeds = []
        for index,s in enumerate(clust_seeds):
        #if clust_ids[index]<0:print(clust_ids[index])
            #print(coords[index], ' s ',s, '  clust_ids[index]  ',clust_ids[index])
       
            if s==1:
                x_seeds.append(coords[index][2])
                y_seeds.append(coords[index][1])
                z_seeds.append(coords[index][0])
            if clust_ids[index]!=-1:
                mask_points_list[clust_ids[index]].append((coords[index][0], coords[index][1], coords[index][2]))


        colors = generate_colors(len(mask_points_list))  # Generate as many colors as there are masks

        for mask_points, color in zip(mask_points_list, colors):
            for x0, x1, x2 in mask_points:

                mask[int(x0), int(x1), int(x2)] = color


    imwrite("test_plot.tif", image)
    imwrite("test_plot_mask.tif", mask)


    #plt.figure(dpi=300)
    #plt.imshow(mask, alpha=0.3)
    #plt.savefig("test_plot_mask.tif")

cell_membrane=True
spim_file = '/mnt/h/PROJECTS-03/clement/Clock_end/20240418_202031_Experiment/Position 1_Settings 1/t0001_Channel 2.tif'
if cell_membrane:
    spim_file = '/mnt/h/PROJECTS-03/clement/240925_nuc(h2b)_memb(cdh2)_z0,5_xy0,347xy_dt4min/t0001_Channel 1.tif'

pixel_size_x_y = 0.347  # um for x and y
pixel_size_z = 1.5      # um for z
pixel_size_x_y = 1
pixel_size_z = 1


image = prepare_image(spim_file)
#To select a part of the image
if not cell_membrane:
    image = image[:, 400:600, 400:600]
    image = image[3]
    #to try 3D with just one z-slice
    #image = image[3:4]
else:
    image = image[:, 1250:1400, 1300:1600]
#image = image[:, 70:165, 120:185]

    image = image[100]
    imwrite("test_plot_ori.tif", image)
    #to try 3D with just one z-slice
    #image = image[100:101]
    #take the negative of the cell membrane
    image = 1-image


print('shape before running: ',image.shape)

df = prepare_clue(image)

#this is import to reduce the number of points, if not seg fault
#df_filtered = df[(df['weight'] > 0.7) &  (df['weight'] < 0.95)] 

df_filtered = df[(df['weight'] > 0.15)] 
df = df_filtered

print(df.head())  # Preview the first few rows

#clust = clue.clusterer(5, 1.9,5)
clust = clue.clusterer(5, 1.9, 4)
#clust.choose_kernel("exp",[1,2])
#clust.choose_kernel("gaus",[1,1,1])
clust.list_devices()

clust.read_data(df)
print('about to run')
#clust.run_clue()
clust.run_clue("gpu cuda")
print('clue run')
#print(clust.clust_prop)
print('nclusters ',clust.clust_prop.n_clusters)
print('nseeds    ',clust.clust_prop.n_seeds)
plot_results(image, clust)


