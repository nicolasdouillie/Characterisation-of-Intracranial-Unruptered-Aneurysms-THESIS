import argparse
import pyvista as pv
import numpy as np
import pymeshlab
import pandas as pd


def get_pv(ml_mesh):

    verts = np.asarray(ml_mesh.vertex_matrix())
    faces = np.asarray(ml_mesh.face_matrix())

    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
    pv_mesh = pv.PolyData(verts, faces_pv)

    return pv_mesh   

def get_ml(pv_mesh):

    verts = np.array(pv_mesh.points)
    faces = pv_mesh.faces.reshape((-1, 4))[:, 1:]  # enlever les "3" des triangles

    ml_mesh = pymeshlab.MeshSet()
    ml_mesh.add_mesh(pymeshlab.Mesh(verts, faces))

    return ml_mesh 

def close_holes(pv_mesh):
    ml_mesh = get_ml(pv_mesh)
    ml_mesh.apply_filter('meshing_close_holes', maxholesize= 1000)

    new_mesh = ml_mesh.current_mesh()
    mesh_pv = get_pv(new_mesh)
    return mesh_pv

def compute_convexity(pv_mesh):
    """
    First requirement: for volume assessement it needs to be a closed mesh 
    (only for volume assessment)
    """

    mesh = get_ml(pv_mesh)
    mesh.apply_filter("meshing_invert_face_orientation", forceflip= False)
    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(mesh.current_mesh()) 

    out_dict = mesh_set.get_geometric_measures()
    surface_orig = out_dict['surface_area']

    #closing the mesh:

    mesh_closed = close_holes(pv_mesh)
    mesh = get_ml(mesh_closed)
    mesh.apply_filter("meshing_invert_face_orientation", forceflip= False)
    mesh_set = pymeshlab.MeshSet()

    """
    plotter = pv.Plotter()
    plotter.add_mesh(get_pv(mesh.current_mesh()))
    plotter.show()
    """
    mesh_set.add_mesh(mesh.current_mesh()) 

    out_dict = mesh_set.get_geometric_measures()
    volume_orig = out_dict['mesh_volume']

    spher = (np.pi**(1/3) * (6*abs(volume_orig))**(2/3))/surface_orig
    Ipr = surface_orig/(volume_orig**(2/3))

    #creation of convex envelope
    mesh.generate_convex_hull()
    new_mesh_set = pymeshlab.MeshSet()
    new_mesh_set.add_mesh(mesh.current_mesh())  
    convex_mesh_pv = get_pv(mesh.current_mesh())

    out_dict2 = new_mesh_set.get_geometric_measures()
    volume_convex = out_dict2['mesh_volume']

    volume_ratio = volume_orig / volume_convex

    return spher, Ipr, volume_ratio, surface_orig, volume_orig, volume_convex, convex_mesh_pv

def compute_curvature(pv_mesh, curve_method):

    mesh = get_ml(pv_mesh)
    mesh.apply_filter("meshing_invert_face_orientation", forceflip= False)
    mesh_set = pymeshlab.MeshSet()
    mesh_set.add_mesh(mesh.current_mesh())
    mesh = mesh_set

    mesh.apply_filter("compute_curvature_principal_directions_per_vertex", curvcolormethod= curve_method)
    vertices = mesh.current_mesh().vertex_matrix()
    scalar = mesh.current_mesh().vertex_scalar_array()


    if curve_method == 'Mean Curvature':
        std_shape_index = np.std(scalar)
        convex = (scalar > 0.1).sum()
        concave = (scalar < -0.1).sum()
        total = len(scalar)

        convex_ratio = convex / total
        concave_ratio = concave / total

        #df.loc[df['Subject'] == subj, ['M_std', 'M_convex', 'M_concave']] = [
        #    std_shape_index, convex_ratio, concave_ratio
        #]


    elif curve_method == 'Shape Index':
        std_shape_index = np.std(scalar)
        convex = (scalar > 0.4).sum()
        concave = (scalar < -0.4).sum()
        total = len(scalar)

        convex_ratio = convex / total
        concave_ratio = concave / total

        #df.loc[df['Subject'] == subj, ['S_std', 'S_convex', 'S_concave']] = [
        #    std_shape_index, convex_ratio, concave_ratio
        #]


    print("Based on the curavture (" + curve_method + "):")
    print(" Convex Ratio: " + str(convex_ratio))
    print(" Concave Ratio: " + str(concave_ratio))
    print(" Std shape index: " + str(std_shape_index))
    print("\n")

    return vertices, scalar

def display_indices(surf, vol, vol_c, mesh, mesh_mean, scal_mean, mesh_shape, scal_shape, mesh_conv):

    plotter = pv.Plotter(shape=(3, 2), row_weights=[2, 2, 1], title="Aneurysm Characterization")

    # -- Subplot (0, 0): Original mesh with surface and volume --
    plotter.subplot(0, 0)
    plotter.add_text(f"Original Mesh\nSurface: {surf:.2f}\nVolume: {vol:.2f}", font_size=10)
    plotter.add_mesh(mesh, color='lightblue', show_edges=True)


    # -- Subplot (0, 1): Convex hull with volume ratio --
    plotter.subplot(0, 1)
    plotter.add_text(f"Convex Hull Volume: {vol_c:.3f}", font_size=10)
    plotter.add_mesh(mesh_conv, color='lightblue', show_edges=True)

    # -- Subplot (1, 0): Mean curvature + histogram --
    plotter.subplot(1, 0)
    plotter.add_mesh(mesh_mean, scalars='Curvature', cmap='viridis', show_scalar_bar=False)
    plotter.add_scalar_bar(title="Mean Curvature")

    plotter.subplot(2, 0)
    chart1 = pv.Chart2D()
    hist1, edges1 = np.histogram(scal_mean, bins=30)
    chart1.bar(edges1[:-1], hist1 / hist1.sum(), color="blue")
    chart1.x_label = "Mean Curvature"
    chart1.y_label = "Frequency"
    plotter.add_chart(chart1)

    # -- Subplot (1, 1): Shape index + histogram --
    plotter.subplot(1, 1)
    plotter.add_mesh(mesh_shape, scalars='Curvature', cmap='plasma', show_scalar_bar=False)
    plotter.add_scalar_bar(title="Shape Index")

    plotter.subplot(2, 1)
    chart2 = pv.Chart2D()
    hist2, edges2 = np.histogram(scal_shape, bins=30)
    chart2.bar(edges2[:-1], hist2 / hist2.sum(), color="purple")
    chart2.x_label = "Shape Index"
    chart2.y_label = "Frequency"
    plotter.add_chart(chart2)

    plotter.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', help='Aneurysm mesh at format .vtk', required=True, type=str)
    args = parser.parse_args()
    

    ent_mesh = pv.read(args.input_file)
    spher, ipr, volume_r, surface, volume, volume_c, conv_mesh = compute_convexity(ent_mesh)
    
    print("\n")
    print("Based on the geometry (surface and volume): ")
    print(" Sphericity: " + str(spher))
    print(" IPR: " + str(ipr))
    print(" Volume ratio convex hull: " + str(volume_r))
    print("\n")
    
    vertices_mean, scalar_mean = compute_curvature(ent_mesh, curve_method='Mean Curvature')
    vertices_shape, scalar_shape = compute_curvature(ent_mesh, curve_method='Shape Index')
    
    mesh_pv_mean = pv.PolyData(vertices_mean)
    mesh_pv_mean['Curvature'] = scalar_mean   

    mesh_pv_shape = pv.PolyData(vertices_shape)
    mesh_pv_shape['Curvature'] = scalar_shape
    
    display_indices(surface, volume, volume_c, ent_mesh, mesh_pv_mean, 
                    scalar_mean, mesh_pv_shape, scalar_shape, conv_mesh)
    
    """
    list_sub = ['sub-022', 'sub-051', 'sub-066', 'sub-074', 'sub-075', 'sub-081',
           'sub-099', 'sub-104', 'sub-106', 'sub-120', 'sub-127', 'sub-132', 
           'sub-140', 'sub-141', 'sub-163', 'sub-192', 'sub-208', 'sub-225']
    
    for sub in list_sub:
        
        path = "Données/aneurysm_mesh_" + sub + "_cleaned.vtk"
        ent_mesh = pv.read(path)

        spher, ipr, volume_r, surface, volume, volume_c, conv_mesh = compute_convexity(ent_mesh)
        
        print("\n")
        print("Based on the geometry (surface and volume): ")
        print(" Sphericity: " + str(spher))
        print(" IPR: " + str(ipr))
        print(" Volume ratio convex hull: " + str(volume_r))
        print("\n")
        
        pd_path = "Characterization_result_M01_S04.csv"
    
        df = pd.read_csv(pd_path)


    #sub="sub-225"




        df = pd.concat([df, pd.DataFrame([{
                'Subject': sub,
                'Spher': spher,
                'IPR': ipr,
                'Vol_r': volume_r
            }])], ignore_index=True)
    
        vertices_mean, scalar_mean, df = compute_curvature(ent_mesh, df, sub, curve_method='Mean Curvature')
        vertices_shape, scalar_shape, df = compute_curvature(ent_mesh, df, sub, curve_method='Shape Index')
    
        mesh_pv_mean = pv.PolyData(vertices_mean)
        mesh_pv_mean['Curvature'] = scalar_mean   
    
        mesh_pv_shape = pv.PolyData(vertices_shape)
        mesh_pv_shape['Curvature'] = scalar_shape
    
        #display_indices(surface, volume, volume_c, ent_mesh, mesh_pv_mean, 
                        #scalar_mean, mesh_pv_shape, scalar_shape, conv_mesh)
        
        #save data
    
        df.to_csv(pd_path, index=False)
    """