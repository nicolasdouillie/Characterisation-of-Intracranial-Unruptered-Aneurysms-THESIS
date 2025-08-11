import pyvista as pv
import numpy as np
import pymeshlab 


subject = "022"
path = "Données/aneurysm_mesh_sub-" + subject +"_cleaned.vtk"
keeping_largest = True #keeping the biggest remaining structure or not

def get_ml(pv_mesh):

    verts = np.array(pv_mesh.points)
    faces = pv_mesh.faces.reshape((-1, 4))[:, 1:]  # enlever les "3" des triangles

    ml_mesh = pymeshlab.MeshSet()
    ml_mesh.add_mesh(pymeshlab.Mesh(verts, faces))

    return ml_mesh 

def get_pv(ml_mesh):

    verts = np.asarray(ml_mesh.vertex_matrix())
    faces = np.asarray(ml_mesh.face_matrix())

    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
    pv_mesh = pv.PolyData(verts, faces_pv)

    return pv_mesh


def close_holes(pv_mesh):
    #if mesh with holes: defining maxholesize depending on hole size
    
    ml_mesh = get_ml(pv_mesh)
    ml_mesh.apply_filter('meshing_close_holes', maxholesize= 10)

    new_mesh = ml_mesh.current_mesh()
    mesh_pv = get_pv(new_mesh)
    return mesh_pv


mesh = pv.read(path)
closed_mesh = close_holes(mesh)

selected_IDS = []

# --- Plotter
plotter = pv.Plotter()
plotter.add_mesh(closed_mesh, show_edges=True, color='lightgrey')

def find_cell_id(original_mesh, picked_cell):
    picked_pts = np.round(picked_cell.points, 8)
    for cid in range(original_mesh.n_cells):
        cell = original_mesh.get_cell(cid)
        cell_pts = np.round(cell.points, 8)
        # On compare les points triés pour être robustes à l'ordre
        if np.allclose(np.sort(cell_pts, axis=0), np.sort(picked_pts, axis=0)):
            return cid
    return -1  # Non trouvé

def callback(picked_face):
    #select all triangles to be removed
    
    cell_id = find_cell_id(closed_mesh, picked_face)
    print(cell_id)
    selected_IDS.append(cell_id)


# --- Activer la sélection
plotter.enable_element_picking(
    callback=callback,
    mode="face",
    show=True
)

plotter.show()

new_mesh = closed_mesh.remove_cells(selected_IDS).clean()

if keeping_largest == True:
    mesh_save = new_mesh.connectivity('largest')

else:
    connected = new_mesh.connectivity()
    labels = connected['RegionId']
    num_regions = labels.max() + 1
    print(f"Nombre de structures détectées : {num_regions}")
    
    sizes = []
    for i in range(num_regions):
        region = connected.threshold([i, i])
        sizes.append((i, region.n_cells))
    
    sizes = sorted(sizes, key=lambda x: x[1])
    
    smallest_id = sizes[0][0]
    
    smallest_region = connected.threshold([smallest_id, smallest_id])
    
    mesh_save = smallest_region.extract_geometry()

#shows the modified mesh
plotter1 = pv.Plotter()
plotter1.add_mesh(mesh_save, show_edges=True, color='lightgrey')
plotter1.show()

path_save = "Données/sub-"+ subject +"/mask/aneurysm_mesh_sub-" + subject +"_cleaned_1.vtk"
mesh_save.save(path_save)