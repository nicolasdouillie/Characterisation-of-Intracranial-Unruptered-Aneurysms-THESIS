from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QStackedWidget, QWidget, QFileDialog, QSlider, QLineEdit, QLabel
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap

import pyvista as pv
pv.global_theme.allow_empty_mesh = True
from pyvistaqt import QtInteractor
import nibabel as nib
import sys
import numpy as np
from skimage.segmentation import random_walker
from skimage import measure, morphology
from skimage.measure import marching_cubes
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from skimage.graph import pixel_graph
import networkx as nx
import open3d as o3d
import sys
import os
#import pymeshlab

##### A faire: clear memory 
##### sauvé cleaned_image même sans cleaning
##### Ajouter thread même pour segmentation avec threshold et cleaning image 



def resource_path(relative_path):

    try:
        base_path = sys._MEIPASS  # PyInstaller ajoute ce chemin en exécutable
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def get_3D_struct(sgm):
    # Extraire les sommets et les faces à partir du volume 3D
    verts, faces, _, _ = marching_cubes(sgm)

    # Créer un mesh Open3D
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Appliquer le lissage de Taubin
    mesh = mesh.filter_smooth_taubin(
        number_of_iterations=5,
        lambda_filter=0.5,
        mu=-0.53
    )

    # Récupérer les données lissées
    verts_smoothed = np.asarray(mesh.vertices)
    faces_smoothed = np.asarray(mesh.triangles)

    return verts_smoothed, faces_smoothed


def find_nearest_node(selected_point, global_graph):
    """
    Find the nearest graph-node from the selected point
    """

    nodes = list(global_graph.nodes)
    nodes_pos = np.array([global_graph.nodes[n]["pos"] for n in nodes])
    distances = np.linalg.norm(nodes_pos - selected_point, axis=1)

    nearest_idx = np.argmin(distances)
    nearest_node = nodes[nearest_idx]
    nearest_node_pos = nodes_pos[nearest_idx]

    return nearest_node, nearest_node_pos


def compute_path(nodes, G):
    path_set = set()

    for i in range(len(nodes) - 1):
        source = nodes[i]
        target = nodes[i + 1]

        shortest = nx.shortest_path(G, source=source, target=target)
        path_set.update(shortest)

    path = list(path_set)
    return path

class WorkerThread(QThread):
    finished = pyqtSignal(np.ndarray)

    def __init__(self, img, trsh1, trsh2):
        super().__init__()
        self.img = img
        self.trsh1 = trsh1
        self.trsh2 = trsh2

    def run(self):
        mask = np.zeros(self.img.shape)
        mask[self.img > self.trsh1] = 1
        mask[self.img < self.trsh2] = 2  

        RW_all = random_walker(self.img, mask, beta=90, mode='cg_j')
        RW_all[np.where(RW_all == 2)] = 0

        self.finished.emit(RW_all)

class PrepThread(QThread):
    finished = pyqtSignal(object, object, object)

    def __init__(self, segment):
        super().__init__()
        self.segment = segment
    
    def run(self):

        skeleton = skeletonize(self.segment)
        x, y, z = np.where(skeleton)
        points_skel = np.column_stack((x, y, z))

        distance_map = distance_transform_edt(self.segment)
        radii = distance_map[x, y, z]

        graph, _ = pixel_graph(skeleton, connectivity=3)

        global_graph = nx.Graph()

        for i in range(graph.shape[0]):
            pos = points_skel[i]
            global_graph.add_node(i, pos=pos)
        
            for idx_j, j in enumerate(graph[i].indices): #pour les voisins de i (j)
                pos = points_skel[j]
                if not global_graph.has_node(j):
                    global_graph.add_node(j, pos=pos)

                if not global_graph.has_edge(i,j):
                    global_graph.add_edge(i,j, weight=graph[i].data[idx_j])

        self.finished.emit(global_graph, points_skel, radii)

class Segment_trsh(QThread):
    finished = pyqtSignal(np.ndarray)

    def __init__(self, img, trsh, threshold_type):
        super().__init__()
        self.img = img
        self.trsh = trsh
        self.trsh_type = threshold_type

    def run(self):
        
        if self.trsh_type == 'upper':
            mask = np.zeros(self.img.shape)
            mask[self.img>self.trsh] = 1

            fmask = mask.flatten(order="F")

        elif self.trsh_type == 'lower':
            mask = np.zeros(self.img.shape)
            mask[self.img<self.trsh] = 1

            fmask = mask.flatten(order="F")

        self.finished.emit(fmask)

class erase_noise_thread(QThread):
    finished = pyqtSignal(np.ndarray)

    def __init__(self, RW, value):
        super().__init__()
        self.RW = RW
        self.value = value

    def run(self):
            
        con=measure.label(self.RW, connectivity=2)
        cleaned_image = morphology.remove_small_objects(con, min_size=self.value)
        cleaned_image[np.where(cleaned_image > 0)]=1

        self.finished.emit(cleaned_image)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("VA-S")
        self.setWindowIcon(QIcon(resource_path("Aneur_1.png")))
        self.resize(800, 600)

        #####################
        ### Menu bar creation
        #####################

        menu = self.menuBar()

        #Segmentation menu
        segmentation_menu = menu.addMenu("Segmentation")

        #Load the MRA_file for random-walker segmentation
        load_MRA = segmentation_menu.addAction("Load MRA-file")
        load_MRA.triggered.connect(self.load_MRA_image)

        #Set thresholds + random walker segmentation
        tresholds = segmentation_menu.addAction("Vessels segmentation") 
        tresholds.triggered.connect(self.get_trsh)

        #Display random_walker
        RW_segment = segmentation_menu.addAction("Display RW-S")
        RW_segment.triggered.connect(self.displayRW)

        #Selection menu
        select_menu = menu.addMenu("Selection")

        #Load the segmentation (after random-walker) for aneurysm selection
        load_segmentation = select_menu.addAction("Load segmentation")
        load_segmentation.triggered.connect(self.load_segmentation)

        #Select the aneurysm
        start_selection = select_menu.addAction("Select aneurysm")
        start_selection.triggered.connect(self.select_aneur)

        #Characterization menu
        #Characterization_menu = menu.addMenu("Characterization")

        #Load the aneurysm mesh for characterization
        #load_aneur = Characterization_menu.addAction("Load aneurysm mesh")
        #triggered.connect ...

        #Start characterization
        #charact_aneur = Characterization_menu.addAction("Characterize aneurysm")
        #triggered.connect ...

        self.plotter = QtInteractor(self)

        ##################
        ### PAGES creation
        ##################

        #page 0 -> first window with image that present the possibilities 

        page0_layout = QVBoxLayout()
        page0_label = QLabel("VA-S")
        font = page0_label.font()
        font.setPointSize(30)
        page0_label.setFont(font)
        page0_label.setAlignment(Qt.AlignHCenter)
        page0_label.setMaximumHeight(60)
        page0_layout.addWidget(page0_label)

        label_image = QLabel()
        pixmap = QPixmap(resource_path("GUI.png"))
        label_image.setPixmap(pixmap)
        label_image.setFixedSize(1000, 615)
        label_image.setScaledContents(True)
        page0_layout.addWidget(label_image)


        """
        svg_widget = QSvgWidget("GUI.svg")
        svg_widget.setFixedSize(600, 400)
        page0_layout.addWidget(svg_widget)
        """
        

        #page 1 -> 3D visualization when MRA-file is uploaded

        page1_layout = QVBoxLayout()
        page1_label = QLabel("3D Visualization")
        font = page1_label.font()
        font.setPointSize(30)
        page1_label.setFont(font)
        page1_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        page1_layout.addWidget(page1_label)

        #page 2 -> Setting thresholds !!!! Voir si j'ajoute bouton compute Random-walker 

        label = QLabel("Threshold Adjustment + RW")
        font = label.font()
        font.setPointSize(15)
        label.setFont(font)
        label.setAlignment(Qt.AlignHCenter)

        label1 = QLabel("Trsh1:")
        label1.setAlignment(Qt.AlignRight)

        self.label2 = QLabel("Lower threshold limit for known vessels")
        self.label2.setAlignment(Qt.AlignLeft)

        label3 = QLabel("Trsh2:")
        label3.setAlignment(Qt.AlignRight)

        self.label4 = QLabel("Higher threshold limit for known not-vessels")
        self.label4.setAlignment(Qt.AlignLeft)

        self.save_button1 = QPushButton("Save Trsh1")
        self.save_button1.clicked.connect(self.save_trsh1)

        self.save_button2 = QPushButton("Save Trsh2")
        self.save_button2.clicked.connect(self.save_trsh2)

        self.compute_RW_button = QPushButton("Compute RW")
        self.compute_RW_button.clicked.connect(self.computeRW)

        self.label_compute = QLabel("RW: Only when thresholds are set")
        self.label_compute.setAlignment(Qt.AlignHCenter)


        layout_title = QVBoxLayout()
        layout_title.addWidget(label)

        thrs_text_layout = QHBoxLayout()
        thrs_text_layout.addWidget(label1)
        thrs_text_layout.addWidget(self.label2)
        thrs_text_layout.addWidget(label3)
        thrs_text_layout.addWidget(self.label4)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.save_button1)
        button_layout.addWidget(self.save_button2)
        button_layout.addWidget(self.compute_RW_button)
        button_layout.addWidget(self.label_compute)


        page2_layout = QVBoxLayout()
        page2_layout.addLayout(layout_title)
        page2_layout.addLayout(thrs_text_layout)
        page2_layout.addLayout(button_layout)

        #page 3 -> Display and save RW segmentation 

        page3_layout = QVBoxLayout()
        page3_label = QLabel("Random walker segmentation")
        font = page3_label.font()
        font.setPointSize(30)
        page3_label.setFont(font)
        page3_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        page3_layout.addWidget(page3_label)

        page3_layout_processing = QHBoxLayout()

        self.save_segment_button = QPushButton("Save 3D-segment")
        self.save_segment_button.clicked.connect(self.save_segment)
        page3_layout_processing.addWidget(self.save_segment_button)
        page3_layout.addLayout(page3_layout_processing)

        #page 4 -> 3D visualization of segmentation meshing

        page4_layout = QVBoxLayout()    
        page4_label = QLabel("3D Visualization")
        font = page4_label.font()
        font.setPointSize(30)
        page4_label.setFont(font)
        page4_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        page4_layout.addWidget(page4_label)

        label_preparation_inf = QLabel("Waiting for preparation before starting characterization")
        label_preparation_inf.setAlignment(Qt.AlignLeft)

        self.labal_preparation = QLabel("Preparation:")
        self.labal_preparation.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        text_layout = QHBoxLayout()
        text_layout.addWidget(label_preparation_inf)
        text_layout.addWidget(self.labal_preparation)

        page4_layout.addLayout(text_layout)

        #page 5 -> Selection of aneurysm 

        page5_layout = QVBoxLayout()
        page5_label = QLabel("Aneurysm Selection")
        font = page5_label.font()
        font.setPointSize(30)
        page5_label.setFont(font)
        page5_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        page5_layout.addWidget(page5_label)

        self.save_aneurysms_button = QPushButton("Save selected aneurysm")
        self.save_aneurysms_button.clicked.connect(self.save_aneurysm)

        page5_layout.addWidget(self.save_aneurysms_button)


        #Defining stacked layout

        self.page0 = QWidget()
        self.page0.setLayout(page0_layout)

        self.page1 = QWidget()
        self.page1.setLayout(page1_layout)

        self.page2 = QWidget()
        self.page2.setLayout(page2_layout)

        self.page3 = QWidget()
        self.page3.setLayout(page3_layout)

        self.page4 = QWidget()
        self.page4.setLayout(page4_layout)

        self.page5 = QWidget()
        self.page5.setLayout(page5_layout)

        self.stackedWidget = QStackedWidget()
        self.stackedWidget.addWidget(self.page0)
        self.stackedWidget.addWidget(self.page1)
        self.stackedWidget.addWidget(self.page2)
        self.stackedWidget.addWidget(self.page3)
        self.stackedWidget.addWidget(self.page4)
        self.stackedWidget.addWidget(self.page5)

        #window = QWidget()
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.addWidget(self.stackedWidget)

        self.setCentralWidget(central_widget)

        
        self.image_data = None
        self.grid = None

        self.page1.layout().addWidget(self.plotter.interactor)

        # === Connect Signal for Page Switching ===
        self.stackedWidget.currentChanged.connect(self.update_interactor)

    def update_interactor(self, index):
        """
        Dynamically reset the parent of the QtInteractor when the page changes.
        """
        if index == 1:  # Page 1
            # Move the interactor to Page 1 layout
            self.plotter.setParent(self.page1)
            self.page1.layout().addWidget(self.plotter.interactor)
            #print("QtInteractor moved to Page 1")

        elif index == 2:  # Page 2
            # Move the interactor to Page 2 layout
            self.plotter.setParent(self.page2)
            self.page2.layout().addWidget(self.plotter.interactor)
            #print("QtInteractor moved to Page 2")

        elif index == 3:  # Page 3
            # Move the interactor to Page 3 layout
            self.plotter.setParent(self.page3)
            self.page3.layout().addWidget(self.plotter.interactor)
            #print("QtInteractor moved to Page 3")
        
        elif index == 4:  # Page 4
            # Move the interactor to Page 4 layout
            self.plotter.setParent(self.page4)
            self.page4.layout().addWidget(self.plotter.interactor)
            #print("QtInteractor moved to Page 4")

        elif index == 5:  # Page 5
            # Move the interactor to Page 5 layout
            self.plotter.setParent(self.page5)
            self.page5.layout().addWidget(self.plotter.interactor)
            #print("QtInteractor moved to Page 4")
        

    def load_MRA_image(self):


        file_path, _ = QFileDialog.getOpenFileName(self,
                                                   "Select .nii image",
                                                   "",
                                                   "Fichiers NIfTI (*.nii; *.nii.gz);;Tous les fichiers (*)")
        
        self.stackedWidget.setCurrentWidget(self.page1)

        if file_path:
            try:
                self.nifti_image = nib.load(file_path)
                self.image_data = np.asarray(self.nifti_image.get_fdata())

                self.grid = pv.ImageData(dimensions=self.image_data.shape)
                self.grid.origin = self.nifti_image.affine[:3,3]
                self.grid.spacing = tuple(np.diag(self.nifti_image.affine)[:3])
                self.grid.point_data["values"] = self.image_data.flatten(order="F")

                self.grid_mask = pv.ImageData(dimensions=self.image_data.shape)
                self.grid_mask.origin = self.nifti_image.affine[:3,3]
                self.grid_mask.spacing = tuple(np.diag(self.nifti_image.affine)[:3])

                self.trsh_type = 'upper'

                if self.grid is None:
                    return 

                self.plotter.clear()  
                self.clip_plane1 = self.plotter.add_volume_clip_plane(self.grid, name="brain", assign_to_axis= 'z', cmap="gray", show_scalar_bar=False)
                self.plotter.camera_position = [0.0, 1.0, -1.0]
                self.plotter.set_background('black')
                self.plotter.render()

            except Exception as e:
                print(f"Error when charging image: {e}")

    def get_trsh(self):
        
        self.stackedWidget.setCurrentWidget(self.page2)

        def update_trsh(value):
            self.trsh_to_save = value

            if value!=0:

                self.sgm_trsh = Segment_trsh(self.image_data, value, self.trsh_type)
                self.sgm_trsh.finished.connect(display_trsh)
                self.sgm_trsh.start()
                
        def display_trsh(fmask):
            self.plotter.clear_plane_widgets()

            self.grid_mask.point_data["values"] = fmask
            self.clip_plane1 = self.plotter.add_volume_clip_plane(self.grid, name="brain", assign_to_axis= 'z', cmap="gray", show_scalar_bar=False, opacity=1.0)
            self.segmentation_actor = self.plotter.add_mesh_clip_plane(
                    self.grid_mask.contour(isosurfaces=[0.5]),  # Isosurface à 0.5 pour le masque binaire
                    color="red",
                    name="mask",
                    assign_to_axis= 'z',
                    opacity=1.0
                    )


        self.plotter.clear()  
        self.clip_plane1 = self.plotter.add_volume_clip_plane(self.grid, name="brain", assign_to_axis= 'z', cmap="gray", show_scalar_bar=False, opacity=1.0)
        self.plotter.camera_position = [0.0, 1.0, -1.0]
        self.plotter.set_background('black')
        self.plotter.render()

        if self.trsh_type == 'upper':
            slider_title= "Treshold_1 value"

        elif self.trsh_type == 'lower':
            slider_title= "Treshold_2 value"
        

        self.plotter.clear_slider_widgets()
        self.plotter.add_slider_widget(
                        update_trsh,
                        [0,1],
                        value=0,
                        title=slider_title,
                        title_opacity=0.5,
                        title_color="White",
                        interaction_event='end',
                        color='white')
            
    def save_trsh1(self):
        
        self.trsh1 = self.trsh_to_save
        self.label2.setText(str(round(self.trsh1,2)))
        self.trsh_type = 'lower'
        self.get_trsh()
            
    def save_trsh2(self):
        self.trsh2 = self.trsh_to_save
        self.label4.setText(str(round(self.trsh2,2)))
        self.plotter.clear()

    def computeRW(self):

        #print("computeRW called")
        self.label_compute.setText("Computing ...")

        self.worker = WorkerThread(self.image_data, self.trsh1, self.trsh2)
        self.worker.finished.connect(self.computeRW_done)
        self.worker.start()
        #print("Worker started")

    def computeRW_done(self,RW_all):
        self.label_compute.setText("Done")
        self.RW_comp = RW_all
    
    def displayRW(self):

        self.stackedWidget.setCurrentWidget(self.page3)
        RW_data = self.RW_comp

        self.grid_mask.point_data["values"] = RW_data.flatten(order="F")

        self.plotter.clear()
        self.plotter.add_mesh_clip_plane(
            self.grid_mask.contour(isosurfaces=[0.5]),
            color="red",
            name="mask",
            assign_to_axis='z',
            opacity=0.8
        )
        self.plotter.camera_position = [0.0, 1.0, -1.0]
        self.plotter.set_background('black')
        self.plotter.render()

        self.plotter.clear_slider_widgets()
        self.plotter.add_slider_widget(
            self.erase_noise,
            [0, 500],
            value=0,
            title="Remove noise",
            title_opacity=0.5,
            title_color="White",
            interaction_event='end',
            color='white'
        )

    def erase_noise(self, value):
        if value != 0:

            self.eraseN = erase_noise_thread(self.RW_comp, value)
            self.eraseN.finished.connect(self.display_erase_noise)
            self.eraseN.start()
            
    def display_erase_noise(self, cleaned_image):
            
        self.grid_mask.point_data["values"] = cleaned_image.flatten(order="F")
        self.cleaned_image = cleaned_image

        self.plotter.clear_plane_widgets()

        self.plotter.add_mesh_clip_plane(
                self.grid_mask.contour(isosurfaces=[0.5]),  # Isosurface à 0.5 pour le masque binaire
                color="red",
                name="mask",
                assign_to_axis= 'z',
                opacity=0.8
                )

        
    def save_segment(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save segmentation as NIfTI",
            "CerebroVascular_segmentation.nii",  # Default name
            "Fichiers NIfTI (*.nii *.nii.gz);;Tous les fichiers (*)"
        )

        if file_path:
            if hasattr(self, 'cleaned_image') and self.cleaned_image is not None:
                cleaned_segm_nifti = nib.Nifti1Image(self.cleaned_image.astype(float), affine=self.nifti_image.affine)
            else:
                cleaned_segm_nifti = nib.Nifti1Image(self.RW_comp.astype(float), affine=self.nifti_image.affine)
                print('yes')

            nib.save(cleaned_segm_nifti, file_path)


    def compute_graph(self, p_graph,points_skeleton):
        """
        Create graph with edges between neighbors skeleton-points 
        """
        global_graph = nx.Graph()

        for i in range(p_graph.shape[0]):
            pos = points_skeleton[i]
            global_graph.add_node(i, pos=pos)
        
            for idx_j, j in enumerate(p_graph[i].indices): #pour les voisins de i (j)
                pos = points_skeleton[j]
                if not global_graph.has_node(j):
                    global_graph.add_node(j, pos=pos)

                if not global_graph.has_edge(i,j):
                    global_graph.add_edge(i,j, weight=p_graph[i].data[idx_j])

        return global_graph

    def load_segmentation(self):

        file_path, _ = QFileDialog.getOpenFileName(self,
                "Select .nii image",
                "",
                "Fichiers NIfTI (*.nii; *.nii.gz);;Tous les fichiers (*)")
        
        self.stackedWidget.setCurrentWidget(self.page4)

        sgm_nifti = nib.load(file_path)
        segment = sgm_nifti.get_fdata()

        vertices, faces = get_3D_struct(segment)

        #vertices, faces = self.get_3D_struct(segment)
        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64)
        self.mesh_pv = pv.PolyData(vertices, faces_pv)
        
        
        self.plotter.clear()
        self.plotter.set_background('white')
        self.plotter.add_mesh(self.mesh_pv, color= "lightblue", show_edges=True)
        self.plotter.camera_position = [0.0, 1.0, -1.0]
        self.plotter.render()

        #preparation for characterization skeletonize, graph, ...
        self.labal_preparation.setText("Preparation: \n"
                                       "In progress ...")
        
        self.prep_thread = PrepThread(segment)
        self.prep_thread.finished.connect(self.preparation_done)
        self.prep_thread.start()
    
    def preparation_done(self, G, skel_points, radii):
        self.labal_preparation.setText("Preparation: \n Done")
        self.global_graph = G
        self.skel_points = skel_points
        self.radii = radii

    def extract_region_around_nodes(self, fpath, mesh, radii):
        # Création du sous-graphe basé sur les nœuds du chemin
        fpath_subgraph = self.global_graph.subgraph(fpath)

        positions_nodes = []
        radii_nodes = []

        for n in fpath:
            pos = self.global_graph.nodes[n]['pos']

            deg_global = self.global_graph.degree[n]
            deg_fpath = fpath_subgraph.degree[n]

            if deg_global == 1:
                facteur = 4.0
            elif deg_fpath > 1:
                facteur = 3.5
            elif deg_global > 1 and deg_fpath ==1:
                facteur = 2.1

            rayon_local = radii[n]
            positions_nodes.append(pos)
            radii_nodes.append(facteur * rayon_local)

        positions_nodes = np.array(positions_nodes)
        radii_nodes = np.array(radii_nodes)

        # Distance de chaque point du maillage à chaque nœud
        distances = np.linalg.norm(mesh.points[:, np.newaxis, :] - positions_nodes[np.newaxis, :, :], axis=2)

        # Associer chaque point à son nœud le plus proche
        idx_closest_node = np.argmin(distances, axis=1)
        min_distances = distances[np.arange(len(mesh.points)), idx_closest_node]
        min_radii = radii_nodes[idx_closest_node]

        # Masque final
        mask = min_distances < min_radii

        submesh = mesh.extract_points(mask, adjacent_cells=True)
        return submesh

    def select_aneur(self):

        self.stackedWidget.setCurrentWidget(self.page5)

        points = np.array([self.global_graph.nodes[n]['pos'] for n in self.global_graph.nodes])
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(self.global_graph.nodes)}

        edges = []
        for u, v in self.global_graph.edges:
            edges.append([2, node_id_to_idx[u], node_id_to_idx[v]])
        lines = np.array(edges).flatten()

        poly = pv.PolyData()
        poly.points = points
        poly.lines = lines

        self.plotter.clear()
        mesh_actor = self.plotter.add_mesh(self.mesh_pv, show_edges=True)

        self.selected_nodes = []
        path_spheres = []
        final_path_actor = []

        def callback(point):

            #Select extremum point of aneurysm
            nearest_node, nearest_pos = find_nearest_node(point, self.global_graph)
            sphere_actor = self.plotter.add_mesh(pv.Sphere(radius=1, center=nearest_pos), color="yellow", reset_camera=False)
            
            self.selected_nodes.append(nearest_node)
            path_spheres.append(sphere_actor)


        def enable_points():
            self.plotter.remove_actor(mesh_actor)
            self.plotter.add_mesh(poly, color="black", line_width=2, reset_camera=False)
            self.plotter.add_points(points, color="red", point_size=5, render_points_as_spheres=True, reset_camera=False)
            self.plotter.add_mesh(self.mesh_pv, color="lightblue", opacity=0.5, style= "wireframe", show_edges=True, reset_camera=False)


        def undo_last():
            if self.selected_nodes:
                removed_node = self.selected_nodes.pop()
                sphere_actor = path_spheres.pop()
                self.plotter.remove_actor(sphere_actor)

        def show_preview():
            path = compute_path(self.selected_nodes, self.global_graph)

            for sphere_actor in path_spheres:
                self.plotter.remove_actor(sphere_actor)
            
            for node in path:
                fnode_actor = self.plotter.add_mesh(pv.Sphere(radius=1, center=self.global_graph.nodes[node]["pos"]), color="yellow", reset_camera=False)
                final_path_actor.append(fnode_actor)

        def cancel_preview():

            for path_actor in final_path_actor:
                self.plotter.remove_actor(path_actor)
            
            for i, node in enumerate(self.selected_nodes):
                sphere_actor = self.plotter.add_mesh(pv.Sphere(radius=1, center=self.global_graph.nodes[node]["pos"]), color="yellow", reset_camera=False)
                path_spheres[i] = sphere_actor           


        help_text = (
            "Instructions:\n"
            "- Press 't' to enable the points \n"
            "- Right-click to select a point\n"
            "- Press 'z' to undo the last point\n"
            "- Press 'Enter' to show path preview\n"
            "- Press 'm' to cancel preview \n"
        )

        task_text = (
            "Task:                             \n"
            "Select points around the aneurysm \n"
            "to create a closed path             "
        )

        self.plotter.add_text(help_text, position='upper_left', font_size=10, color='black')
        self.plotter.add_text(task_text, position='upper_right', font_size=10, color='black')

        self.plotter.enable_point_picking(callback, show_message=False)
        self.plotter.add_key_event("z", undo_last)
        self.plotter.add_key_event("Return", show_preview)
        self.plotter.add_key_event("m", cancel_preview)
        self.plotter.add_key_event("t", enable_points)

    def save_aneurysm(self):
        self.plotter.disable_picking()

        final_path = compute_path(self.selected_nodes, self.global_graph)
        for node in final_path:
            pos = self.global_graph.nodes[node]['pos']
            rad = self.radii[node]

            sphere = pv.Sphere(radius=rad*2.1, center=pos, theta_resolution=30, phi_resolution=30)
            self.plotter.add_mesh(sphere, color="blue", reset_camera=False)

        extracted_mesh = self.extract_region_around_nodes(final_path, self.mesh_pv, self.radii)
        extracted_mesh = extracted_mesh.extract_surface()  

        cleaned_mesh = extracted_mesh.connectivity('largest')

        self.plotter.clear()
        self.plotter.add_mesh(cleaned_mesh, color= "lightblue", show_edges=True, show_scalar_bar=False)

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save aneurysm meshing as VTK",
            "Selected_aneurysm.vtk",  # Default name
            "Fichiers NIfTI (*.vtk);;Tous les fichiers (*)"
        )

        if file_path:
            cleaned_mesh.save(file_path)      



app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())



        