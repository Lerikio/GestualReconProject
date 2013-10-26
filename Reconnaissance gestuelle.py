#!/usr/bin/python
# -*- coding: utf-8 -*-


# Importations des bibliothèques
import cv
import time
from scipy import *
from scipy.cluster import vq
import numpy
import sys, os, random, hashlib
from math import *
import math

# Déclaration des variables globales.
# color sers à choisir la couleur à filtrer.
global color

# Choix de la couleur à filtrer en cliquant sur l'image originale
def getObjectColor(event, x, y, flags, param):
	
	if event == cv.CV_EVENT_LBUTTONUP:
		hsv = cv.CloneImage(param)
		cv.CvtColor(param, hsv, cv.CV_BGR2HSV)

		pixel = cv.Get2D(hsv, y, x)
 
		color[0] = pixel[0]
		color[1] = pixel[1]
		color[2] = pixel[2]
	return color

# Fonction de seuillage par couleur de l'image
def filterByColor(img):

# 	Création des images	nécessaires lors de l'algorithme
	imghsv = cv.CreateImage(cv.GetSize(img), 8, 3)
	img_filtered = cv.CreateImage(cv.GetSize(img), 8, 3)
	img_filtered_RGB = cv.CreateImage(cv.GetSize(img), 8, 3)

#	Définition de la tolérance des couleurs
	color_min = [0,0,0]
	color_max = [0,0,0]
	color_min[0] = color[0]*90/100
	color_max[0] = color[0]*110/100

#	En réalité, un test ne prenant en compte que la teinte suffit !	
	color_min[1] = color[1]*20/100
	color_max[1] = color[1]*150/100
	color_min[2] = color[2]*20/100
	color_max[2] = color[2]*150/100

#	Conversion de l'image de RGB vers HSV, pour un test plus simple de la teinte
	cv.CvtColor(img, imghsv, cv.CV_BGR2HSV)

#	Flou gaussien sur l'image, pour éliminer les imperfections
#	cv.Smooth( imghsv, imghsv, cv.CV_GAUSSIAN, 15, 0 )

#	Pixel définissant la couleur de "fond". De base, il est noir (0,0,0).
	null_pixel = cv.Scalar(0,0,0)

#	Test de tous les pixels de l'image pour la couleur choisie.
	width, height = cv.GetSize(imghsv)
	for i in range(0, height-1):
		for j in range(0, width-1):
			pixel = imghsv[i,j]
			h = (color_min[0] <= pixel[0] <= color_max[0])
		#	En réalité, un test ne prenant en compte que la teinte suffit !
			s = (color_min[1] <= pixel[1] <= color_max[1])
			v = (color_min[2] <= pixel[2] <= color_max[2])
			if (h and s and v):
				cv.Set2D(img_filtered, i, j, pixel)
			else:
				cv.Set2D(img_filtered, i, j, null_pixel)	

#	Conversion vers RGB, pour permettre l'affichage
	cv.CvtColor(img_filtered, img_filtered_RGB, cv.CV_HSV2BGR)

	return img_filtered_RGB

# Format des boites de collision (BBoxes) :
# { {(topleft_x), (topleft_y)}, {(bottomright_x), (bottomright_y)} }
top = 0
bottom = 1
left = 0
right = 1

# Algorithme de détection de l'objet en mouvement (AABB)
def merge_collided_bboxes( bbox_list ):
	
# 	Pour chaque boite...
	for this_bbox in bbox_list:
		
	#	On parcours la liste de toutes las autres boites...
		for other_bbox in bbox_list:
		#	La boite étudiée n'est bien sûr pas en collision avec elle-même :		
			if this_bbox is other_bbox: continue
			
		#	On commence par supposer qu'il y a bien collision :
			has_collision = True

		#	Les coordonnées sont celles de l'écran :
		#	">" signifie soit "plus bas que" soit "plus à droite que"
		#	"<" signigie soit "plus haut que" soit "plus à gauche que"
		#	De plus, on augmente la taille des Bbox de 10%
		#	pour éviter d'avoir des petites boites seules et très proches d'autres boites		
			if (this_bbox[bottom][0]*1.1 < other_bbox[top][0]*0.9): has_collision = False
			if (this_bbox[top][0]*.9 > other_bbox[bottom][0]*1.1): has_collision = False
			
			if (this_bbox[right][1]*1.1 < other_bbox[left][1]*0.9): has_collision = False
			if (this_bbox[left][1]*0.9 > other_bbox[right][1]*1.1): has_collision = False
			
		#	Si il y a toujours collision après ça...
			if has_collision:

			#	On fusionne les boites en question, qu'on rajoute à la liste :
				top_left_x = min( this_bbox[left][0], other_bbox[left][0] )
				top_left_y = min( this_bbox[left][1], other_bbox[left][1] )
				bottom_right_x = max( this_bbox[right][0], other_bbox[right][0] )
				bottom_right_y = max( this_bbox[right][1], other_bbox[right][1] )
				
				new_bbox = ( (top_left_x, top_left_y), (bottom_right_x, bottom_right_y) )
				
				bbox_list.remove( this_bbox )
				bbox_list.remove( other_bbox )
				bbox_list.append( new_bbox )
				
			#	... et on recommence le processus depuis début avec la nouvelle liste :
				return merge_collided_bboxes( bbox_list )
	
# 	Quand il n'y a plus de collision, on renvoie la nouvelle liste, où toutes les boites sont fusionnées :
	return bbox_list

# Classe principale
class Target:

# 	Récupération du flux vidéo et premier traitement
	def __init__(self):

		self.capture = cv.CaptureFromCAM(-1)
	
	#	Définition de la résolution d'exécution : 320*240 pour execution rapide 
		#frame_width, frame_height = [640,480]
		frame_width, frame_height = [320,240]
		cv.SetCaptureProperty( self.capture, cv.CV_CAP_PROP_FRAME_WIDTH, frame_width )
		cv.SetCaptureProperty( self.capture, cv.CV_CAP_PROP_FRAME_HEIGHT, frame_height )

	# 	Récupération de l'image et de sa taille
		frame = cv.QueryFrame(self.capture)
		frame_size = cv.GetSize(frame)

	#	Création des fenêtres de visualisation
		cv.NamedWindow("Original", 1)
		cv.MoveWindow("Original", 0, 59)
		cv.NamedWindow("Filtre des couleurs", 1)
		cv.MoveWindow("Filtre des couleurs", 325, 59)
		cv.NamedWindow("Calcul des différences", 1)
		cv.MoveWindow("Calcul des différences", 650, 59)
		cv.NamedWindow("Seuillage", 1)
		cv.MoveWindow("Seuillage", 162, 350)
		cv.NamedWindow("Suivi de la cible", 1)
		cv.MoveWindow("Suivi de la cible", 488, 350)

	def run(self):

	###	Initialisation

	# 	Vecteur d'interprétation
		vector_coords = [0,0,0,0]
		count = 0
		vector = [0,0]
		ordre = "..."


	##	Déclaration des différentes images utilisées dans le programme
	#	Récupération d'une image de la caméra pour utiliser ses propriétés
		display_image = cv.QueryFrame( self.capture )
	#	Masque de mouvement, en Noir & Blanc (image seuillée, un seul canal)
		grey_image = cv.CreateImage( cv.GetSize(display_image), cv.IPL_DEPTH_8U, 1 )
	#	La fonction RunningAvg() nécessite une image en 32bit...
		running_average_image = cv.CreateImage( cv.GetSize(display_image), cv.IPL_DEPTH_32F, 3 )
	# 	... mais pour réaliser la fonction AbsDiff(), il nous faut des images de même profondeur :
		running_average_in_display_color_depth = cv.CloneImage( display_image )
	#	Déclaration de la différence entre la moyenne temporelle et l'image actuelle :
		difference = cv.CloneImage( display_image )

	##	Initialisation des variables
		target_count = 1
		last_target_count = 1
		last_target_change_t = 0.0
		frame_count=0
		last_known_position = [0,0]
		
		t0 = time.time()

		# Prep for text drawing:
		text_font = cv.InitFont(cv.CV_FONT_HERSHEY_COMPLEX, .4, .4, 0.0, 1, cv.CV_AA )
		text_color = cv.CV_RGB(255,255,255)
		text_ordre = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1.2, 1.2, 0.0, 1, cv.CV_AA )

		while True:

			# Récupération de l'image de la webcam
			camera_image = cv.QueryFrame( self.capture )
			
			frame_size = cv.GetSize(camera_image)
			frame_count += 1
			frame_t0 = time.time()
			
			# Création d'une image pour écriture (cibles)
			display_image = cv.CloneImage( camera_image )
			
			# Création d'unecopie pour modifications
			color_image = cv.CloneImage( display_image )

			# Lissage pour éviter les faux-positifs
			cv.Smooth( color_image, color_image, cv.CV_GAUSSIAN, 19, 0 )
			
			# Seuillage par couleur
			cv.SetMouseCallback("Original", getObjectColor, display_image)
			one_color_image = filterByColor(color_image)
			one_color_image_32b = cv.CreateImage( cv.GetSize(display_image), cv.IPL_DEPTH_32F, 3 )
			cv.ConvertScale(one_color_image, one_color_image_32b)

			# Utilisation d'un moyennage temporel comme "fond statique"
			# 0.320 = 1/frame environ
			cv.RunningAvg( one_color_image_32b, running_average_image, 0.320, None )
			
			# Convrsion d'échelle
			cv.ConvertScale( running_average_image, running_average_in_display_color_depth, 1.0, 0.0 )

			# Calcul des différences entre l'image actuelle et le "fond statique"
			cv.AbsDiff( one_color_image, running_average_in_display_color_depth, difference )
			
			# Conversion vers des nuances de gris (1 canal)
			cv.CvtColor( difference, grey_image, cv.CV_RGB2GRAY )

			# Seuillage pour obtention d'un masque de mouvement
			cv.Threshold( grey_image, grey_image, 80, 255, cv.CV_THRESH_BINARY )
			# Lissage et nouveau seuillage pour éviter les faux-positifs
			cv.Smooth( grey_image, grey_image, cv.CV_GAUSSIAN, 25, 0 )
			cv.Threshold( grey_image, grey_image, 100, 255, cv.CV_THRESH_BINARY )
			
			grey_image_as_array = numpy.asarray( cv.GetMat( grey_image ) )
			non_black_coords_array = numpy.where( grey_image_as_array > 3 )
			# Conversion en une seule liste de coordonnées:
			non_black_coords_array = zip( non_black_coords_array[1], non_black_coords_array[0] )


			# Textes à faire apparaître
			informations_distance = "Distance avec l'image precedente : --"
			informations_count = "Superieure a 15 depuis : 0 images."
			informations_couleur = "Couleur (en HSV) : %d, %.1f, %f." % (color[0], color[1], color[2])

		#	Calcul du barycentre des points restant
			if len(non_black_coords_array):	
				center_array, distortion = vq.kmeans( array( non_black_coords_array ), 1, 15 )
				this_frame_position = center_array[0].tolist()

			##	Interprétation des mouvements

				# Conditions de prise en compte du mouvement
				absolute_distance = math.sqrt(pow((last_known_position[0] - this_frame_position[0]),2) + pow((last_known_position[1] - this_frame_position[1]),2))
				informations_distance = "Distance avec l'image precedente : %d." % (int(absolute_distance))
				if absolute_distance > 15:
					if len(vector_coords):
						vector_coords[2] = this_frame_position[0]
						vector_coords[3] = this_frame_position[1]
						count += 1
						informations_count = "Superieure a 15 depuis : %f images." % (int(count))

						if count > 3:
							vector[0] = (vector_coords[2] - vector_coords[0])
							vector[1] = (vector_coords[3] - vector_coords[1])
							x_direction = math.fabs(vector[0]) - math.fabs(vector[1]) > 15
							y_direction = math.fabs(vector[1]) - math.fabs(vector[0]) > 15
							if x_direction:
								if vector[0] > 0:
									ordre = "Droite !"
								else:
									ordre = "Gauche !"
							elif y_direction:
								if vector[1] > 0:
									ordre = "Bas !"
								else:
									ordre = "Haut !"
							else:
								ordre = "..."
							count = 0
				else:
					vector_coords[0] = this_frame_position[0]
					vector_coords[1] = this_frame_position[1]
					count = 0

			#	Mémoire pour la prochaine image :
				last_known_position = this_frame_position

			#	Dessin de la cible sur l'écran
				center_point = (this_frame_position[0], this_frame_position[1])
				cv.Circle(display_image, center_point, 20, cv.CV_RGB(color[0], color[1], color[2]), 1)
				cv.Circle(display_image, center_point, 15, cv.CV_RGB(color[0], color[1], color[2]), 1)
				cv.Circle(display_image, center_point, 10, cv.CV_RGB(color[0], color[1], color[2]), 2)
				cv.Circle(display_image, center_point,  5, cv.CV_RGB(color[0], color[1], color[2]), 3)
				
			
		#	Sortie du programme avec Entrée ou Esc
			c = cv.WaitKey(7) % 0x100
			if c == 27 or c == 10:
				break
			
		#	Conversion de l'image grise	pour affichage
			grey_image_for_display = cv.CloneImage(display_image)		
			cv.CvtColor( grey_image, grey_image_for_display, cv.CV_GRAY2RGB )


		#	Affichages des images calculées
			cv.PutText( camera_image, ordre, (120,120), text_ordre, text_color )
			cv.ShowImage( "Original", camera_image )
			cv.PutText( one_color_image, informations_couleur, (5,15), text_font, text_color )
			cv.ShowImage( "Filtre des couleurs", one_color_image )
			cv.ShowImage( "Calcul des différences", difference )
			cv.ShowImage( "Seuillage", grey_image_for_display )
			cv.PutText( display_image, informations_distance, (5,15), text_font, text_color )
			cv.PutText( display_image, informations_count, (5,35), text_font, text_color )
			cv.ShowImage( "Suivi de la cible", display_image )
			print ordre

		t1 = time.time()
		time_delta = t1 - t0
		processed_fps = float( frame_count ) / time_delta
		print "%d images en %.1f s, pour une moyenne de %f fps." % ( frame_count, time_delta, processed_fps )
		

# Démarrage de l'application
if __name__=="__main__":
	color = [75, 100, 100]
	t = Target()
	t.run()			