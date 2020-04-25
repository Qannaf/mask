                        #  ================================== Decomentation de code ========================================= #  
                            #                 ce code  fait detection de masque                                               #  
                            #                 Nom : AL-SAHMI    Prénom : Qannaf Ahmed Saleh         Date : 25/04/2020         #     
                            #                 Email : q.alsahmi@gmail.com                                                     #    
                            #                 Demo pour un védio:                                                             #
                        #  ================================================================================================== #


# ================================================ déclaretion des bibliotèques ================================================ #
import cv2 as cv
import numpy as np
import os
import argparse 
import os.path                                                                                         # pour pouvoir utiliser la fonction  isfile
import sys 
from colorama import Fore 
from tensorflow.keras.models import load_model

# =============================================================================================================== #
# ================================ définition le constructeur  ================================================== #
# =============================================================================================================== #
def constructor():
	parser = argparse.ArgumentParser(description="ce code fait la  FER ")
	parser.add_argument('--webCam',help=" Choisir l'appareil photo automatique (par défaut)",default=0)
	parser.add_argument('--video',help="Mettre la chemin du vidéo")
	parser.add_argument('--cascad',help="configuration du modele",default="haarcascade_frontalface_default.xml")
	args = parser.parse_args()
	return args


# =============================================================================================================== #
# ================================ fonction main ================================================================ #
# =============================================================================================================== #


if __name__ == '__main__':
	
	#===>             1) Appel du constructeur 
	args = constructor() 
	
	#===>             2) Télécharger le modelde deep learning
	face_clf = cv.CascadeClassifier(args.cascad)
	model = load_model('face_mask1.h5')

	#===>             3) Création une fenetre
	titleWinsows = "Reconnaissance de l'expression faciale"     
	fps = 25
	frame_array = []

	#===>             4) Faire le choix entre entre le camera ou le vidéo 
	outputFile,fps, frame_array = "input+output/Output_py.avi", 25, []
	try:
		if (args.video):
			print(Fore.BLUE,"\n\t\tvous avez choisi de traiter le fichier ", args.video,"...",Fore.WHITE)
			if not os.path.isfile(args.video):
				print(Fore.RED,"\t\tle fichier  ", args.video, " est introuvable !!!")
				sys.exit(1)
			cap = cv.VideoCapture(args.video)
			outputFile = args.video[:-4]+'_output_py.avi'  
		else:
			print(Fore.GREEN,"\n\t\tvous avez choisi de traiter le fichier capter par le caméra ...")
			cap = cv.VideoCapture(0)                                                                            # Camera par défaut
	except:
		print(Fore.RED,"Nous ne pouvions pas ouvrir ni le vidéo ni la caméra, Veuillez vérifier et réesseyer à nouveau...\n Merci")


	#===>              5) Enregistrer le Traitement  
	vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

	print(Fore.YELLOW,"\n\n\n\n \t\t Etes_vous pret! \n\t\tLe traitement commence ....")
	while True:
		has,img = cap.read()
		if not has:
			print(For.YELLOW,"Le traitement terminé !!!\n Le fichier de sortie est sauvegardé sous le nome  ", outputFile)
			cv.waitKey(3000)
			cap.release()
			break
		
	#===>				6)  Detecter le face
		gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		faces = face_clf.detectMultiScale(gray, 1.3, 5)

		for (x, y, w, h) in faces:
			fc = gray[y:y+h, x:x+w]
			org,fontFace, fontScale,COLOR = (x, y), cv.FONT_HERSHEY_SIMPLEX,2,(200,25, 25)
			m = model.predict(img.reshape(-1,160,160,3)) == model.predict(img.reshape(-1,160,160,3)).max()
			text = np.array(['with mask','no mask'])
			pred = text[m[0]]
			
			if pred == text[1]:
				cv.putText(img, "No Mask", org,fontFace, fontScale,  (0,0,255), 2)
				cv.rectangle(img,org,(x+w,y+h),(0,0,255),2)
			else:
				cv.putText(img, "Mask", org,fontFace, fontScale,  (0,255,0), 2)
				cv.rectangle(img,org,(x+w,y+h),(0,255,0),2)
			

			
			

		vid_writer.write(img.astype(np.uint8))                                                                    # Convertir les np.array de type float64 à unit8 et les afficher sous format vidéo 
		cv.imshow(titleWinsows, img)
		if cv.waitKey(1) & 0xFF == ord('q'):
			print("Le traitement terminé...!")
			break

	cap.release()
	cv.destroyAllWindows()

	