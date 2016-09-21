"""Raspberry Pi Face Recognition Treasure Box
Treasure Box Script
Copyright 2013 Tony DiCola 
"""
import cv2

import config
import face
#import hardware


if __name__ == '__main__':
	# Load training data into model
	print 'EaglePy Version 1.0.1'
	print 'Cargando Red Neuronal EaglePy...'
	model = cv2.createEigenFaceRecognizer()
	model.load(config.TRAINING_FILE)
	print 'Entrenamiento de Datos finalizado!'
	# Initialize camer and box.
	camera = config.get_camera()
	#box = hardware.Box()
	# Move box to locked position.
	#box.lock()
	print 'Ejecutando EaglePy...'
	#print 'Press button to lock (if unlocked), or unlock if the correct face is detected.'
	print 'Presione Ctrl-C para salir.'
	while True:
		# Check if capture should be made.
		# TODO: Check if button is pressed.
		#if box.is_button_up():
			#if not box.is_locked:
				# Lock the box if it is unlocked
				#box.lock()
				#print 'Box is now locked.'
			#else:
				#print 'Button pressed, looking for face...'
				# Check for the positive face and unlock if found.
				image = camera.read()
				# Convert image to grayscale.
				image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
				# Get coordinates of single face in captured image.
				result = face.detect_single(image)
				if result is None:
					print 'No se detecta ninguna rostro!  Verificar la imagen capturada en el archivo capture.pgm' \
						  ' para ver que problema (iluminacion, espejos...) hubo e intentelo de nuevo'
					continue
				x, y, w, h = result
				# Crop and resize image to face.
				crop = face.resize(face.crop(image, x, y, w, h))
				# Test face against model.
				label, confidence = model.predict(crop)
				print 'Prediccion {0} de Rostro con una confidencia de {1} (a menor valor mejor).'.format(
					'POSITIVO' if label == config.POSITIVE_LABEL else 'NEGATIVE', 
					confidence)
				if label == config.POSITIVE_LABEL and confidence < config.POSITIVE_THRESHOLD:
					print 'Rostro reconodico, eres NICOLAS :D :D!'
					#box.unlock()
				else:
					print 'NO se reconocio el Rostro, ubiquese mas cerca de la camara por favor!'
