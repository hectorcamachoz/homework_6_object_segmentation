import cv2
import argparse
import numpy as np

window_params = {'capture_window_name':'Input video',
                 'detection_window_name':'Detected object'}

HSV_params_pasto = {'low_H': 80, 'low_S': 22, 'low_V': 5, 'high_H': 180, 'high_S': 255, 'high_V': 255}
HSV_params_pista = {'low_H': 35, 'low_S': 0, 'low_V': 5, 'high_H': 143, 'high_S': 36, 'high_V': 116}

def parse_cli_data():
    """
    This function parses the command line arguments

    Returns:
        args: input arguments

    """
    parser = argparse.ArgumentParser(description='Tunning HSV bands for object detection')
    parser.add_argument('--video_file', type=str, default='camera', help='Video file used for the object detection process')
    parser.add_argument('--frame_resize_percentage', type=int, help='Rescale the video frames, e.g., 20 if scaled to 20%')
    return parser.parse_args()

def initialise_camera(args):
    """
    This function initialises the camera to capture the video

    Args:
        args: input arguments

    Returns:
        cap: video capture object
    
    """
    cap = cv2.VideoCapture(args.video_file)
    if not cap.isOpened():
        print("Error opening video file.")
        exit()
    return cap

def rescale_frame(frame, percentage=20):
    """
    This function rescales the input frame by a given percentage

    Args:
        frame: input frame
        percentage: percentage to rescale the frame

    Returns:
        frame: rescaled frame

    """
    width = int(frame.shape[1] * percentage / 100)
    height = int(frame.shape[0] * percentage / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def segment_object2(cap: cv2.VideoCapture, args: argparse.Namespace) -> None:
    """
    This function segments the object in the video

    Args:
        cap: video capture object
        args: input arguments

    Returns:
        None

    """
    # Crea las ventanas antes de añadir trackbars o mostrar el contenido
    cv2.namedWindow(window_params['capture_window_name'])
    cv2.namedWindow(window_params['detection_window_name'])
    cv2.namedWindow('rect')

    # Obtiene la longitud total del video en frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Añade un trackbar al 'Input video' para controlar la posición del video
    cv2.createTrackbar('Frame', window_params['capture_window_name'], 0, total_frames-1, lambda x: None)

    last_trackbar_pos = -1  # Inicializa con un valor que nunca será una posición válida del trackbar

    while cap.isOpened():
        # Lee la posición actual del trackbar
        trackbar_pos = cv2.getTrackbarPos('Frame', window_params['capture_window_name'])

        # Verifica si la posición del trackbar ha cambiado
        if trackbar_pos != last_trackbar_pos:
            # Solo actualiza la posición del video si la posición del trackbar ha cambiado
            cap.set(cv2.CAP_PROP_POS_FRAMES, trackbar_pos)
            last_trackbar_pos = trackbar_pos

        ret, frame = cap.read()
        if not ret:
            print("ERROR! - current frame could not be read")
            break

        frame = rescale_frame(frame, args.frame_resize_percentage)
        frame = cv2.medianBlur(frame, 5)
        frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        """
        frame_threshold = cv2.inRange(frame_HSV, (84, 23, 0), (180, 255, 255))
        # Usinf bitwise or to filter out the grassy region from current frame, but keep the moving object

        bitwise_AND = cv2.bitwise_and(frame, frame, mask=frame_threshold)
        """
        # Primer filtro HSV para el pasto
        mask1 = cv2.inRange(frame_HSV, (HSV_params_pasto['low_H'], HSV_params_pasto['low_S'], HSV_params_pasto['low_V']), 
                                        (HSV_params_pasto['high_H'], HSV_params_pasto['high_S'], HSV_params_pasto['high_V']))

        # Segundo filtro HSV para la pista de atletismo
        mask2 = cv2.inRange(frame_HSV, (HSV_params_pista['low_H'], HSV_params_pista['low_S'], HSV_params_pista['low_V']), 
                                        (HSV_params_pista['high_H'], HSV_params_pista['high_S'], HSV_params_pista['high_V']))

        # Combinar las máscaras con OR
        combined_mask = cv2.bitwise_or(mask1, mask2)

        # Invertir la máscara combinada para mantener solo la persona
        inverted_mask = cv2.bitwise_not(combined_mask)

        # Aplicar la máscara invertida para obtener el resultado
        bitwise_AND = cv2.bitwise_and(frame, frame, mask=inverted_mask)

        #Convinar mark 1 y frame
        bitwise_OR = cv2.bitwise_and(frame, frame, mask=mask1)

        #Convinar mark 2 y frame
        bitwise_OR2 = cv2.bitwise_and(frame, frame, mask=mask2)


        cnts, _ = cv2.findContours(bitwise_AND, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if cnts:
            largest_cnt = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_cnt)

            fixed_width, fixed_height = 30, 30  # Ajusta el tamaño fijo del rectángulo aquí
            x_center, y_center = x + w // 2, y + h // 2
            x_fixed = max(0, x_center - fixed_width // 2)
            y_fixed = max(0, y_center - fixed_height // 2)

            cv2.rectangle(frame, (x_fixed, y_fixed), (x_fixed + fixed_width, y_fixed + fixed_height), (0, 255, 0), 2)

        cv2.imshow(window_params['capture_window_name'], frame)
        cv2.imshow(window_params['detection_window_name'], bitwise_AND)
        cv2.imshow('rect', inverted_mask)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

def segment_object(cap: cv2.VideoCapture, args: argparse.Namespace) -> None:
    # Crea las ventanas antes de añadir trackbars o mostrar el contenido
    cv2.namedWindow(window_params['capture_window_name'])
    cv2.namedWindow(window_params['detection_window_name'])
    cv2.namedWindow('rect')

    # Obtiene la longitud total del video en frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Añade un trackbar al 'Input video' para controlar la posición del video
    cv2.createTrackbar('Frame', window_params['capture_window_name'], 0, total_frames-1, lambda x: None)

    last_trackbar_pos = -1  # Inicializa con un valor que nunca será una posición válida del trackbar

    while cap.isOpened():
        # Lee la posición actual del trackbar
        trackbar_pos = cv2.getTrackbarPos('Frame', window_params['capture_window_name'])

        # Verifica si la posición del trackbar ha cambiado
        if trackbar_pos != last_trackbar_pos:
            # Solo actualiza la posición del video si la posición del trackbar ha cambiado
            cap.set(cv2.CAP_PROP_POS_FRAMES, trackbar_pos)
            last_trackbar_pos = trackbar_pos

        ret, frame = cap.read()
        if not ret:
            print("ERROR! - current frame could not be read")
            break

        frame = rescale_frame(frame, args.frame_resize_percentage)
        frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Filtros HSV
        mask1 = cv2.inRange(frame_HSV, (HSV_params_pasto['low_H'], HSV_params_pasto['low_S'], HSV_params_pasto['low_V']),
                                        (HSV_params_pasto['high_H'], HSV_params_pasto['high_S'], HSV_params_pasto['high_V']))

        mask2 = cv2.inRange(frame_HSV, (HSV_params_pista['low_H'], HSV_params_pista['low_S'], HSV_params_pista['low_V']),
                                        (HSV_params_pista['high_H'], HSV_params_pista['high_S'], HSV_params_pista['high_V']))

        
        # Combinación de máscaras y su inversión
        combined_mask = cv2.bitwise_or(mask1, mask2)
        inverted_mask = cv2.bitwise_not(combined_mask)

        # Uso correcto de la máscara invertida para encontrar contornos
        cnts, _ = cv2.findContours(combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Si no hay contornos, continúa con la siguiente iteración del bucle
        if not cnts:
            continue

        # Encuentra el contorno con la mayor área
        largest_cnt = max(cnts, key=cv2.contourArea)

        # Obtiene el rectángulo del contorno más grande y ajusta su tamaño si es necesario
        x, y, w, h = cv2.boundingRect(largest_cnt)  # x, y son las coordenadas del punto superior izquierdo, w, h son el ancho y la altura
        
        # Aquí puedes ajustar el tamaño del rectángulo fijo
        fixed_width, fixed_height = 30, 30  # Tamaño fijo para el rectángulo
        x_center, y_center = x + w // 2, y + h // 2

        # Ajusta las coordenadas para centrar el rectángulo fijo sobre el contorno más grande
        x_fixed = max(0, x_center - fixed_width // 2)
        y_fixed = max(0, y_center - fixed_height // 2)

        # Dibuja el rectángulo fijo
        cv2.rectangle(frame, (x_fixed, y_fixed), (x_fixed + fixed_width, y_fixed + fixed_height), (0, 255, 0), 2)

        cv2.imshow(window_params['capture_window_name'], frame)
        cv2.imshow(window_params['detection_window_name'], combined_mask)
        cv2.imshow('rect', inverted_mask)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break


def close_windows(cap):
    """
    This function closes the windows and releases the camera

    Args:
        cap: video capture object

    """
    cv2.destroyAllWindows()
    cap.release()

def run_pipeline(args):
    cap = initialise_camera(args)
    segment_object(cap, args)
    close_windows(cap)

if __name__ == '__main__':
    args = parse_cli_data()
    run_pipeline(args)
