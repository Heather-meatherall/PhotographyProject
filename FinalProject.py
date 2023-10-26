# Faisal Z. Qureshi
# www.vclab.ca

import argparse
import PySimpleGUI as sg
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def np_im_to_data(im):
    array = np.array(im, dtype=np.uint8)
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format='PNG')
        data = output.getvalue()
    return data

def construct_image_histogram(np_image):
    L = 256
    bins = np.arange(L+1)
    hist, _ = np.histogram(np_image, bins)
    return hist

def display_image(np_image):
    
    # Convert numpy array to data that sg.Graph can understand
    image_data = np_im_to_data(np_image)
    
    height = 300
    width = 400
    
    # output image frame
    frame_output = [[sg.Graph(
        canvas_size=(width, height),
        graph_bottom_left=(0, 0),
        graph_top_right=(width, height),
        key='-OUTPUT-',
        background_color='white',
        change_submits=True,
        drag_submits=True)]]

    slider1 = [
        sg.Text(
        'H', 
        enable_events=True,
        key='-TEXT-',
        justification='left'),
        sg.Slider(
        range=(0, 15), 
        default_value=0, 
        size=(40, 15),
        enable_events=True,
        orientation='horizontal', 
        key='-H-')]
    slider2 = [
        sg.Text(
        'S', 
        enable_events=True,
        key='-TEXT-',
        justification='left'),
        sg.Slider(
        range=(0, 15), 
        default_value=0, 
        size=(40, 15),
        enable_events=True,
        orientation='horizontal', 
        key='-S-')]
    slider3 = [
        sg.Text(
        'V', 
        enable_events=True,
        key='-TEXT-',
        justification='left'),
        sg.Slider(
        range=(0, 15), 
        default_value=0, 
        size=(40, 15),
        enable_events=True,
        orientation='horizontal', 
        key='-V-')]

        
    # Define the layout
    layout = [
        [sg.Column(frame_output)],
        # sliders to ajust hue, saturation and vibrency
        [sg.Button('Greyscale'), sg.Button('Histogram Equalization')],
        [slider1, slider2, slider3],
        [sg.Button('Exit')]
        ]

    # Create the window
    window = sg.Window('Display Image', layout, finalize=True)    
    window['-OUTPUT-'].draw_image(data=image_data, location=(0, height))
        
    # Event loop
    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == 'Exit':
            break

        # convert the image to grayscale
        if event == 'Greyscale':
            # convert to grey
            gray_image = greyscale(np_image)

            # convert numpy array to data
            gray_image_data = np_im_to_data(gray_image)
            
            # display image
            window['-OUTPUT-'].draw_image(data=gray_image_data, location=(0,height))

        
        if event == 'Histogram Equalization':
            # preform histogram equalization
            rgb_image = histogramEqual(np_image)
            
             # convert numpy array to data
            rgb_image_data = np_im_to_data(rgb_image)

            # display new output image
            window['-OUTPUT-'].draw_image(data=rgb_image_data, location=(0,height))
        
        # third filter

        # fourth filter
        
        #Slider events


    window.close()



def main():
    parser = argparse.ArgumentParser(description='A simple image viewer.')

    parser.add_argument('file', action='store', help='Image file.')
    args = parser.parse_args()

    print(f'Loading {args.file} ... ', end='')
    image = cv2.imread(args.file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f'{image.shape}')

    print(f'Resizing the image ...', end='')
    dim = aspect_ratio(image, 400) # finds the aspect ratio dimentions for a certain width
    image = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
    print(f'{image.shape}')

    display_image(image)


def aspect_ratio(image, width):
    (h,w) = image.shape[:2] # gets the height and width

    ratio = width/float(w)
    dim = (width, int(h*ratio))

    return dim

def greyscale(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grey

def histogramEqual(image):
    #convert image to hsv
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # preform histogram equalization
        hsv_hist = construct_image_histogram(hsv_image[:,:,2])

        pdf = hsv_hist / np.sum(hsv_hist)
        cdf = np.cumsum(pdf)
        adjustment_curve = cdf*255

        adjusted_image = adjustment_curve[hsv_image].astype(np.uint8)

        # convert image back to RGB
        rgb_image = cv2.cvtColor(adjusted_image, cv2.COLOR_HSV2RGB)
        return rgb_image




if __name__ == '__main__':
    main()