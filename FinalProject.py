import argparse
import PySimpleGUI as sg
from PIL import Image, ImageEnhance
import numpy as np
import cv2
from io import BytesIO
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# this will to convert numpy image to PySimpleGUI-compatible data
def np_im_to_data(im):
    array = np.array(im, dtype=np.uint8)
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format='PNG')
        data = output.getvalue()
    return data

#  construct image histogram
def construct_image_histogram(np_image):
    L = 256
    bins = np.arange(L+1)
    hist, _ = np.histogram(np_image, bins)
    return hist

# display image in PySimpleGUI window
def display_image(np_image):
    # convert numpy array to data that sg.Graph can understand
    image_data = np_im_to_data(np_image)
    
    height = 300
    width = 400

    # set orginal values for reset functions
    original_image = np_image
    original_hsv_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2HSV)
    original_h = original_hsv_image[:,:,0]
    original_s = original_hsv_image[:,:,1]
    original_v = original_hsv_image[:,:,2]
    
    
    # output image frame
    frame_output = [[sg.Graph(
        canvas_size=(width, height),
        graph_bottom_left=(0, 0),
        graph_top_right=(width, height),
        key='-OUTPUT-',
        background_color='white',
        change_submits=True,
        drag_submits=True)]]

    #these are the  Sliders for adjusting hue, saturation, and vibrancy
    hueSlider = [
        sg.Text(
            'H', 
            enable_events=True,
            key='-TEXT-H-',
            justification='left'),
        sg.Slider(
            range=(-180, 180), 
            default_value=0, 
            size=(40, 15),
            enable_events=True,
            orientation='horizontal', 
            key='-H-'),
            sg.Button('Reset', key='Reset_H')]
    saturationSlider = [
        sg.Text(
            'S', 
            enable_events=True,
            key='-TEXT-S-',
            justification='left'),
        sg.Slider(
            range=(-100, 100), 
            default_value=0, 
            size=(40, 15),
            enable_events=True,
            orientation='horizontal', 
            key='-S-'),
            sg.Button('Reset', key='Reset_S')]
    valueSlider = [
        sg.Text(
            'V', 
            enable_events=True,
            key='-TEXT-V-',
            justification='left'),
        sg.Slider(
            range=(-100, 100), 
            default_value=0, 
            size=(40, 15),
            enable_events=True,
            orientation='horizontal', 
            key='-V-'),
            sg.Button('Reset', key='Reset_V')]

    # this is the defined  layout
    layout = [
        [sg.Column(frame_output)],
        # buttons for different filters
        [sg.Button('Greyscale'), sg.Button('Histogram Equalization'), sg.Button('Blur'), sg.Button('Sharpness')],
        [hueSlider, saturationSlider, valueSlider],
        [sg.Button('Save'), sg.Button('Exit'), sg.Button('Reset Image')]
    ]

    # Create the window
    window = sg.Window('Photo Editor', layout, finalize=True)    
    window['-OUTPUT-'].draw_image(data=image_data, location=(0, height))
        
    # this is the event loop
    while True:
        event, values = window.read()
 
        # this makes sure that if the reset buttons are pressed without any changes being made the program doesn't break

        if event == sg.WINDOW_CLOSED or event == 'Exit':
            break

        # convert the image to grayscale
        if event == 'Greyscale':
            # convert to greyscale
            gray_image = greyscale(np_image)

            # convert numpy array to data
            gray_image_data = np_im_to_data(gray_image)
            
            # thi displays the greyscale image
            window['-OUTPUT-'].draw_image(data=gray_image_data, location=(0, height))

        # Histogram equalization
        if event == 'Histogram Equalization':
            # this  performs histogram equalization
            rgb_image = histogram_equalization(np_image)
            
            # convert numpy array to data
            rgb_image_data = np_im_to_data(rgb_image)

            #  thi displays the  equalized image
            window['-OUTPUT-'].draw_image(data=rgb_image_data, location=(0, height))

        #  filter to apply the blur 
        if event == 'Blur':
            blurred_image = apply_blur(np_image)
            window['-OUTPUT-'].draw_image(data=np_im_to_data(blurred_image), location=(0, height))

        # applying of the sharpness filter
        if event == 'Sharpness':
            sharpened_image = apply_sharpness(np_image)
            window['-OUTPUT-'].draw_image(data=np_im_to_data(sharpened_image), location=(0, height))

        # these are the adjustments using sliders
        if event == '-H-':
            h = int(values['-H-'])
            np_image = adjust_image(np_image, h, 'H') 
            window['-OUTPUT-'].draw_image(data=np_im_to_data(np_image), location=(0,height))
        
        if event == '-S-':
            s = int(values['-S-'])
            np_image = adjust_image(np_image, s, 'S') 
            window['-OUTPUT-'].draw_image(data=np_im_to_data(np_image), location=(0,height))

        if event == '-V-':
            v = int(values['-V-'])
            np_image = adjust_image(np_image, v, 'V') 
            window['-OUTPUT-'].draw_image(data=np_im_to_data(np_image), location=(0,height))

        # resets the image
        if event == 'Reset Image':
            np_image = original_image
            reset_data = np_im_to_data(original_image)
            window['-OUTPUT-'].draw_image(data=reset_data, location=(0,height))   

        # resets the hue value
        if event == 'Reset_H':
            np_image =  reset_image(np_image, 0, original_h)
            window['-OUTPUT-'].draw_image(data=np_im_to_data(np_image), location=(0,height))
        
        # resets the saturation value
        if event == 'Reset_S':
            np_image =  reset_image(np_image, 1, original_s)
            window['-OUTPUT-'].draw_image(data=np_im_to_data(np_image), location=(0,height))
        
        # resets the colour value
        if event == 'Reset_V':
            hsv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2HSV)
            hsv_image[:,:,1] = s # sets the saturation value to userr given input
            np_image =  cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

            np_image =  reset_image(np_image, 2, original_v)
            window['-OUTPUT-'].draw_image(data=np_im_to_data(np_image), location=(0,height))

        # this saaves the edited image
        if event == 'Save':
            save_path = sg.popup_get_file('Save As', save_as=True, file_types=(("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg")))
            if save_path:
                cv2.imwrite(save_path, cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR))
                sg.popup('Image saved successfully!', auto_close_duration=2)

    window.close()

def adjust_image(image, value, event):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    if event == 'H':
        hsv_image[:,:,0] = value # sets the hue channel to the original value
    elif event == 'S':
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] + value, 0, 255) # sets the saturation channel to the input value
    elif event == 'V':
       hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] + value, 0, 255) # sets the saturation channel to the input value

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

def reset_image(image, i, original):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_image[:,:,i] = original # sets the saturation channel to the original value

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

# this  function applies histogram equalization
def histogram_equalization(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

# a function to apply blur filter
def apply_blur(image):
    return cv2.GaussianBlur(image, (15, 15), 0)

# a function to apply sharpness filter
def apply_sharpness(image):
    enhancer = ImageEnhance.Sharpness(Image.fromarray(image))
    return np.array(enhancer.enhance(2.0))

# the main function
def main():
    parser = argparse.ArgumentParser(description='A simple photo editing application.')

    parser.add_argument('file', action='store', help='Image file.')
    args = parser.parse_args()

    print(f'Loading {args.file} ... ', end='')
    image = cv2.imread(args.file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f'{image.shape}')

    print(f'Resizing the image ...', end='')
    dim = aspect_ratio(image, 400) 
    image = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
    print(f'{image.shape}')

    #this will display the image using the photo editing application
    display_image(image)

# this function is to calculate aspect ratio
def aspect_ratio(image, width):
    (h,w) = image.shape[:2] 
    ratio = width/float(w)
    dim = (width, int(h*ratio))
    return dim

# this function to convert image to greyscale
def greyscale(image):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB)

if __name__ == '__main__':
    main()
