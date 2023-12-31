import argparse
import PySimpleGUI as sg
from PIL import Image, ImageEnhance
import numpy as np
import cv2
from io import BytesIO
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# import matplotlib.pyplot as plt
import matplotlib  
matplotlib.use('TkAgg')
import os.path


'''
Title: np_im_to_data
Author: Faisal Qureshi
Date: December 2023
Type: Source Code 
Availability: Provided for lab 2 on September 2023 for CSCI 3240U 
'''
# this will to convert numpy image to PySimpleGUI-compatible data
def np_im_to_data(im):
    array = np.array(im, dtype=np.uint8)
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format='PNG')
        data = output.getvalue()
    return data

'''
Title: construct_image_histogram
Author: Faisal Qureshi
Date: December 2023
Type: Source Code 
Availability: Provided for lab 2 on September 2023 for CSCI 3240U 
'''
# construct image histogram
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
            range=(0, 180), 
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
            range=(0, 100), 
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
            range=(0, 100), 
            default_value=0, 
            size=(40, 15),
            enable_events=True,
            orientation='horizontal', 
            key='-V-'),
            sg.Button('Reset', key='Reset_V')]

    # this is the defined  layout
    layout = [
        [sg.Input(size=(25, 1), key="-FILE-"), sg.FileBrowse(), sg.Button('Load Image')],
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
 
        if event == sg.WINDOW_CLOSED or event == 'Exit':
            break
        
        # loads the image
        if event == 'Load Image':
            filename = values['-FILE-']
            if os.path.exists(filename):
                np_image = cv2.imread(filename)
                np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)

                dim = aspect_ratio(np_image, 400) # finds the aspect ratio dimentions for a certain width
                np_image = cv2.resize(np_image, dim, interpolation=cv2.INTER_LINEAR)

                # set orginal values for reset functions
                original_image = np_image
                original_hsv_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2HSV)
                original_h = original_hsv_image[:,:,0]
                original_s = original_hsv_image[:,:,1]
                original_v = original_hsv_image[:,:,2]

                # this makes sure that if the 'Reset_V' button is pressed without any changes being made the program doesn't break
                h = original_h
                s = original_s

                window['-OUTPUT-'].erase()
                window['-OUTPUT-'].draw_image(data=np_im_to_data(np_image), location=(0, height))

        # convert the image to grayscale
        if event == 'Greyscale':
            # convert to greyscale
            np_image = greyscale(np_image)

            # convert numpy array to data
            gray_image_data = np_im_to_data(np_image)
            
            # thi displays the greyscale image
            window['-OUTPUT-'].draw_image(data=gray_image_data, location=(0, height))

        # Histogram equalization
        if event == 'Histogram Equalization':
            # this performs histogram equalization
            np_image = histogram_equalization(np_image)
            
            # convert numpy array to data
            rgb_image_data = np_im_to_data(np_image)

            # this displays the equalized image
            window['-OUTPUT-'].draw_image(data=rgb_image_data, location=(0, height))

        #  filter to apply the blur 
        if event == 'Blur':
            np_image = apply_blur(np_image)
            window['-OUTPUT-'].draw_image(data=np_im_to_data(np_image), location=(0, height))

        # applying of the sharpness filter
        if event == 'Sharpness':
            np_image = apply_sharpness(np_image)
            window['-OUTPUT-'].draw_image(data=np_im_to_data(np_image), location=(0, height))

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
            h = original_h # sets the hue value to original hue value
            np_image =  reset_image(np_image, 0, original_h)
            window['-OUTPUT-'].draw_image(data=np_im_to_data(np_image), location=(0,height))
        
        # resets the saturation value
        if event == 'Reset_S':
            s = original_s # sets the saturation value to original saturation value
            np_image =  reset_image(np_image, 1, original_s)
            window['-OUTPUT-'].draw_image(data=np_im_to_data(np_image), location=(0,height))
        
        # resets the colour value
        if event == 'Reset_V':
            hsv_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2HSV)
            hsv_image[:,:,0] = h # sets the hue value to userr given input
            hsv_image[:,:,1] = s # sets the saturation value to userr given input
            hsv_image[:,:,2] = original_v
            np_image =  cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

            # np_image =  reset_image(np_image, 2, original_v)
            window['-OUTPUT-'].draw_image(data=np_im_to_data(np_image), location=(0,height))

        # this saaves the edited image
        if event == 'Save':
            save_path = sg.popup_get_file('Save As', save_as=True, file_types=(("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg")))
            if save_path:
                cv2.imwrite(save_path, cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR))
                sg.popup('Image saved successfully!', auto_close_duration=2)

    window.close()

'''
Title: main
Author: Faisal Qureshi
Date: December 2023
Type: Source Code 
Availability: Provided for lab 2 on September 2023 for CSCI 3240U 
'''
# the main function
def main():
    image = np.zeros((300,400,3))

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
# this  function applies histogram equalization
def histogram_equalization(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

# a function to apply blur filter
def apply_blur(image):
    return custom_gaussian_blur(image)

# this is your custom Gaussian blur implementation
def custom_gaussian_blur(image):
    # Define the size of the kernel (standard deviation)
    kernel_size = 15
    kernel_size_half = kernel_size // 2

    # Generate the Gaussian kernel
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * (kernel_size / 2)**2)) * 
                     np.exp(-((x - kernel_size_half)**2 + (y - kernel_size_half)**2) / (2 * (kernel_size / 2)**2)),
        (kernel_size, kernel_size)
    )

    # Normalize the kernel
    kernel = kernel / np.sum(kernel)

    # Apply the convolution to each channel separately
    output_image = np.zeros_like(image)
    for channel in range(image.shape[2]):
        filtered_channel = apply_filter_to_image(image[:, :, channel], kernel)

        # Pad the filtered channel to match the output image dimensions
        pad_height = (output_image.shape[0] - filtered_channel.shape[0]) // 2
        pad_width = (output_image.shape[1] - filtered_channel.shape[1]) // 2
        padded_filtered_channel = np.pad(filtered_channel, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

        output_image[:, :, channel] = padded_filtered_channel

    return output_image.astype(np.uint8)

# this function to apply a filter to a patch in the image
def apply_filter_to_patch(patch, filter):
    return np.sum(patch * filter)

# this function will apply a filter to the entire image (custom implementation)
def apply_filter_to_image(image, filter):
    h, w = image.shape
    h_filter, w_filter = filter.shape
    result = np.zeros((h - h_filter + 1, w - w_filter + 1))
    for i in range(h - h_filter + 1):
        for j in range(w - w_filter + 1):
            patch = image[i:i + h_filter, j:j + w_filter]
            result[i, j] = apply_filter_to_patch(patch, filter)
    return result


# a function to apply sharpness filter
def apply_sharpness(image):
    enhancer = ImageEnhance.Sharpness(Image.fromarray(image))
    return np.array(enhancer.enhance(2.0))

def adjust_image(image, value, event):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) # converts image values into HSV

    # adjusts the channel based on the event given
    if event == 'H':
        hsv_image[:,:,0] = value # sets the hue channel to the original value
    elif event == 'S':
        hsv_image[:, :, 1] = hsv_image[:,:,1] + value # add the input value to the saturation channel
    elif event == 'V':
       hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] + value, 0, 255) # add the input value to the value channel
    
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB) # returns the adjusted image converted back to RBG 

def reset_image(image, c, original):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) # converts image values into HSV
    hsv_image[:,:,c] = original # sets the channel to the original value based on given channel value

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB) # returns the image converted back to RBG 

if __name__ == '__main__':
    main()

