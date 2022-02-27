import os
import pandas as pd
import keras
import kivy
import numpy
from PIL import Image
import PIL
import csv
import numpy as np
import tensorflow as tf
from kivy.app import App
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.screenmanager import ScreenManager
from kivy.uix.screenmanager import Screen
from kivy.properties import StringProperty, ObjectProperty, ListProperty, BooleanProperty
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from os.path import sep, expanduser, isdir, dirname
import sys
from kivy_garden.filebrowser import FileBrowser
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.recyclegridlayout import RecycleGridLayout
from pandasgui import show

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
class FirstScreen(Screen):
    pass


class SecondScreen(Screen):
    loadfile = ObjectProperty(None)

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def predict_single_function(self, ImagePath):
        saved_model3 = load_model('best_new_modeltest25.h5')
        pred_Class = ''
        image = load_img(ImagePath, color_mode='grayscale', target_size=(256, 256, 1))
        input_arr = img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to a batch.
        predictions = saved_model3.predict(input_arr)
        print('predictions : ' + str(predictions))
        pred_in = np.argmax(predictions, axis=1)
        if pred_in == 1: pred_Class = 'Normal'
        if pred_in == 0: pred_Class = 'Pneumonia Bacteria'
        if pred_in == 2: pred_Class = 'Pneumonia Virus'
        print('predictions Class : ' + str(pred_Class))
        self.ids.img_class.text = pred_Class

    def load(self, path, filename):
        l = os.path.join(path, filename[0])
        print(l)
        print(filename[0])
        self.ids.img.source = filename[0]
        self.predict_single_function(filename[0])
        self.dismiss_popup()

    def clearSecondPage(self):
        self.ids.img_class.text = ''
        self.ids.img.source = ''


class ThirdScreen(Screen):
    path = ""
    csvfile = ''
    file = 'enter zip path or select it'

    def open(self):
        print("doing")
        self.popup = Popup(title='Test popup',
                           content=self.explorer(),
                           size_hint=(None, None), size=(600, 600))
        self.popup.open()

    def explorer(self):
        if sys.platform == 'win':
            user_path = dirname(expanduser('~')) + sep + 'Documents'
        else:
            user_path = expanduser('~') + sep + 'Documents'
        browser = FileBrowser(select_string='Select',
                              favorites=[(user_path, 'Documents')], dirselect=True)
        browser.bind(
            on_success=self._fbrowser_success,
            on_canceled=self._fbrowser_canceled)
        return browser

    def _fbrowser_canceled(self, instance):
        print('cancelled, Close self.')
        self.popup.dismiss()

    def _fbrowser_success(self, instance):
        print(instance.selection[0])
        self.file = instance.selection[0]
        self.ids.path.text = self.file
        self.popup.dismiss()
        self.path = self.ids.path.text
        self.ids.path.text = ""

    def press(self):

        print(self.path)
        # self.ids.csv.text =self.path
        df = self.predict_function(self.path)
        ShowResults().graph(df)

    def predict_function(self, FilePath):
        Virus_counter1 = 0
        Normal_counter1 = 0
        BAC_counter1 = 0
        cases1 = 0
        predict_file = ""
        saved_model3 = load_model('vgg16_best_values.h5')
        normal_list = []
        bac_list = []
        virus_list = []
        for f in sorted(os.listdir(FilePath)):
            cases1 += 1
            image = load_img(os.path.join(FilePath, '', '')+f, target_size=(224, 224, 3))
            input_arr = img_to_array(image)
            input_arr = np.array([input_arr])  # Convert single image to a batch.
            imag = preprocess_input(input_arr)

            predictions=saved_model3.predict(imag)
            #predict_file = pd.DataFrame(predictions)
            #predict_file.columns=['bacteria','normal','virus']
            pred_in = numpy.argmax(predictions, axis=1)
            if pred_in == 1:
                normal_list.append(1)
                bac_list.append(0)
                virus_list.append(0)
            if pred_in == 0:
                normal_list.append(0)
                bac_list.append(1)
                virus_list.append(0)
            if pred_in == 2:
                normal_list.append(0)
                bac_list.append(0)
                virus_list.append(1)
        predict_file = pd.DataFrame({' Bacteria ':bac_list , "Normal":normal_list,"Virus":virus_list})
        #xl= predict_file.to_excel('prediction.xsl')
        print('MixData/test/VIR_PNEUMONIA file ')
        print(predict_file)
        print('cases total = ' + str(cases1))
        print('BAC_counter = ' + str(BAC_counter1))
        print('Normal_counter = ' + str(Normal_counter1))
        print('Virus_counter = ' + str(Virus_counter1))
        #predict_file.to_csv('predict_cnn.csv', index=False)
        #df = pd.read_csv('predict_cnn.csv')
        #xls = pd.read_excel('vgg16_prediction.csv')
        return predict_file


class ShowResults():
    def graph(self , df):
        show(df)


class WindowManager(ScreenManager):
    pass


kv = Builder.load_file('pneumonia.kv')


class PneumoniaApp(App):
    def build(self):
        return kv


if __name__ == '__main__':
    PneumoniaApp().run()
