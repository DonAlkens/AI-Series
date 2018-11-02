from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput


class LoginDisplay(GridLayout):

    def __init__(self, **kwargs):
        super(LoginDisplay, self).__init__()


class Simplekivy(App):
    def build(self):
        return LoginDisplay()


if __name__ == '__main__':
    Simplekivy().run()