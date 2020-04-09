from threading import Thread
import random as rnd
import time

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput


class GenericEquationSolver:

    def __init__(self, a, b, c, d, y, pop_len):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.y = y
        self.pop_len = pop_len
        self.solutions = []
        self.is_finish = False
        self.population = []
        self.fitnesses = []

        self.generate_population()
        self.calculate_fitnesses()
        if not self.is_finish:
            self.generate_children()
            self.calculate_fitnesses()
            if not self.is_finish:
                while True:
                    self.generate_children()
                    self.calculate_fitnesses()
                    if self.is_finish:
                        break
        self.get_solutions()

    def get_solutions(self):
        for i in range(len(self.population)):
            if self.fitnesses[i] == 0:
                self.solutions.append(self.population[i])

    def generate_population(self):
        for i in range(self.pop_len):
            py = round(self.y/4)
            self.population.append([rnd.randrange(py), rnd.randrange(py), rnd.randrange(py), rnd.randrange(py)])

    def calculate_fitnesses(self):
        reverse_deltas = []
        for genotype in self.population:
            delta = abs(self.a * genotype[0] + self.b * genotype[1] + self.c * genotype[2] + self.d * genotype[3] - self.y)
            if delta == 0:
                self.is_finish = True
                reverse_deltas.append(0)
            else:
                reverse_deltas.append(1 / delta)

        sum_reverse_deltas = sum(reverse_deltas)
        for i in reverse_deltas:
            self.fitnesses.append(i / sum_reverse_deltas)

    def crossover(self, the_parents):
        rand = rnd.randrange(1, 4)
        for r in range(rand):
            the_parents[0][r], the_parents[1][r] = the_parents[1][r], the_parents[0][r]
        for r in range(rand, 4):
            the_parents[0][r], the_parents[1][r] = the_parents[1][r], the_parents[0][r]
        return [the_parents[0], the_parents[1]]

    def generate_children(self):
        sorted_population = list(sorted(zip(self.population, self.fitnesses), key=lambda x: x[1], reverse=True))
        children = []
        for i in range(0, len(sorted_population), 2):
            cross = self.crossover([sorted_population[i][0], sorted_population[i + 1][0]])
            children.append(cross[0])
            children.append(cross[1])
        self.population = children


class Perceptron():
    def __init__(self, learning_rate, deadline, iterations):
        self.learning_rate = learning_rate
        self.deadline = deadline
        self.iterations = iterations
        self.weights = self.train(learning_rate, deadline, iterations)

    def predict(self, dot, weights, P):
        sum = 0
        for i in range(len(dot)):
            sum += weights[i] * dot[i]
        return 1 if sum > P else 0

    def train(self, learning_rate, deadline, iterations):
        threshold = 4
        data = [(0, 6), (1, 5), (3, 3), (2, 4)]
        n = len(data[0])
        weights = [0.001, -0.004]
        outputs = [0, 0, 0, 1]

        start = time.time()
        for i in range(iterations):
            total_error = 0
            for i in range(len(outputs)):
                prediction = self.predict(data[i], weights, threshold)
                error = outputs[i] - prediction
                total_error += error
                for j in range(n):
                    delta = learning_rate * data[i][j] * error
                    weights[j] += delta
            if total_error == 0 or time.time() - start > deadline:
                break
        return ['w1 = ' + str(weights[0]), 'w2 = ' + str(weights[1])]


class MyButton1(Button):
    def on_press(self, *args):
        Thread(target=self.worker).start()

    def worker(self):

        start = time.time()

        try:
            n = int(App.get_running_app().ti1.text)
        except:
            n = 39746930799
            App.get_running_app().ti1.text = str(n)
        i = 2
        results = []
        while i * i <= n:
            while n % i == 0:
                results.append(i)
                n = n / i
            i = i + 1
        if n > 1:
            results.append(round(n))

        res = 'Results = '
        for i in results:
            res += str(i) + ', '

        res = res[:-2]

        finish = time.time() - start

        App.get_running_app().lb1.text = res
        App.get_running_app().lb1_1.text += str(finish)


class MyButton2(Button):
    def on_press(self, *args):
        Thread(target=self.worker).start()

    def worker(self):
        start = time.time()
        try:
            learning_rate = float(App.get_running_app().ti2.text)
            deadline = float(App.get_running_app().ti3.text)
            iterations = int(App.get_running_app().ti4.text)
        except:
            learning_rate = 0.1
            deadline = 5
            iterations = 500
            App.get_running_app().ti2.text = str(learning_rate)
            App.get_running_app().ti3.text = str(deadline)
            App.get_running_app().ti4.text = str(iterations)
        perceptron = Perceptron(learning_rate, deadline, iterations)

        finish = time.time() - start

        App.get_running_app().lb2.text = perceptron.weights[0] + ', ' + perceptron.weights[1]
        App.get_running_app().lb2_1.text += str(finish)


class MyButton3(Button):
    def on_press(self):
        Thread(target=self.worker).start()

    def worker(self):
        a = int(App.get_running_app().ti3_1.text)
        b = int(App.get_running_app().ti3_2.text)
        c = int(App.get_running_app().ti3_3.text)
        d = int(App.get_running_app().ti3_4.text)
        y = int(App.get_running_app().ti3_5.text)
        pop_len = int(App.get_running_app().ti3_6.text)
        ges = GenericEquationSolver(a, b, c, d, y, pop_len)
        App.get_running_app().lb3.text = str(ges.solutions)


class MainApp(App):
    # Lab 3.1
    ti1 = TextInput(text="Input here number for factorization")
    lb1 = Label(text="Hello! Its Lab#3.1. Results will be here.")
    lb1_1 = Label(text='Calc time = ')
    bt1 = MyButton1(text="Calculate")
    res = 'empty'
    # Lab 3.2
    lb2 = Label(text="Hello! Its Lab#3.2. Weights will be here.")
    lb2_1 = Label(text='Calc time = ')
    ti2 = TextInput(text="Input here learning rate")
    ti3 = TextInput(text="Input here deadline in seconds")
    ti4 = TextInput(text="Input here number of iterations")
    bt2 = MyButton2(text="Calculate")
    # Lab 3.3
    lb3 = Label(text="Hello! Its Lab#3.3. Results will be here.")
    ti3_1 = TextInput(text="a = ")
    ti3_2 = TextInput(text="b = ")
    ti3_3 = TextInput(text="c = ")
    ti3_4 = TextInput(text="d = ")
    ti3_5 = TextInput(text="y = ")
    ti3_6 = TextInput(text="population length = ")
    bt3 = MyButton3(text="Calculate")

    def build(self):
        bl1 = BoxLayout(orientation='vertical')

        bl1_1 = BoxLayout(padding=20, spacing=20)

        bl1_1_1 = BoxLayout(spacing=20, orientation='vertical')
        bl1_1_1.add_widget(self.lb1)
        bl1_1_1.add_widget(self.lb1_1)

        bl1_1.add_widget(bl1_1_1)
        bl1_1.add_widget(self.ti1)
        bl1_1.add_widget(self.bt1)

        bl1_2 = BoxLayout(padding=10, spacing=10)

        bl1_2_1 = BoxLayout(spacing=10, orientation='vertical')
        bl1_2_1.add_widget(self.lb2)
        bl1_2_1.add_widget(self.lb2_1)

        bl1_2.add_widget(bl1_2_1)

        bl1_2_2 = BoxLayout(spacing=10, orientation='vertical')
        bl1_2_2.add_widget(self.ti2)
        bl1_2_2.add_widget(self.ti3)
        bl1_2_2.add_widget(self.ti4)

        bl1_2.add_widget(bl1_2_2)
        bl1_2.add_widget(self.bt2)

        bl1_4 = BoxLayout(spacing=10, padding=10)

        bl1_3 = BoxLayout(spacing=2, orientation='vertical')
        bl1_3.add_widget(self.ti3_1)
        bl1_3.add_widget(self.ti3_2)
        bl1_3.add_widget(self.ti3_3)
        bl1_3.add_widget(self.ti3_4)
        bl1_3.add_widget(self.ti3_5)
        bl1_3.add_widget(self.ti3_6)
        bl1_4.add_widget(bl1_3)

        bl1_5 = BoxLayout(spacing=20, orientation='vertical')
        bl1_5.add_widget(self.lb3)
        bl1_5.add_widget(self.bt3)
        bl1_4.add_widget(bl1_5)

        bl1.add_widget(bl1_1)
        bl1.add_widget(bl1_2)
        bl1.add_widget(bl1_4)

        return bl1


if __name__ == '__main__':
    MainApp().run()
