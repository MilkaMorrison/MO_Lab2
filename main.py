from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QComboBox, QTextEdit, QGroupBox, QSpinBox)
from PyQt6.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from gradient_methods import gradient_descent, steepest_gradient_descent, f


class GradientApp(QMainWindow):
    def setup_ui(self):
        self.setWindowTitle("Градиентный спуск")
        self.resize(1000, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        left_panel = QWidget()
        left_panel.setFixedWidth(380)
        left_layout = QVBoxLayout(left_panel)

        self.setup_controls()
        left_layout.addLayout(self.controls_layout)

        main_layout.addWidget(left_panel)

        self.setup_plots()
        main_layout.addWidget(self.plot_widget)

    def setup_controls(self):
        self.controls_layout = QVBoxLayout()

        group_font = QFont("Times New Roman", 12)
        label_font = QFont("Times New Roman", 13)
        button_font = QFont("Times New Roman", 14)
        output_font = QFont("Times New Roman", 12)

        func_group = QGroupBox("Параметры функции")
        func_group.setFont(group_font)
        func_layout = QVBoxLayout()

        self.a_input = self.create_input("Коэффициент a:")
        self.b_input = self.create_input("Коэффициент b:")
        self.c_input = self.create_input("Коэффициент c:")

        func_layout.addWidget(self.a_input)
        func_layout.addWidget(self.b_input)
        func_layout.addWidget(self.c_input)
        func_group.setLayout(func_layout)

        algo_group = QGroupBox("Параметры алгоритма")
        algo_group.setFont(group_font)
        algo_layout = QVBoxLayout()

        self.x_input = self.create_input("Начальный вектор x:")
        self.e1_input = self.create_input("Точность e1:")
        self.e2_input = self.create_input("Точность e2:")
        self.t_input = self.create_input("Шаг t:")

        m_widget = QWidget()
        m_layout = QHBoxLayout(m_widget)
        m_label = QLabel("Максимальное число итераций M:")
        m_label.setFont(label_font)
        self.m_spin = QSpinBox()
        self.m_spin.setRange(1, 100)
        self.m_spin.setValue(10)
        m_layout.addWidget(m_label)
        m_layout.addWidget(self.m_spin)

        self.method_combo = QComboBox()
        self.method_combo.setFont(label_font)
        self.method_combo.addItems([
            "Метод градиентного спуска с постоянным шагом",
            "Метод наискорейшего градиентного спуска"
        ])

        algo_layout.addWidget(self.x_input)
        algo_layout.addWidget(self.e1_input)
        algo_layout.addWidget(self.e2_input)
        algo_layout.addWidget(self.t_input)

        algo_layout.addWidget(m_widget)
        method_label = QLabel("Выберите метод:")
        method_label.setFont(label_font)
        algo_layout.addWidget(method_label)
        algo_layout.addWidget(self.method_combo)
        algo_group.setLayout(algo_layout)

        self.calc_button = QPushButton("Рассчитать")
        self.calc_button.setFont(button_font)
        self.calc_button.clicked.connect(self.run_calculation)

        self.output = QTextEdit()
        self.output.setFont(output_font)
        self.output.setReadOnly(True)

        self.controls_layout.addWidget(func_group)
        self.controls_layout.addWidget(algo_group)
        self.controls_layout.addWidget(self.calc_button)
        self.controls_layout.addWidget(self.output)

    def setup_plots(self):
        self.plot_widget = QWidget()
        plot_layout = QVBoxLayout(self.plot_widget)

        plot_layout.setStretch(0, 2)  # 3D график
        plot_layout.setStretch(1, 1)

        # 3D график
        self.figure_3d = Figure()
        self.canvas_3d = FigureCanvas(self.figure_3d)
        plot_layout.addWidget(self.canvas_3d)

        # 2D график
        self.figure_2d = Figure()
        self.canvas_2d = FigureCanvas(self.figure_2d)
        plot_layout.addWidget(self.canvas_2d)

    def create_input(self, label_text, default=""):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        label = QLabel(label_text)
        input_field = QLineEdit(default)
        layout.addWidget(label)
        layout.addWidget(input_field)
        return widget

    def run_calculation(self):
        try:
            a = float(self.a_input.findChild(QLineEdit).text())
            b = float(self.b_input.findChild(QLineEdit).text())
            c = float(self.c_input.findChild(QLineEdit).text())
            x = list(map(float, self.x_input.findChild(QLineEdit).text().split()))
            e1 = float(self.e1_input.findChild(QLineEdit).text())
            e2 = float(self.e2_input.findChild(QLineEdit).text())
            t = float(self.t_input.findChild(QLineEdit).text())
            M = int(self.m_spin.value())
            method = self.method_combo.currentText()

            if method == "Метод градиентного спуска с постоянным шагом":
                result, points = gradient_descent(a, b, c, x, e1, e2, M, t)
            else:
                result, points = steepest_gradient_descent(a, b, c, x, e1, e2, M)

            self.output.clear()
            self.output.append(f"Найденная точка: ({result['point'][0]:.8f}, {result['point'][1]:.8f})")
            self.output.append(f"Значение функции: {result['value']:.8f}")
            self.output.append(f"Итераций: {result['iterations']}")
            self.update_plots(a, b, c, result['point'], points)

        except Exception as e:
            self.output.append(f"Ошибка: {str(e)}")

    def update_plots(self, a, b, c, final_point, points):
        points = np.array(points)
        x1 = points[:, 0]
        x2 = points[:, 1]

        # 3D график
        self.figure_3d.clear()
        ax = self.figure_3d.add_subplot(111, projection='3d')

        X1, X2 = np.meshgrid(np.linspace(min(x1) - 1, max(x1) + 1, 50),
                             np.linspace(min(x2) - 1, max(x2) + 1, 50))
        Y = f(a, b, c, X1, X2)

        ax.plot_surface(X1, X2, Y, cmap='viridis', alpha=0.7)
        ax.plot(x1, x2, f(a, b, c, x1, x2), 'r.-', markersize=5)
        ax.set_box_aspect([1, 1, 0.8])

        # 2D график
        self.figure_2d.clear()
        ax2 = self.figure_2d.add_subplot(111)

        x1_min, x1_max = min(x1) - 0.2, max(x1) + 0.2
        x2_min, x2_max = min(x2) - 0.2, max(x2) + 0.2

        X1, X2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                             np.linspace(x2_min, x2_max, 100))
        Y = f(a, b, c, X1, X2)

        cntr = ax2.contourf(X1, X2, Y, levels=15, cmap='viridis', alpha=0.5)
        ax2.contour(X1, X2, Y, levels=15, colors='gray', linewidths=0.5)

        ax2.plot(x1, x2, 'r-', linewidth=1)
        ax2.scatter(x1, x2, c='red', s=50, edgecolors='white')
        ax2.scatter(final_point[0], final_point[1], c='yellow', s=100, edgecolors='black')

        ax2.set_xlim(x1_min, x1_max)
        ax2.set_ylim(x2_min, x2_max)
        ax2.grid(True, linestyle='--', alpha=0.5)
        self.figure_2d.colorbar(cntr, ax=ax2, label='Значение функции')

        self.figure_3d.tight_layout()
        self.figure_2d.tight_layout()
        self.canvas_3d.draw()
        self.canvas_2d.draw()

if __name__ == "__main__":
    app = QApplication([])
    window = GradientApp()
    window.setup_ui()
    window.show()
    app.exec()