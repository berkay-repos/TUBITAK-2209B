import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

# Define initial parameters
init_nx = 0
nx_range = np.linspace(-3, 3, 50)
init_gamma = 0
gamma_range = np.linspace(-180, 180, 50)
init_a = 0
a_range = np.linspace(-10, 10, 50)

linecolor = "#1a66c2"
bg_primary = "#2e2e2e"
bg_secondary = "#333"
fontcolor = ".85"
canvascolor = "#777"  # White area surrounding fig
gridcolor = "#666"
gridcolor_secondary = "#7f7f7f"


def f(nx, gamma, a=-100):
    if a == -100:
        return 9.81 * (nx - np.sin(np.deg2rad(gamma)))
    else:
        return a / 9.81 + np.sin(np.deg2rad(gamma))


fig, ax = plt.subplots(facecolor=canvascolor)
line, = ax.plot(nx_range, f(nx_range, init_gamma), lw=2, color=linecolor)
ax.grid(which='both')
ax.set_facecolor(bg_primary)
ax.margins(x=0)
ax.tick_params(labelcolor=fontcolor)
ax.set_xlabel('$n_x$', color=fontcolor)
ax.set_ylabel('$a$', color=fontcolor)
ax.set_title("Acceleration vs $n_x$ with varying $\gamma$",
             color="#fff")

gamma_ax = plt.axes([0.15, 0.1, 0.2, 0.03], facecolor=bg_primary)
gamma_slider = Slider(ax=gamma_ax,
                      label="$\gamma$",
                      valmin=-90,
                      valmax=90,
                      valinit=0,
                      valstep=5)
nx_ax = plt.axes([0.4, 0.1, 0.2, 0.03], facecolor=bg_primary)
nx_slider = Slider(ax=nx_ax, label="$n_x$", valmin=-6, valmax=5, valinit=0)
a_ax = plt.axes([0.7, 0.1, 0.2, 0.03], facecolor=bg_primary)
a_slider = Slider(ax=a_ax, label="$a$", valmin=-10, valmax=10, valinit=0)


def radio_update(label):
    if label == "$\gamma$":
        line.set_ydata(f(nx_range, gamma_slider.val))
        line.set_xdata(nx_range)
        ax.set_xlim(nx_range[0], nx_range[-1])
        ax.set_ylim(a_range[0], a_range[-1])
        fig.canvas.draw_idle()
        ax.set_xlabel('$n_x$', color=".7", weight=800)
        ax.set_ylabel('$a$', color=".7", weight=400)
        ax.set_title("Acceleration vs $n_x$ with varying $\gamma$",
                     color="#fff")
    if label == "$n_x$":
        line.set_ydata(f(nx_slider.val, gamma_range))
        line.set_xdata(gamma_range)
        ax.set_xlim(gamma_range[0], gamma_range[-1])
        ax.set_ylim(a_range[0], a_range[-1])
        fig.canvas.draw_idle()
        ax.set_xlabel('$\gamma$', color=".8")
        ax.set_ylabel('$a$', color=".8")
        ax.set_title("Acceleration vs $\gamma$ with varying $n_x$",
                     color="#fff")
    if label == "$a$":
        line.set_ydata(f(nx_slider.val, gamma_range, a_slider.val))
        line.set_xdata(gamma_range)
        ax.set_xlim(gamma_range[0], gamma_range[-1])
        ax.set_ylim(nx_range[0], nx_range[-1])
        fig.canvas.draw_idle()
        ax.set_xlabel('$\gamma$', color=".8")
        ax.set_ylabel('$n_x$', color=".8")
        ax.set_title("$n_x$ vs $\gamma$ with varying acceleration",
                     color="#fff")


rax = plt.axes([0.05, 0.52, 0.05, 0.1], facecolor=bg_primary)
radio = RadioButtons(rax, ('$\gamma$', '$n_x$', '$a$'), activecolor="#21f3ff")
for slider in (a_slider, nx_slider, gamma_slider):
    slider.label.set_color(".8")
    slider.valtext.set_color(".8")
for labels in radio.labels:
    labels.set_color(".8")
radio.on_clicked(radio_update)


def func_update(val):
    if radio.value_selected == "$\gamma$":
        line.set_ydata(f(nx_range, gamma_slider.val))
        fig.canvas.draw_idle()
    if radio.value_selected == "$n_x$":
        line.set_ydata(f(nx_slider.val, gamma_range))
        fig.canvas.draw_idle()
    if radio.value_selected == "$a$":
        line.set_ydata(f(nx_slider.val, gamma_range, a_slider.val))
        fig.canvas.draw_idle()


for x in (gamma_slider, nx_slider, a_slider):
    x.on_changed(func_update)

# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.15, bottom=0.25)
plt.show()
