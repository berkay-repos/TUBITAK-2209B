import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, RadioButtons

# Define initial parameters
init_nz = 0
nz_range = np.linspace(-3, 3, 50)
init_bank = 0
bank_range = np.linspace(-180, 180, 50)
init_gamma = -1
gamma_range = np.linspace(-180, 180, 50)
init_vel = 250
vel_range = np.linspace(-400, 400, 50)

linecolor = "#1a66c2"
bg_primary = "#2e2e2e"
bg_secondary = "#333"
fontcolor = ".85"
canvascolor = "#777"  # White area surrounding fig
gridcolor = "#666"
gridcolor_secondary = "#7f7f7f"


def f(nz, gamma, bank, vel, mode="head"):
    if mode == "head":
        return 9.81 * nz * np.sin(np.deg2rad(bank)) / vel / np.cos(gamma)
    else:
        return 9.81 * (nz * np.cos(np.deg1rad(bank)) - np.cos(gamma)) / vel


fig = plt.figure(facecolor=canvascolor)
gs = GridSpec(1, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
theta_line = ax1.plot(nz_range,
                      f(nz_range, init_gamma, init_bank, init_vel),
                      lw=2,
                      color=linecolor)
bank_line = ax1.plot(nz_range,
                     f(nz_range, init_gamma, init_bank, init_vel),
                     lw=2,
                     color=linecolor)
# rearrangeticks(ax, nx_range[0], nx_range[-1], -10, 10, 3, 8)
for axes in (ax1, ax2):
    axes.grid(which='both')
    axes.set_facecolor(bg_primary)
    axes.margins(x=0)
    axes.tick_params(labelcolor=fontcolor)
ax1.set_xlabel('$n_z$', color=fontcolor)
ax1.set_ylabel('$\dot{\\theta}$', color=fontcolor)
ax2.set_xlabel('$n_z$', color=fontcolor)
ax2.set_ylabel('$\dot{\gamma}$', color=fontcolor)

gamma_ax = plt.axes([0.148, 0.1, 0.343, 0.03], facecolor=bg_primary)
gamma_slider = Slider(ax=gamma_ax,
                      label="$\gamma$",
                      valmin=-90,
                      valmax=90,
                      valinit=0,
                      valstep=5)
bank_ax = plt.axes([0.558, 0.1, 0.343, 0.03], facecolor=bg_primary)
bank_slider = Slider(ax=bank_ax,
                     label="$\phi$",
                     valmin=-90,
                     valmax=90,
                     valinit=0,
                     valstep=5)

# def radio_update(label):
#     if label == "$\gamma$":
#         line.set_ydata(f(nx_range, gamma_slider.val))
#         line.set_xdata(nx_range)
#         ax.set_xlim(nx_range[0], nx_range[-1])
#         ax.set_ylim(a_range[0], a_range[-1])
#         fig.canvas.draw_idle()
#         ax.set_xlabel('$n_x$', color=".7", weight=800)
#         ax.set_ylabel('$a$', color=".7", weight=400)
#         ax.set_title("Acceleration vs $n_x$ with varying $\gamma$",
#                      color="#fff")
#     if label == "$n_x$":
#         line.set_ydata(f(nx_slider.val, gamma_range))
#         line.set_xdata(gamma_range)
#         ax.set_xlim(gamma_range[0], gamma_range[-1])
#         ax.set_ylim(a_range[0], a_range[-1])
#         fig.canvas.draw_idle()
#         ax.set_xlabel('$\gamma$', color=".8")
#         ax.set_ylabel('$a$', color=".8")
#         ax.set_title("Acceleration vs $\gamma$ with varying $n_x$",
#                      color="#fff")
#     if label == "$a$":
#         line.set_ydata(f(nx_slider.val, gamma_range, a_slider.val))
#         line.set_xdata(gamma_range)
#         ax.set_xlim(gamma_range[0], gamma_range[-1])
#         ax.set_ylim(nx_range[0], nx_range[-1])
#         fig.canvas.draw_idle()
#         ax.set_xlabel('$\gamma$', color=".8")
#         ax.set_ylabel('$n_x$', color=".8")
#         ax.set_title("$n_x$ vs $\gamma$ with varying acceleration",
#                      color="#fff")

rax = plt.axes([0.05, 0.52, 0.05, 0.1], facecolor=bg_primary)
radio = RadioButtons(rax, ('$\gamma$', '$n_x$', '$a$'), activecolor="#21f3ff")
for slider in (gamma_slider, bank_slider):
    slider.label.set_color(".8")
    slider.valtext.set_color(".8")
for labels in radio.labels:
    labels.set_color(".8")
# radio.on_clicked(radio_update)


def func_update(val):
    # if radio.value_selected == "$\gamma$":
    theta_line.set_ydata(f(nz_range, init_gamma, init_bank, init_vel))
    bank_line.set_ydata(f(nz_range, init_gamma, init_bank, init_vel),
                        mode="bank")
    fig.canvas.draw_idle()
    # if radio.value_selected == "$n_x$":
    #     theta_line.set_ydata(f(nz_range, init_gamma, init_bank, init_vel))
    #     bank_line.set_ydata(f(nz_range, init_gamma, init_bank, init_vel),
    #                         mode="bank")
    #     fig.canvas.draw_idle()
    # if radio.value_selected == "$a$":
    #     theta_line.set_ydata(f(nz_range, init_gamma, init_bank, init_vel))
    #     bank_line.set_ydata(f(nz_range, init_gamma, init_bank, init_vel),
    #                         mode="bank")
    #     fig.canvas.draw_idle()


for x in (gamma_slider, bank_slider):
    x.on_changed(func_update)

# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.15, bottom=0.25)
plt.show()
