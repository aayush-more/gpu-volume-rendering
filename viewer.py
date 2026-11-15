"""Interactive viewer.

python viewer.py [dataset or path] [size]

Drag to rotate, sliders for camera / opacity / step, preset / MIP selector,
"s" saves the current frame to results/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RadioButtons, Slider

from camera import orbit_camera
from kernel import MODE_COMPOSITE, MODE_MIP
from render import Renderer, save_image
from tf import DEFAULT, preset_lut, presets_for
from volume import dataset_name, load_volume, make_test_sphere

def main():
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        source = DEFAULT
        FEATURE_FLAGS = int(sys.argv[1])
    else:
        source = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
        FEATURE_FLAGS = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if source == "sphere":
        data, spacing, name = make_test_sphere(), (1.0, 1.0, 1.0), "sphere"
    else:
        (data, spacing), name = load_volume(source), dataset_name(source)
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    presets = list(presets_for(name))
    print("Compiling kernel ...")
    renderer = Renderer(data, preset_lut(name, presets[0]), spacing)

    fig = plt.figure(figsize=(8, 9.5))
    ax = fig.add_axes([0.2, 0.3, 0.75, 0.65])
    ax.axis("off")
    image = ax.imshow(np.zeros((size, size, 3)), origin="lower")

    def slider(row, label, lo, hi, start):
        return Slider(fig.add_axes([0.3, 0.22 - row * 0.04, 0.55, 0.025]), label, lo, hi, valinit=start)

    azimuth = slider(0, "azimuth", 0, 360, 30)
    elevation = slider(1, "elevation", -89, 89, 20)
    distance = slider(2, "distance", 1.0, 4.0, 2.0)
    opacity = slider(3, "opacity x", 0.1, 4.0, 1.0)
    step = slider(4, "step (voxels)", 0.25, 2.0, 0.5)
    choice = RadioButtons(fig.add_axes([0.02, 0.6, 0.15, 0.15]), presets + ["MIP"])

    state = {"img": None}

    def update(_=None):
        preset = choice.value_selected
        mode = MODE_MIP if preset == "MIP" else MODE_COMPOSITE
        if mode == MODE_COMPOSITE:
            renderer.set_tf(preset_lut(name, preset, opacity.val))
        cam = orbit_camera(azimuth.val, elevation.val, distance.val, width=size, height=size)
        img, ms = renderer.render(cam, step=step.val, mode=mode, flags = FEATURE_FLAGS)
        state["img"] = img
        image.set_data(np.clip(img, 0, 1))
        ax.set_title(f"{name}  {size}x{size}  {preset}:  {ms:.1f} ms  ({1000 / ms:.0f} FPS)")
        fig.canvas.draw_idle()

    for s in (azimuth, elevation, distance, opacity, step):
        s.on_changed(update)
    choice.on_clicked(update)

    drag = {"xy": None}

    def on_press(event):
        if event.inaxes is ax:
            drag["xy"] = (event.x, event.y, azimuth.val, elevation.val)

    def on_move(event):
        if drag["xy"] is None or event.x is None:
            return
        x0, y0, az0, el0 = drag["xy"]
        elevation.eventson = False  # update both sliders with a single render
        elevation.set_val(float(np.clip(el0 - (event.y - y0) * 0.4, -89, 89)))
        elevation.eventson = True
        azimuth.set_val((az0 - (event.x - x0) * 0.4) % 360)

    def on_release(_event):
        drag["xy"] = None

    def on_key(event):
        if event.key == "s" and state["img"] is not None:
            Path("results").mkdir(exist_ok=True)
            out = f"results/viewer_{name}_{choice.value_selected}_{int(azimuth.val)}.png"
            save_image(state["img"], out)
            print(f"Saved {out}")

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    update()
    plt.show()


if __name__ == "__main__":
    main()
