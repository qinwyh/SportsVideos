import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle
import json

VIDEO_PATH = 'test_single_view.mp4'
SAVE_DIR = 'metadata'
os.makedirs(SAVE_DIR, exist_ok=True)

# Read all frames into memory
cap = cv2.VideoCapture(VIDEO_PATH)
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)
cap.release()
print(f'Total frames: {len(frames)}')

def save_rect_metadata(frame_idx, rect_type, rect):
    """
    Save frame image with rectangles and single rectangle metadata to SAVE_DIR.
    """
    # Draw both rectangles on a copy for visualization
    frame = frames[frame_idx].copy()
    # Draw ball_flight_area if exists
    meta_path_ball = os.path.join(SAVE_DIR, f'frame_ball_flight_area_meta.json')
    if os.path.exists(meta_path_ball):
        with open(meta_path_ball) as f:
            meta = json.load(f)
            ball_rect = meta['rect']
            x, y, w, h = map(int, ball_rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Red box
    # Draw rim if exists
    meta_path_rim = os.path.join(SAVE_DIR, f'frame_rim_meta.json')
    if os.path.exists(meta_path_rim):
        with open(meta_path_rim) as f:
            meta = json.load(f)
            rim_rect = meta['rect']
            x, y, w, h = map(int, rim_rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box
    # Always draw the current rect
    x, y, w, h = map(int, rect)
    color = (255, 0, 0) if rect_type == 'ball_flight_area' else (0, 255, 0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    img_name = f'frame_{rect_type}.jpg'
    img_path = os.path.join(SAVE_DIR, img_name)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(img_path, frame_bgr)

    meta = {'frame_idx': frame_idx, 'rect_type': rect_type, 'rect': rect}
    meta_path = os.path.join(SAVE_DIR, f'frame_{rect_type}_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {img_path} and {meta_path}")

class FrameLabeler:
    """
    Interactive tool for drawing and saving two rectangles independently per frame.
    """
    def __init__(self, frames):
        self.frames = frames
        self.idx = 0
        self.curr_rects = {'ball_flight_area': None, 'rim': None}
        self.drawing_type = None  # Which rect are we drawing now?
        self.start = None
        self.rect_patch = {'ball_flight_area': None, 'rim': None, 'temp': None}

        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.23)
        self.img = self.ax.imshow(self.frames[self.idx])
        self.ax.set_title(f'Frame {self.idx+1}/{len(self.frames)}')

        # Prev/Next buttons
        axprev = plt.axes([0.05, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.16, 0.05, 0.1, 0.075])
        self.bprev = Button(axprev, 'Prev')
        self.bnext = Button(axnext, 'Next')
        self.bprev.on_clicked(self.prev_frame)
        self.bnext.on_clicked(self.next_frame)
        # Draw Ball Flight Area button
        axdraw1 = plt.axes([0.35, 0.05, 0.23, 0.075])
        self.bdraw1 = Button(axdraw1, 'Draw Ball Flight Area')
        self.bdraw1.on_clicked(self.set_draw_ball)
        # Draw Rim button
        axdraw2 = plt.axes([0.60, 0.05, 0.15, 0.075])
        self.bdraw2 = Button(axdraw2, 'Draw Rim')
        self.bdraw2.on_clicked(self.set_draw_rim)

        # Event handlers
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.cancel_on_right)

        self.load_rects()
        self.update_display()
        plt.show()

    def set_draw_ball(self, event):
        self.drawing_type = 'ball_flight_area'
        self.ax.set_title(f'Frame {self.idx+1}/{len(self.frames)} - Drawing ball_flight_area (drag mouse)')
        self.fig.canvas.draw_idle()

    def set_draw_rim(self, event):
        self.drawing_type = 'rim'
        self.ax.set_title(f'Frame {self.idx+1}/{len(self.frames)} - Drawing rim (drag mouse)')
        self.fig.canvas.draw_idle()

    def prev_frame(self, event):
        if self.idx > 0:
            self.idx -= 1
            self.drawing_type = None
            self.load_rects()
            self.update_display()

    def next_frame(self, event):
        if self.idx < len(self.frames) - 1:
            self.idx += 1
            self.drawing_type = None
            self.load_rects()
            self.update_display()

    def load_rects(self):
        for rect_type in ['ball_flight_area', 'rim']:
            meta_path = os.path.join(
                SAVE_DIR, f'frame_{rect_type}_meta.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                    self.curr_rects[rect_type] = meta['rect']
            else:
                self.curr_rects[rect_type] = None

    def update_display(self):
        self.img.set_data(self.frames[self.idx])
        # Remove previous patches
        for rect_type in ['ball_flight_area', 'rim', 'temp']:
            if self.rect_patch[rect_type]:
                self.rect_patch[rect_type].remove()
                self.rect_patch[rect_type] = None
        self.ax.set_title(f'Frame {self.idx+1}/{len(self.frames)}')
        # Draw loaded rectangles
        colors = {'ball_flight_area': 'r', 'rim': 'g'}
        for rect_type in ['ball_flight_area', 'rim']:
            rect = self.curr_rects[rect_type]
            if rect:
                patch = Rectangle(
                    (rect[0], rect[1]), rect[2], rect[3],
                    linewidth=2, edgecolor=colors[rect_type], facecolor='none', alpha=0.8)
                self.rect_patch[rect_type] = self.ax.add_patch(patch)
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if self.drawing_type not in ['ball_flight_area', 'rim']:
            return
        if event.inaxes != self.ax or event.button != 1:
            return
        self.start = (event.xdata, event.ydata)
        # Remove temp patch if any
        if self.rect_patch['temp']:
            self.rect_patch['temp'].remove()
            self.rect_patch['temp'] = None

    def on_motion(self, event):
        if self.drawing_type not in ['ball_flight_area', 'rim']:
            return
        if self.start is None or event.inaxes != self.ax:
            return
        x0, y0 = self.start
        x1, y1 = event.xdata, event.ydata
        width, height = x1 - x0, y1 - y0
        if self.rect_patch['temp']:
            self.rect_patch['temp'].remove()
            self.rect_patch['temp'] = None
        color = 'r' if self.drawing_type == 'ball_flight_area' else 'g'
        patch = Rectangle((x0, y0), width, height, linewidth=2, edgecolor=color, facecolor='y', alpha=0.3)
        self.rect_patch['temp'] = self.ax.add_patch(patch)
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        if self.drawing_type not in ['ball_flight_area', 'rim']:
            self.start = None
            return
        if self.start is None or event.inaxes != self.ax or event.button != 1:
            self.start = None
            return
        x0, y0 = self.start
        x1, y1 = event.xdata, event.ydata
        rect = [min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)]
        self.curr_rects[self.drawing_type] = rect
        save_rect_metadata(self.idx, self.drawing_type, rect)
        print(f"Saved {self.drawing_type} rect: {rect}")
        self.drawing_type = None
        self.start = None
        self.update_display()

    def on_key(self, event):
        # Press ESC to cancel drawing mode
        if event.key == 'escape':
            self.drawing_type = None
            self.start = None
            self.ax.set_title(f'Frame {self.idx+1}/{len(self.frames)}')
            if self.rect_patch['temp']:
                self.rect_patch['temp'].remove()
                self.rect_patch['temp'] = None
            self.fig.canvas.draw_idle()

    def cancel_on_right(self, event):
        # Right-click cancels drawing
        if self.drawing_type and event.button == 3:
            self.drawing_type = None
            self.start = None
            self.ax.set_title(f'Frame {self.idx+1}/{len(self.frames)}')
            if self.rect_patch['temp']:
                self.rect_patch['temp'].remove()
                self.rect_patch['temp'] = None
            self.fig.canvas.draw_idle()

FrameLabeler(frames)
