# tooltip.py: dynamic tooltip class for tkinter
import tkinter as tk

class StaticToolTip:
    def __init__(self,widget,text=None):

        def on_enter(event):
            self.tooltip=tk.Toplevel()
            self.tooltip.overrideredirect(True)
            self.tooltip.geometry(f'+{event.x_root+15}+{event.y_root+10}')

            self.label=tk.Label(self.tooltip, text=self.text, background="#ffffe0", relief="solid", borderwidth=1,
                wraplength=728, justify="left")
            self.label.pack()

        def on_leave(event):
            self.tooltip.destroy()

        self.widget=widget
        self.text=text

        self.widget.bind('<Enter>',on_enter)
        self.widget.bind('<Leave>',on_leave)


class DynamicToolTip(object):
    def __init__(self, widget, children_to_bind, fill_tooltip_callback):
        self.widget = widget
        self.tip_window = None
        self.id = None
        self.x = self.y = 0
        self.fill_tooltip_callback = fill_tooltip_callback

        for child in children_to_bind:
            child.bind('<Leave>', self.leave)
            child.bind('<Motion>', self.motion)

    def leave(self, event):
        self.hide_tip()

    def motion(self, event):
        if self.tip_window:
            self.hide_tip()

        self.show_tip(event.x_root, event.y_root, event)

    def show_tip(self, x, y, event):
        "Display text in a tooltip window"
        x += 20
        y += 20        
        
        # create a top level window to hold the tooltip control(s)
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        
        result = self.fill_tooltip_callback(event, self.tip_window)
        if isinstance(result, str):
            # fallback to default tooltip
            label = tk.Label(tw, text=result, background="#ffffe0", relief="solid", borderwidth=1,
                wraplength = 180)
            label.pack(ipadx=1)

    def hide_tip(self):
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()

   